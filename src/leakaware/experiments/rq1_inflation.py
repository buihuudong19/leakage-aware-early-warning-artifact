"""RQ1: how much do evaluation shortcuts inflate reported performance?

Design
  * Reference protocol: rolling origins on t0, purge on the label-availability time, as-of features at t0 only.
  * Four shortcuts in a factorial design:
      R  - random K-fold split instead of a temporal split
      NP - no purge (labels not yet available at the origin are used for training)
      G  - target aggregates (low-rating rates by seller, category, state, ...) computed over the full period
      P  - variables recorded after t0 (actual delivery time, delay against the promise)
    The purge does not apply to a random split, so the design has 8 temporal and 4 random conditions (12 in total).
  * Shapley decomposition of the combined inflation (all shortcuts versus the reference) by shortcut.
  * Two selection shortcuts: choosing the model configuration and the threshold on the evaluation windows.
"""
import itertools
import math
import time

import numpy as np
import pandas as pd

from ..data.features import feature_set
from ..evaluation.metrics import window_metrics, block_indices, moving_block_bootstrap_mean, best_f1_threshold
from ..evaluation.models import make_model
from ..evaluation.splits import rolling_windows, temporal_train_mask, eval_mask, random_folds
from sklearn.metrics import f1_score, precision_score, recall_score

METRICS = ["ap", "ap_norm", "auc", "p_at_10"]
FACTORS = ["R", "NP", "G", "P"]
FACTOR_LABEL = {"R": "Random split", "NP": "No label-availability purge", "G": "Full-period target aggregates",
                "P": "Post-handover features"}


def conditions():
    out = []
    for purge, agg, post in itertools.product([True, False], ["asof", "global"], [False, True]):
        out.append(dict(split="temporal", purge=purge, agg=agg, post=post))
    for agg, post in itertools.product(["asof", "global"], [False, True]):
        out.append(dict(split="random", purge=None, agg=agg, post=post))
    for c in out:
        S = set()
        if c["split"] == "random": S.add("R")
        if c["purge"] is False: S.add("NP")
        if c["agg"] == "global": S.add("G")
        if c["post"]: S.add("P")
        c["shortcuts"] = "+".join(f for f in FACTORS if f in S) or "none"
    return out


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_factorial(D, vocab, cfg, models=("xgb", "lr"), extra_conditions=(), fs=feature_set, only=None):
    """fs maps (agg, post) to (num, cat); the default is the feature set of the main task, the second task passes its own.
    `only` restricts the run to the listed conditions (used by the robustness analyses)."""
    W = rolling_windows(cfg)
    conds = conditions() + list(extra_conditions)
    if only is not None:
        conds = [c for c in conds if c["shortcuts"] in only]
    rows, preds = [], []
    fold = random_folds(D, cfg["random_split"]["n_folds"], cfg["seed"])
    for mname in models:
        for c in conds:
            num, cat = fs(c["agg"], c["post"])
            t = time.time()
            if c["split"] == "temporal":
                for w in W:
                    tr = temporal_train_mask(D, w["origin"], c["purge"]); ev = eval_mask(D, w)
                    m = make_model(mname, num, cat, vocab, cfg).fit(D[tr], D["y"].to_numpy()[tr])
                    s = m.predict_proba(D[ev])
                    r = window_metrics(D["y"].to_numpy()[ev], s)
                    rows.append(dict(model=mname, **c, k=w["k"], origin=w["origin"], n_train=int(tr.sum()), **r))
                    if c["shortcuts"] == "none":
                        preds.append(pd.DataFrame(dict(model=mname, k=w["k"], order_id=D["order_id"].to_numpy()[ev],
                                                       y=D["y"].to_numpy()[ev], score=s)))
            else:
                oof = np.empty(len(D))
                for f in range(fold.max() + 1):
                    tr, te = fold != f, fold == f
                    m = make_model(mname, num, cat, vocab, cfg).fit(D[tr], D["y"].to_numpy()[tr])
                    oof[te] = m.predict_proba(D[te])
                for w in W:
                    ev = eval_mask(D, w)
                    r = window_metrics(D["y"].to_numpy()[ev], oof[ev])
                    rows.append(dict(model=mname, **c, k=w["k"], origin=w["origin"], n_train=int((fold != 0).sum()), **r))
            _log(f"RQ1 {mname} {c['split']:8s} {c['shortcuts']:10s} done ({time.time()-t:.0f}s)")
    return pd.DataFrame(rows), (pd.concat(preds) if preds else None)


def summarize_inflation(R, cfg):
    bs = cfg["bootstrap"]
    out = []
    for mname, g in R.groupby("model"):
        ref = g[g["shortcuts"] == "none"].set_index("k")
        for sc, h in g.groupby("shortcuts"):
            h = h.set_index("k").loc[ref.index]
            for met in METRICS:
                d = (h[met] - ref[met]).to_numpy()
                mean, lo, hi, _ = moving_block_bootstrap_mean(d, bs["n_boot"], bs["block_len"], cfg["seed"])
                out.append(dict(model=mname, shortcuts=sc, metric=met, ref_mean=ref[met].mean(), cond_mean=h[met].mean(),
                                delta=mean, ci_lo=lo, ci_hi=hi, rel_pct=100 * mean / ref[met].mean(),
                                share_windows_inflated=float((d > 0).mean())))
    return pd.DataFrame(out)


def _value_table(g, met):
    """Window-level v(S) for every subset S of the four shortcuts (NP is void under R)."""
    vt = {}
    piv = g.pivot_table(index="k", columns="shortcuts", values=met)
    for r in range(len(FACTORS) + 1):
        for S in itertools.combinations(FACTORS, r):
            S = set(S)
            key = set(S)
            if "R" in key: key.discard("NP")
            name = "+".join(f for f in FACTORS if f in key) or "none"
            vt[frozenset(S)] = piv[name].to_numpy()
    return vt


def shapley(vt):
    n = len(FACTORS); phi = {}
    for i in FACTORS:
        tot = 0
        others = [f for f in FACTORS if f != i]
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot = tot + w * (vt[frozenset(S) | {i}] - vt[frozenset(S)])
        phi[i] = tot
    return phi


def shapley_decomposition(R, cfg):
    bs = cfg["bootstrap"]; rows = []
    for mname, g in R.groupby("model"):
        for met in METRICS:
            vt = _value_table(g, met)
            nW = len(next(iter(vt.values())))
            phi = shapley({k: v.mean() for k, v in vt.items()})
            idx = block_indices(nW, bs["n_boot"], bs["block_len"], cfg["seed"])
            boot = {f: [] for f in FACTORS}
            for b in idx:
                pb = shapley({k: v[b].mean() for k, v in vt.items()})
                for f in FACTORS: boot[f].append(pb[f])
            total = vt[frozenset(FACTORS)].mean() - vt[frozenset()].mean()
            for f in FACTORS:
                rows.append(dict(model=mname, metric=met, factor=f, label=FACTOR_LABEL[f], phi=phi[f],
                                 ci_lo=np.percentile(boot[f], 2.5), ci_hi=np.percentile(boot[f], 97.5),
                                 share_of_total=phi[f] / total if total else np.nan,
                                 v_none=vt[frozenset()].mean(), v_all=vt[frozenset(FACTORS)].mean()))
    return pd.DataFrame(rows)


SELECTION_GRID = (
    [("xgb", dict(max_depth=d, learning_rate=lr, n_estimators=n)) for d in (3, 5, 7) for lr, n in ((0.05, 300), (0.1, 600))]
    + [("lr", dict(C=C)) for C in (0.01, 0.1, 1.0, 10.0)]
    + [("rf", dict(max_depth=d, min_samples_leaf=20, n_estimators=200)) for d in (8, 16)]
)


def run_selection(D, vocab, cfg):
    """Train every candidate configuration under the reference protocol in every window (AP by window and configuration)."""
    W = rolling_windows(cfg)
    num, cat = feature_set("asof", False)
    rows = []
    for ci, (mname, p) in enumerate(SELECTION_GRID):
        t = time.time()
        for w in W:
            tr = temporal_train_mask(D, w["origin"], True); ev = eval_mask(D, w)
            m = make_model(mname, num, cat, vocab, cfg, **p).fit(D[tr], D["y"].to_numpy()[tr])
            s = m.predict_proba(D[ev])
            rows.append(dict(config=ci, model=mname, params=str(p), k=w["k"], **window_metrics(D["y"].to_numpy()[ev], s)))
        _log(f"Selection cfg {ci} {mname} {p} ({time.time()-t:.0f}s)")
    return pd.DataFrame(rows)


def selection_summary(S, cfg):
    bs = cfg["bootstrap"]; out = []
    for met in METRICS:
        piv = S.pivot_table(index="k", columns="config", values=met).sort_index()
        ks = piv.index.to_numpy()[1:]
        fwd = np.array([piv.loc[k, piv.loc[k - 1].idxmax()] for k in ks])         # chosen on the previous window
        orc = piv.loc[ks].max(1).to_numpy()                                         # chosen on the evaluation window itself
        board = piv.loc[ks, piv.loc[ks].mean().idxmax()].to_numpy()                 # leaderboard over all test windows
        for name, arr in (("oracle_per_window", orc), ("leaderboard_on_test", board)):
            mean, lo, hi, _ = moving_block_bootstrap_mean(arr - fwd, bs["n_boot"], bs["block_len"], cfg["seed"])
            out.append(dict(metric=met, shortcut=name, forward_mean=fwd.mean(), shortcut_mean=arr.mean(),
                            delta=mean, ci_lo=lo, ci_hi=hi, rel_pct=100 * mean / fwd.mean()))
    return pd.DataFrame(out)


def threshold_experiment(P, cfg):
    """F1 threshold chosen on the evaluation window (shortcut) versus the threshold carried over from the previous window."""
    bs = cfg["bootstrap"]; rows = []
    for mname, g in P.groupby("model"):
        ks = sorted(g["k"].unique())
        thr = {k: best_f1_threshold(g[g.k == k].y.to_numpy(), g[g.k == k].score.to_numpy()) for k in ks}
        for k in ks[1:]:
            h = g[g.k == k]; y, s = h.y.to_numpy(), h.score.to_numpy()
            t_or, f_or = thr[k]; t_fw = thr[k - 1][0]
            rows.append(dict(model=mname, k=k, thr_oracle=t_or, thr_forward=t_fw,
                             f1_oracle=f_or, f1_forward=f1_score(y, s >= t_fw, zero_division=0),
                             prec_forward=precision_score(y, s >= t_fw, zero_division=0),
                             rec_forward=recall_score(y, s >= t_fw, zero_division=0),
                             prec_oracle=precision_score(y, s >= t_or, zero_division=0),
                             rec_oracle=recall_score(y, s >= t_or, zero_division=0)))
    T = pd.DataFrame(rows)
    summ = []
    for mname, g in T.groupby("model"):
        mean, lo, hi, _ = moving_block_bootstrap_mean((g.f1_oracle - g.f1_forward).to_numpy(), bs["n_boot"], bs["block_len"], cfg["seed"])
        summ.append(dict(model=mname, f1_forward=g.f1_forward.mean(), f1_oracle=g.f1_oracle.mean(), delta=mean,
                         ci_lo=lo, ci_hi=hi, rel_pct=100 * mean / g.f1_forward.mean()))
    return T, pd.DataFrame(summ)


def run_deployment_gap(D, vocab, cfg, models=("xgb", "lr"), fs=feature_set, cols=None):
    """Realised performance of models trained through leaky pipelines when they are scored at t0.

    The model is trained as under the shortcut (full-period aggregates and/or post-t0 variables), but at scoring time
    only the admissible version of those columns exists: as-of aggregates replace full-period ones, and post-t0
    variables are missing.
    """
    if cols is None:
        from ..data.features import HIST_ASOF, HIST_GLOBAL, HIST_GLOBAL_RAW, POST
    else:
        HIST_ASOF, HIST_GLOBAL, HIST_GLOBAL_RAW, POST = (cols[k] for k in ("asof", "global", "global_raw", "post"))
    W = rolling_windows(cfg)
    variants = {"G": ("global", False), "Graw": ("global_raw", False), "P": ("asof", True), "G+P": ("global", True)}
    rows = []
    for mname in models:
        for vname, (agg, post) in variants.items():
            num, cat = fs(agg, post)
            t = time.time()
            for w in W:
                tr = temporal_train_mask(D, w["origin"], True); ev = eval_mask(D, w)
                m = make_model(mname, num, cat, vocab, cfg).fit(D[tr], D["y"].to_numpy()[tr])
                E = D[ev].copy()
                if agg in ("global", "global_raw"):
                    src = HIST_GLOBAL if agg == "global" else HIST_GLOBAL_RAW
                    for g_col, h_col in zip(src, HIST_ASOF):
                        E[g_col] = E[h_col].to_numpy()
                if post:
                    for c in POST:
                        E[c] = np.nan
                rows.append(dict(model=mname, variant=vname, k=w["k"], origin=w["origin"],
                                 **window_metrics(E["y"].to_numpy(), m.predict_proba(E))))
            _log(f"Deploy {mname} {vname} ({time.time()-t:.0f}s)")
    return pd.DataFrame(rows)


def deployment_summary(G, R, cfg):
    bs = cfg["bootstrap"]; out = []
    for mname, g in G.groupby("model"):
        ref = R[(R.model == mname) & (R.shortcuts == "none")].set_index("k")
        for v, h in g.groupby("variant"):
            h = h.set_index("k").loc[ref.index]
            rep = R[(R.model == mname) & (R.shortcuts == v.replace("Graw", "Graw"))].set_index("k")
            for met in METRICS:
                d = (h[met] - ref[met]).to_numpy()
                mean, lo, hi, _ = moving_block_bootstrap_mean(d, bs["n_boot"], bs["block_len"], cfg["seed"])
                out.append(dict(model=mname, variant=v, metric=met, correct_mean=ref[met].mean(),
                                reported_mean=rep[met].mean() if len(rep) else np.nan,
                                realized_mean=h[met].mean(), realized_minus_correct=mean, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def ranking_flip(R, cfg, a="xgb", b="lr"):
    """Paired window-level difference (a - b) under each condition: does a shortcut change which learner looks better?"""
    bs = cfg["bootstrap"]; out = []
    for sc, g in R.groupby("shortcuts"):
        pa = g[g.model == a].set_index("k"); pb = g[g.model == b].set_index("k")
        ks = pa.index.intersection(pb.index)
        for met in METRICS:
            d = (pa.loc[ks, met] - pb.loc[ks, met]).to_numpy()
            mean, lo, hi, _ = moving_block_bootstrap_mean(d, bs["n_boot"], bs["block_len"], cfg["seed"])
            out.append(dict(a=a, b=b, shortcuts=sc, metric=met, mean_a=pa.loc[ks, met].mean(), mean_b=pb.loc[ks, met].mean(),
                            diff=mean, ci_lo=lo, ci_hi=hi, a_wins_share=float((d > 0).mean())))
    return pd.DataFrame(out)


def learner_pairs(models):
    """Pairs compared in the learner analyses, in a fixed order: XGBoost - LR first, then the pairs with RF."""
    order = [("xgb", "lr"), ("rf", "lr"), ("xgb", "rf")]
    return [p for p in order if p[0] in models and p[1] in models]


def ranking_flips(R, cfg):
    return pd.concat([ranking_flip(R, cfg, a, b) for a, b in learner_pairs(set(R["model"]))], ignore_index=True)


# Tuning grids. The depth and penalty ranges extend beyond the values chosen most often, so that the selected
# configuration is rarely at the edge of the grid (tables *_tuned_complexity.csv report the shares).
TUNE_GRID = {
    "xgb": [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=mcw)
            for d in (1, 2, 3, 5, 8) for lr, n in ((0.05, 300), (0.03, 600)) for mcw in (5, 20)],
    "lr": [dict(C=C) for C in (1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
    "rf": [dict(max_depth=d, min_samples_leaf=leaf) for d in (2, 4, 8, 12, 20, None) for leaf in (10, 40)],
}
# The second task selected the deepest XGBoost trees and the weakest LR penalty in part of the windows, so its grid
# extends both ranges: depth 12, 16, 20 and unlimited (max_depth=0 with the hist method; trees then stop when
# min_child_weight binds), and C from 30 to 1000. With unlimited depth the upper end of the XGBoost grid is a natural
# limit, as for random forests in the main task.
TUNE_GRID_TASK2 = dict(TUNE_GRID,
                       xgb=[dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=mcw)
                            for d in (1, 2, 3, 5, 8, 12, 16, 20, 0) for lr, n in ((0.05, 300), (0.03, 600))
                            for mcw in (5, 20)],
                       lr=[dict(C=C) for C in (1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                                               300.0, 1000.0)])


def _inner_select(mname, Dtr, ytr, inner_splits, num, cat, vocab, cfg, grid=None):
    """Select the configuration with the highest mean PR-AUC over the inner validation splits (training data only)."""
    from sklearn.metrics import average_precision_score
    grid = grid or TUNE_GRID
    best, best_s = None, -1
    for p in grid[mname]:
        sc = []
        for itr, iva in inner_splits:
            if ytr[iva].sum() == 0:
                continue
            m = make_model(mname, num, cat, vocab, cfg, **p).fit(Dtr[itr], ytr[itr])
            sc.append(average_precision_score(ytr[iva], m.predict_proba(Dtr[iva])))
        s = float(np.mean(sc)) if sc else np.nan
        if s > best_s:
            best, best_s = p, s
    if best is None:                      # no valid inner split: fall back to the first grid entry
        best = grid[mname][0]
    return best, best_s


def run_tuned_ranking(D, vocab, cfg, conds=None, fs=feature_set, models=("xgb", "lr"), grid=None):
    """RQ1b: repeat the XGBoost / logistic regression comparison with nested hyper-parameter tuning.

    Reference protocol: at each origin the inner validation set is the last 28 days of labelled training data, and the
    inner training set uses only orders whose label was available before that period. Random split: a random
    three-fold inner split within each outer fold, as in the shortcut itself.
    """
    from sklearn.model_selection import StratifiedKFold
    conds = conds or [dict(split="temporal", purge=True, agg="asof", post=False, shortcuts="none"),
                      dict(split="random", purge=None, agg="asof", post=False, shortcuts="R"),
                      dict(split="random", purge=None, agg="global", post=True, shortcuts="R+G+P")]
    W = rolling_windows(cfg)
    y_all = D["y"].to_numpy()
    rows, chosen = [], []
    H = pd.Timedelta(days=28)
    for c in conds:
        num, cat = fs(c["agg"], c["post"])
        for mname in models:
            t = time.time()
            if c["split"] == "temporal":
                for w in W:
                    a = w["origin"]
                    purge = c.get("purge", True) is not False
                    tr = temporal_train_mask(D, a, purge); ev = eval_mask(D, w)
                    Dtr, ytr = D[tr].reset_index(drop=True), y_all[tr]
                    itr = (Dtr["t0"] < a - H).to_numpy()
                    if purge:                     # the inner split applies the same purge at a - 28 days
                        itr = itr & (Dtr["label_available_ts"] < a - H).to_numpy()
                    iva = (Dtr["t0"] >= a - H).to_numpy()
                    p, s_in = _inner_select(mname, Dtr, ytr, [(itr, iva)], num, cat, vocab, cfg, grid)
                    m = make_model(mname, num, cat, vocab, cfg, **p).fit(Dtr, ytr)
                    rows.append(dict(model=mname, shortcuts=c["shortcuts"], k=w["k"], origin=a,
                                     **window_metrics(y_all[ev], m.predict_proba(D[ev]))))
                    chosen.append(dict(model=mname, shortcuts=c["shortcuts"], k=w["k"], params=str(p), inner_ap=s_in))
            else:
                fold = random_folds(D, cfg["random_split"]["n_folds"], cfg["seed"])
                oof = np.empty(len(D))
                for f in range(fold.max() + 1):
                    tr, te = fold != f, fold == f
                    Dtr, ytr = D[tr].reset_index(drop=True), y_all[tr]
                    skf = StratifiedKFold(3, shuffle=True, random_state=cfg["seed"])
                    splits = [(np.isin(np.arange(len(ytr)), a_), np.isin(np.arange(len(ytr)), b_))
                              for a_, b_ in skf.split(np.zeros(len(ytr)), ytr)]
                    p, s_in = _inner_select(mname, Dtr, ytr, splits, num, cat, vocab, cfg, grid)
                    oof[te] = make_model(mname, num, cat, vocab, cfg, **p).fit(Dtr, ytr).predict_proba(D[te])
                    chosen.append(dict(model=mname, shortcuts=c["shortcuts"], k=f, params=str(p), inner_ap=s_in))
                for w in W:
                    ev = eval_mask(D, w)
                    rows.append(dict(model=mname, shortcuts=c["shortcuts"], k=w["k"], origin=w["origin"],
                                     **window_metrics(y_all[ev], oof[ev])))
            _log(f"RQ1b tuned {mname} {c['shortcuts']} ({time.time()-t:.0f}s)")
    return pd.DataFrame(rows), pd.DataFrame(chosen)


def tuned_complexity(Ch, grid=None):
    """Complexity of the selected configurations: does the random split favour more complex models?"""
    grid_all = grid
    import ast
    rows = []
    for _, r in Ch.iterrows():
        p = ast.literal_eval(r["params"])
        rows.append(dict(model=r["model"], shortcuts=r["shortcuts"], **p))
    X = pd.DataFrame(rows)
    out = []
    for (m, sc), g in X.groupby(["model", "shortcuts"]):
        grid = pd.DataFrame((grid_all or TUNE_GRID)[m]).astype(float).fillna(np.inf)   # max_depth=None -> inf
        g = g.astype({p_: float for p_ in grid.columns}).fillna({p_: np.inf for p_ in grid.columns})
        if m == "xgb" and "max_depth" in grid:                     # XGBoost: max_depth=0 means unlimited depth
            grid["max_depth"] = grid["max_depth"].replace(0, np.inf); g["max_depth"] = g["max_depth"].replace(0, np.inf)
        for param in grid.columns:
            lo, hi = grid[param].min(), grid[param].max()
            for v, c in g[param].value_counts().sort_index().items():
                edge = bool(v in (lo, hi)) if grid[param].nunique() >= 3 else None   # two-level settings: no edge
                out.append(dict(model=m, shortcuts=sc, param=param, value=v, share=c / len(g), n=len(g), grid_edge=edge))
    return pd.DataFrame(out)
