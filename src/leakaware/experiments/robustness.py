"""Robustness and sensitivity analyses that complement the main experiments.

Groups of tasks (raw results in outputs/robustness/parts, combined files in outputs/robustness/*.csv):
  seeds       learner comparison with nested tuning under the reference protocol and a random split, for extra seeds
  nested      the remaining nine factorial conditions with nested tuning (seed 42), for a tuned Shapley attribution
  serving     realised performance of models trained with post-hand-over features under four serving rules
  milestones  extra prediction milestones (t0 + 1 and 2 days), with scores saved for several alert budgets
  recal       recalibration variants: weight caps, stratified latency, dropping young orders, EM after calibration
  neutral     three-star orders counted as not low rated (y = 0)
  smoothing   alternative smoothing weights of the as-of rates and alternative priors of the seller late rate
Each task runs in a separate process, writes its result to disk and is skipped when the pipeline is resumed.
"""
import copy
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..data.features import feature_set, BASE_CAT, BASE_NUM, HIST_ASOF, HIST_GLOBAL, POST
from ..data.io import load_raw
from ..evaluation.metrics import window_metrics
from ..evaluation.models import make_model
from ..evaluation.splits import rolling_windows, temporal_train_mask, eval_mask
from . import rq1_inflation as rq1
from . import rq2_timing as rq2

GROUPS = ("seeds", "nested", "serving", "milestones", "recal", "neutral", "smoothing")
COND = {c["shortcuts"]: c for c in rq1.conditions()}
IN_MAIN_RUN = ("none", "R", "R+G+P")          # conditions already tuned in the main RQ1b stage
_G = {}                                       # per-process cache: configuration and data sets


# ----------------------------------------------------------------------------------------------- per-process data
def _init(cfg, quick):
    warnings.filterwarnings("ignore")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    _G["cfg"], _G["quick"] = cfg, quick


def _cfg(seed=None):
    cfg = copy.deepcopy(_G["cfg"])
    if seed is not None:
        cfg["seed"] = seed
    if _G["quick"]:
        cfg["rolling"]["step_days"] = 140      # three origins only, for a smoke test
    return cfg


def _features():
    if "F" not in _G:
        _G["F"] = pd.read_pickle(_G["cfg"]["out_dir"] / "data" / "features_t0.pkl")
    return _G["F"]


def _data(variant):
    """Analysis data for a variant: main sample, three-star orders as negatives, or an alternative feature file."""
    key = ("D", variant)
    if key not in _G:
        if variant == "main":
            F = _features(); D = F[F["y"] >= 0].reset_index(drop=True)
        elif variant == "neutral0":
            D = _features().copy(); D.loc[D["y"] < 0, "y"] = 0; D = D.reset_index(drop=True)
        else:
            F2 = pd.read_pickle(_G["cfg"]["out_dir"] / "robustness" / f"features_{variant}.pkl")
            D = F2[F2["y"] >= 0].reset_index(drop=True)
        _G[key] = (D, {c: sorted(D[c].unique()) for c in BASE_CAT})
    return _G[key]


# ----------------------------------------------------------------------------------------------- tasks
def task_fixed(variant, sc, mname, seed):
    D, vocab = _data(variant)
    R, _ = rq1.run_factorial(D, vocab, _cfg(seed), models=(mname,), only=[sc])
    return {"metrics": R.assign(variant=variant, seed=seed)}


def task_tuned(variant, sc, mname, seed):
    D, vocab = _data(variant)
    T, Ch = rq1.run_tuned_ranking(D, vocab, _cfg(seed), conds=[COND[sc]], models=(mname,))
    return {"metrics": T.assign(variant=variant, seed=seed), "chosen": Ch.assign(variant=variant, seed=seed)}


def task_serving(vname, mname):
    """Model trained with post-hand-over features (P or G+P), scored at t0 under four replacements of those features."""
    from xgboost import XGBRegressor, XGBClassifier
    D, vocab = _data("main"); cfg = _cfg(); y_all = D["y"].to_numpy()
    agg = "global" if vname == "G+P" else "asof"
    num, cat = feature_set(agg, True)
    adm = BASE_NUM + HIST_ASOF
    rows = []
    for w in rolling_windows(cfg):
        tr = temporal_train_mask(D, w["origin"], True); ev = eval_mask(D, w)
        m = make_model(mname, num, cat, vocab, cfg).fit(D[tr], y_all[tr])
        E0 = D[ev].copy()
        if agg == "global":
            for g_col, h_col in zip(HIST_GLOBAL, HIST_ASOF):
                E0[g_col] = E0[h_col].to_numpy()
        # expected value of each post-hand-over variable, learned on the training set from admissible features
        Xtr = D.loc[tr, adm + BASE_CAT].copy(); Xev = E0[adm + BASE_CAT].copy()
        for c_ in BASE_CAT:
            Xtr[c_] = pd.Categorical(Xtr[c_], categories=vocab[c_]); Xev[c_] = pd.Categorical(Xev[c_], categories=vocab[c_])
        expected = {}
        for c_ in POST:
            kw = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_estimators=300, max_depth=5,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=cfg["seed"], n_jobs=2)
            yt = D.loc[tr, c_].to_numpy(dtype=float); ok = np.isfinite(yt)
            if c_ == "p_late":
                expected[c_] = XGBClassifier(**kw).fit(Xtr[ok], yt[ok].astype(int)).predict_proba(Xev)[:, 1]
            else:
                expected[c_] = XGBRegressor(**kw).fit(Xtr[ok], yt[ok]).predict(Xev)
        med = {c_: float(np.nanmedian(D.loc[tr, c_])) for c_ in POST}
        rules = {
            "missing": {c_: np.full(ev.sum(), np.nan) for c_ in POST},
            "train_median": {c_: np.full(ev.sum(), med[c_]) for c_ in POST},
            "promise": {"p_delivery_days": E0["f_promised_days"].to_numpy(),
                        "p_transit_days": E0["f_remaining_promise_days"].to_numpy(),
                        "p_delay_vs_estimate": np.zeros(ev.sum()), "p_late": np.zeros(ev.sum())},
            "expected": expected,
        }
        for rname, vals in rules.items():
            E = E0.copy()
            for c_ in POST:
                E[c_] = vals[c_]
            rows.append(dict(model=mname, variant=vname, rule=rname, k=w["k"],
                             **window_metrics(E["y"].to_numpy(), m.predict_proba(E))))
    return {"metrics": pd.DataFrame(rows)}


def task_milestone(m, mname):
    if "raw" not in _G:
        _G["raw"], _ = load_raw(_G["cfg"]["data_dir"], verify=False)
    R, _, Ch, Sc = rq2.run_rq2(_features(), _G["raw"], _cfg(), models=(mname,), milestones=[m])
    return {"metrics": R, "chosen": Ch, "scores": Sc}


def _latency_strat(D, a, cal, y_all, strat_col="f_remaining_promise_days", horizon_days=90, clip=20.0):
    """1/G_{y,s}(age): latency distribution by class and by tercile of the days left on the promise at t0."""
    mat = ((D["t0"] < a - pd.Timedelta(days=horizon_days)) & (D["label_available_ts"] < a)).to_numpy()
    lag = ((D["label_available_ts"] - D["t0"]).dt.total_seconds() / 86400).to_numpy()
    age = ((a - D["t0"]).dt.total_seconds() / 86400).to_numpy()
    z = D[strat_col].to_numpy()
    cuts = np.nanquantile(z[mat], [1 / 3, 2 / 3])
    st = np.digitize(np.nan_to_num(z, nan=np.nanmedian(z[mat])), cuts)
    w = np.ones(cal.sum()); yc, agec, sc = y_all[cal], age[cal], st[cal]
    for cls in (0, 1):
        Lc = np.sort(lag[mat & (y_all == cls)])
        for s_ in (0, 1, 2):
            L = np.sort(lag[mat & (y_all == cls) & (st == s_)])
            if len(L) < 50:
                L = Lc
            sel = (yc == cls) & (sc == s_)
            G = np.searchsorted(L, agec[sel], side="right") / len(L)
            w[sel] = 1 / np.clip(G, 1 / clip, 1)
    return w


def task_recal(mname):
    """The nine strategies of the RQ3 stage (as a check) plus six variants; probabilities of every strategy are saved."""
    from .rq3_recalibration import Platt, Iso, em_prior, prior_shift, latency_weights, calib_stats, decisions
    D, vocab = _data("main"); cfg = _cfg()
    num, cat = feature_set("asof", False); y_all = D["y"].to_numpy()
    H28, H14, H56 = pd.Timedelta(days=28), pd.Timedelta(days=14), pd.Timedelta(days=56)
    rows, probs_all, info = [], [], []
    static = None
    for w in rolling_windows(cfg):
        a = w["origin"]
        known = (D["label_available_ts"] < a).to_numpy()
        trA = (D["t0"] < a - H28).to_numpy() & known
        cal = ((D["t0"] >= a - H28) & (D["t0"] < a)).to_numpy() & known
        full = (D["t0"] < a).to_numpy() & known
        unl = ((D["t0"] >= a - H14) & (D["t0"] < a)).to_numpy()
        ev = ((D["t0"] >= a) & (D["t0"] < w["end"])).to_numpy()
        sel = (D["t0"] < a - H28).to_numpy() & known
        Ds = D[sel].reset_index(drop=True); ys = y_all[sel]
        itr = ((Ds["t0"] < a - H56) & (Ds["label_available_ts"] < a - H56)).to_numpy()
        iva = (Ds["t0"] >= a - H56).to_numpy()
        p, _ = rq1._inner_select(mname, Ds, ys, [(itr, iva)], num, cat, vocab, cfg)
        mA = make_model(mname, num, cat, vocab, cfg, **p).fit(D[trA], y_all[trA])
        pA_cal, pA_ev, pA_unl = mA.predict_proba(D[cal]), mA.predict_proba(D[ev]), mA.predict_proba(D[unl])
        mF = make_model(mname, num, cat, vocab, cfg, **p).fit(D[full], y_all[full])
        pF_ev, pF_unl = mF.predict_proba(D[ev]), mF.predict_proba(D[unl])
        pi_tr = y_all[full].mean(); pi_hat = em_prior(pF_unl, pi_tr)
        if static is None:
            static = (mA, Platt().fit(pA_cal, y_all[cal]))
        yc = y_all[cal]
        wl = latency_weights(D, a, cal, y_all)
        wr = latency_weights(D, a, cal, y_all, window_days=180)
        pi_ipw_roll = float((wr * yc).sum() / wr.sum())
        rec = ((D["t0"] >= a - H14) & (D["t0"] < a)).to_numpy() & known
        wn = latency_weights(D, a, rec, y_all, window_days=180)
        pi_now = float((wn * y_all[rec]).sum() / wn.sum()) if rec.sum() >= 30 else pi_ipw_roll
        p_roll = Platt().fit(pA_cal, yc, wr)(pA_ev)
        w10 = latency_weights(D, a, cal, y_all, clip=10.0)
        w50 = latency_weights(D, a, cal, y_all, clip=50.0)
        ws = _latency_strat(D, a, cal, y_all)
        age_cal = ((a - D.loc[cal, "t0"]).dt.total_seconds() / 86400).to_numpy()
        old = age_cal >= 14
        pl_u = Platt().fit(pA_cal, yc); pi_c = float(yc.mean())
        pl_w = Platt().fit(pA_cal, yc, wl); pi_ipw = float((wl * yc).sum() / wl.sum())
        pi_em_u = em_prior(pl_u(pA_unl), pi_c); pi_em_w = em_prior(pl_w(pA_unl), pi_ipw)
        P = {"static": static[1](static[0].predict_proba(D[ev])), "raw": pF_ev,
             "platt_ipw": pl_w(pA_ev), "platt_ipw_roll": p_roll,
             "platt_ipw_nowcast": prior_shift(p_roll, pi_ipw_roll, pi_now),
             "platt": pl_u(pA_ev), "isotonic": Iso().fit(pA_cal, yc)(pA_ev),
             "prior_recent": prior_shift(pF_ev, pi_tr, pi_c), "prior_em": prior_shift(pF_ev, pi_tr, pi_hat),
             "platt_ipw_cap10": Platt().fit(pA_cal, yc, w10)(pA_ev),
             "platt_ipw_cap50": Platt().fit(pA_cal, yc, w50)(pA_ev),
             "platt_ipw_strat": Platt().fit(pA_cal, yc, ws)(pA_ev),
             "platt_drop_young": Platt().fit(pA_cal[old], yc[old])(pA_ev),
             "platt_em": prior_shift(pl_u(pA_ev), pi_c, pi_em_u),
             "platt_ipw_em": prior_shift(pl_w(pA_ev), pi_ipw, pi_em_w)}
        y = y_all[ev]
        for s, pr in P.items():
            rows.append(dict(model=mname, strategy=s, k=w["k"], n=int(ev.sum()), **calib_stats(y, pr),
                             **decisions(y, pr, None), ap=window_metrics(y, pr)["ap"]))
        mat = ((D["t0"] < a - pd.Timedelta(days=90)) & (D["label_available_ts"] < a)).to_numpy()
        ev_all = ((D["t0"] >= a - H28) & (D["t0"] < a)).to_numpy()
        info.append(dict(model=mname, k=w["k"], n_mature=int(mat.sum()), n_mature_pos=int((mat & (y_all == 1)).sum()),
                         n_cal=int(cal.sum()), n_cal_old=int(old.sum()), pi_cal=pi_c, pi_ipw=pi_ipw,
                         pi_ipw_strat=float((ws * yc).sum() / ws.sum()), pi_ipw_cap10=float((w10 * yc).sum() / w10.sum()),
                         pi_ipw_cap50=float((w50 * yc).sum() / w50.sum()), pi_em_platt=pi_em_u, pi_em_platt_ipw=pi_em_w,
                         pi_eventual_28d=float(y_all[ev_all].mean()), pi_eval=float(y.mean())))
        probs_all.append(pd.DataFrame(dict(model=mname, k=w["k"], order_id=D["order_id"].to_numpy()[ev], y=y,
                                           day=D.loc[ev, "t0"].dt.normalize().to_numpy(), **P)))
    return {"metrics": pd.DataFrame(rows), "probs": pd.concat(probs_all), "info": pd.DataFrame(info)}


# ----------------------------------------------------------------------------------------------- planning and running
def build_smoothing_variants(cfg):
    """Alternative feature files: other smoothing weights of the as-of rates and other priors of the seller late rate."""
    from ..data.sample import build_sample
    from ..data.features import build_features
    out = cfg["out_dir"]; rd = out / "robustness"; rd.mkdir(parents=True, exist_ok=True)
    rb = cfg.get("robustness", {})
    names = [f"mu{m}" for m in rb.get("smoothing_variants", [5, 50])] + \
            [f"lp{int(round(100 * p)):02d}" for p in rb.get("late_prior_variants", [0.04, 0.12])]
    todo = [v for v in names if not (rd / f"features_{v}.pkl").is_file()]
    if not todo:
        return names
    F = pd.read_pickle(out / "data" / "features_t0.pkl")
    raw = None
    for v in todo:
        t = time.time()
        if v.startswith("mu"):
            if raw is None:
                raw, _ = load_raw(cfg["data_dir"], verify=False)
            c2 = copy.deepcopy(cfg); c2["features"]["smoothing_m"] = int(v[2:])
            S, _, _ = build_sample(raw, c2)
            F2 = build_features(S, raw, c2)
        else:                         # the late rate is (s + 5 * 0.08) / (n + 5); swap the prior without a rebuild
            p = int(v[2:]) / 100.0; F2 = F.copy()
            for pre in ("h_", "gs_"):
                n = F2[f"{pre}seller_late_n"].to_numpy(); r = F2[f"{pre}seller_laterate"].to_numpy()
                s = r * (n + 5) - 5 * 0.08
                F2[f"{pre}seller_laterate"] = (s + 5 * p) / (n + 5)
        F2.to_pickle(rd / f"features_{v}.pkl")
        print(f"[{time.strftime('%H:%M:%S')}] Robustness: built features_{v}.pkl ({time.time() - t:.0f}s)", flush=True)
    return names


def plan(cfg, groups, smoothing_names=()):
    rb = cfg.get("robustness", {})
    seeds = rb.get("seeds", list(range(1, 11)))
    T = []
    if "seeds" in groups:
        for s in seeds:
            T += [("tuned", "main", "none", "xgb", s), ("tuned", "main", "R", "xgb", s), ("tuned", "main", "R", "lr", s)]
    if "nested" in groups:
        for sc in COND:
            if sc not in IN_MAIN_RUN:
                T += [("tuned", "main", sc, m, cfg["seed"]) for m in ("xgb", "lr")]
    if "milestones" in groups:
        for ms in rb.get("extra_milestones", ["t0+1d", "t0+2d"]):
            T += [("milestone", ms, m) for m in ("xgb", "lr")]
    if "recal" in groups:
        T += [("recal", "xgb"), ("recal", "lr")]
    if "neutral" in groups:
        T += [("fixed", "neutral0", sc, m, cfg["seed"]) for sc in COND for m in ("xgb", "lr")]
        T += [("tuned", "neutral0", sc, m, cfg["seed"]) for sc in ("none", "R") for m in ("xgb", "lr")]
    if "serving" in groups:
        T += [("serving", v, m) for v in ("P", "G+P") for m in ("xgb", "lr")]
    if "smoothing" in groups:
        T += [("fixed", v, sc, m, cfg["seed"]) for v in smoothing_names for sc in ("none", "G", "R+G+P") for m in ("xgb", "lr")]
    cost = {"tuned": 0, "milestone": 0, "recal": 0, "serving": 1, "fixed": 2}   # long tasks first
    return sorted(dict.fromkeys(T), key=lambda t: cost[t[0]])


def _key(t):
    return "__".join(str(x).replace("+", "p") for x in t)


def run_task(t):
    t0 = time.time()
    try:
        fn = {"tuned": task_tuned, "fixed": task_fixed, "serving": task_serving,
              "milestone": task_milestone, "recal": task_recal}[t[0]]
        res = fn(*t[1:])
        parts = _G["cfg"]["out_dir"] / "robustness" / ("parts_quick" if _G["quick"] else "parts")
        for name, df in res.items():
            df.to_pickle(parts / f"{_key(t)}__{name}.pkl")
        (parts / f"{_key(t)}.done").write_text(f"{time.time() - t0:.0f}")
        return t, time.time() - t0, None
    except Exception:
        return t, time.time() - t0, traceback.format_exc()


def collect(cfg, quick=False):
    """Combine the per-task results into one file per result type."""
    rd = cfg["out_dir"] / "robustness"
    parts = rd / ("parts_quick" if quick else "parts")
    groups = {}
    for p in sorted(parts.glob("*.pkl")):
        kind, name = p.stem.split("__")[0], p.stem.split("__")[-1]
        groups.setdefault(f"{kind}_{name}", []).append(pd.read_pickle(p))
    sfx = "_quick" if quick else ""
    for g, L in groups.items():
        df = pd.concat(L, ignore_index=True)
        if g in ("milestone_scores", "recal_probs"):
            df.to_csv(rd / f"{g}{sfx}.csv.gz", index=False, compression="gzip")
        else:
            df.to_csv(rd / f"{g}{sfx}.csv", index=False)
    return sorted(groups)


def _signature(cfg):
    """Fingerprint of everything a saved task depends on; saved tasks from a different set-up are not reused."""
    import hashlib, json
    keys = ("seed", "sample", "features", "rolling", "random_split", "models", "robustness")
    grid = {m: rq1.TUNE_GRID[m] for m in ("xgb", "lr")}          # the robustness analyses use these two learners
    blob = json.dumps({"cfg": {k: cfg.get(k) for k in keys}, "grid": grid}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_robustness(cfg, groups=GROUPS, workers=0, models=("xgb", "lr"), quick=False):
    rd = cfg["out_dir"] / "robustness"
    parts = rd / ("parts_quick" if quick else "parts"); parts.mkdir(parents=True, exist_ok=True)
    sig, sig_file = _signature(cfg), parts / "_signature.txt"
    if any(parts.glob("*.done")) and (not sig_file.is_file() or sig_file.read_text().strip() != sig):
        raise RuntimeError(f"{parts} holds results of a different configuration or tuning grid. Move that folder "
                           "away (or delete it) and rerun, so that old and new results are not mixed.")
    sig_file.write_text(sig)
    names = build_smoothing_variants(cfg) if "smoothing" in groups else []
    model_of = lambda t: t[3] if t[0] in ("tuned", "fixed") else t[-1]
    tasks = [t for t in plan(cfg, groups, names) if model_of(t) in models and not (parts / f"{_key(t)}.done").is_file()]
    nw = workers or max(1, (os.cpu_count() or 2) // 4)
    print(f"[{time.strftime('%H:%M:%S')}] Robustness: {len(tasks)} tasks, {nw} worker processes", flush=True)
    t_start, errors = time.time(), 0
    with ProcessPoolExecutor(max_workers=nw, initializer=_init, initargs=(cfg, quick)) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            t, sec, err = f.result()
            if err:
                errors += 1
                print(f"[ERROR] {t}\n{err}", flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Robustness {i}/{len(tasks)} done {t} ({sec / 60:.1f} min; "
                      f"elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
    collect(cfg, quick)
    if errors:
        raise RuntimeError(f"{errors} robustness task(s) failed; rerun the stage to retry them")
