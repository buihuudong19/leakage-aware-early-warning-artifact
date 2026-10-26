"""Summary tables of the robustness analyses (written to outputs/tables/robustness_*.csv)."""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from .inference import paired_summary, holm

F4 = ["R", "NP", "G", "P"]
MILESTONE_ORDER = ["purchase", "approval", "t0", "t0+1d", "t0+2d", "t0+3d", "t0+7d", "delivery"]


_SFX = ""          # "_quick" when the summary is built from a smoke-test run


def _q(path):
    """Path of a combined robustness file, with the suffix of a smoke-test run when there is one."""
    if not _SFX:
        return path
    stem, ext = path.name.split(".", 1)
    return path.with_name(f"{stem}{_SFX}.{ext}")


def _read(path):
    path = _q(path)
    return pd.read_csv(path) if path.is_file() else None


def _ps(d, cfg):
    bs = cfg["bootstrap"]
    return paired_summary(np.asarray(d, float), bs["block_len"], bs["n_boot"], cfg["seed"])


def shapley_shares(mean_by_condition):
    """Shapley, standalone and last-in shares of the combined inflation, from mean PR-AUC by condition."""
    pm = mean_by_condition

    def v(S):
        S = set(S)
        if "R" in S:
            S.discard("NP")
        return pm["+".join(f for f in F4 if f in S) or "none"] - pm["none"]

    tot = v(F4); out = {}
    for i in F4:
        others = [f for f in F4 if f != i]; s = 0.0
        for r in range(4):
            for S in itertools.combinations(others, r):
                s += math.factorial(r) * math.factorial(3 - r) / 24 * (v(list(S) + [i]) - v(S))
        out[i] = dict(shapley=s / tot, standalone=v([i]) / tot, last_in=(tot - v([f for f in F4 if f != i])) / tot)
    for players in (["R", "G", "P"], ["NP", "G", "P"]):
        n = len(players); t3 = v(players)
        for i in players:
            others = [f for f in players if f != i]; s = 0.0
            for r in range(n):
                for S in itertools.combinations(others, r):
                    s += math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (v(list(S) + [i]) - v(S))
            out[i]["shapley_" + "".join(players)] = s / t3
    return tot, out


def seeds_table(cfg, out, rd):
    tm = _read(rd / "tuned_metrics.csv"); tb = pd.read_csv(out / "tables" / "rq1b_tuned_window_metrics.csv")
    main = tb.assign(seed=cfg["seed"], variant="main")
    if tm is not None:
        main = pd.concat([main, tm[(tm.variant == "main") & tm.shortcuts.isin(["none", "R"])]], ignore_index=True)
    lr_none = main[(main.model == "lr") & (main.shortcuts == "none") & (main.seed == cfg["seed"])].set_index("k").ap
    rows = []
    for s in sorted(main.seed.unique()):
        for sc in ("none", "R"):
            x = main[(main.model == "xgb") & (main.shortcuts == sc) & (main.seed == s)].set_index("k").ap
            l = lr_none if sc == "none" else main[(main.model == "lr") & (main.shortcuts == "R") & (main.seed == s)].set_index("k").ap
            if len(x) == 0 or len(l) == 0:
                continue
            d = (x - l).dropna()
            rows.append(dict(seed=s, condition=sc, ap_xgb=x.mean(), ap_lr=l.mean(), xgb_wins=float((d > 0).mean()),
                             **_ps(d.to_numpy(), cfg)))
    S = pd.DataFrame(rows)
    S.to_csv(out / "tables" / "robustness_seeds.csv", index=False)
    # variation across seeds: each seed contributes its mean window-level difference
    agg = []
    for sc, g in S.groupby("condition"):
        x = g["mean"].to_numpy(); n = len(x); sd = float(x.std(ddof=1)) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
        agg.append(dict(condition=sc, n_seeds=n, mean=float(x.mean()), sd=sd, ci_lo=float(x.mean()) - half,
                        ci_hi=float(x.mean()) + half, min=float(x.min()), max=float(x.max()),
                        n_negative=int((x < 0).sum()), n_ci_below_0=int((g.ci_hi < 0).sum()),
                        n_ci_above_0=int((g.ci_lo > 0).sum())))
    pd.DataFrame(agg).to_csv(out / "tables" / "robustness_seeds_summary.csv", index=False)
    return S


def nested_factorial_table(cfg, out, rd):
    tm = _read(rd / "tuned_metrics.csv"); tb = pd.read_csv(out / "tables" / "rq1b_tuned_window_metrics.csv")
    g_all = tb.copy()
    if tm is not None:
        g_all = pd.concat([g_all, tm[(tm.variant == "main") & (tm.seed == cfg["seed"])]], ignore_index=True)
    fixed = pd.read_csv(out / "tables" / "rq1_window_metrics.csv")
    rows, shares = [], []
    for setting, df in (("fixed configuration", fixed[~fixed.shortcuts.isin(["Graw", "Gloo"])]), ("nested tuning", g_all)):
        for m, g in df.groupby("model"):
            piv = g.pivot_table(index="k", columns="shortcuts", values="ap")
            for sc in piv.columns:
                if sc != "none":
                    d = (piv[sc] - piv["none"]).dropna()
                    rows.append(dict(setting=setting, model=m, condition=sc, ref_ap=piv["none"].mean(),
                                     rel_pct=100 * d.mean() / piv["none"].mean(), **_ps(d.to_numpy(), cfg)))
            if set(["none", "R", "NP", "G", "P", "R+G+P"]).issubset(piv.columns) and piv.shape[1] >= 12:
                tot, sh = shapley_shares(piv.mean())
                for f, v in sh.items():
                    shares.append(dict(setting=setting, model=m, shortcut=f, total_gain=tot, **v))
    pd.DataFrame(rows).to_csv(out / "tables" / "robustness_nested_inflation.csv", index=False)
    pd.DataFrame(shares).to_csv(out / "tables" / "robustness_shapley_rules.csv", index=False)


def serving_table(cfg, out, rd):
    sv = _read(rd / "serving_metrics.csv")
    if sv is None:
        return
    R0 = pd.read_csv(out / "tables" / "rq1_window_metrics.csv"); rows = []
    for (m, v, rl), g in sv.groupby(["model", "variant", "rule"]):
        ref = R0[(R0.model == m) & (R0.shortcuts == "none")].set_index("k").ap
        d = (g.set_index("k").ap - ref).dropna()
        rows.append(dict(model=m, pipeline=v, rule=rl, realised_ap=g.ap.mean(), ref_ap=ref.mean(),
                         rel_pct=100 * d.mean() / ref.mean(), **_ps(d.to_numpy(), cfg)))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby("model").groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"])
    df.to_csv(out / "tables" / "robustness_serving_rules.csv", index=False)


def milestone_tables(cfg, out, rd):
    base_m = pd.read_csv(out / "tables" / "rq2_window_metrics.csv")
    base_s = pd.read_csv(out / "data" / "rq2_scores.csv.gz")
    em, es = _read(rd / "milestone_metrics.csv"), _read(rd / "milestone_scores.csv.gz")
    M = pd.concat([base_m] + ([em] if em is not None else []), ignore_index=True)
    S = pd.concat([base_s] + ([es] if es is not None else []), ignore_index=True)
    S = S.sort_values(["milestone", "model", "k", "score"], ascending=[True, True, True, False], kind="stable")
    base = M.set_index(["milestone", "model", "k"])[["n_base", "pos_base"]]
    budgets = cfg.get("robustness", {}).get("alert_budgets", [0.05, 0.10, 0.20])
    rows = []
    for (ms, m, k), g in S.groupby(["milestone", "model", "k"]):
        nb, pb = base.loc[(ms, m, k)]
        for b in budgets:
            rows.append(dict(milestone=ms, model=m, k=k, budget=b, early_catch=g.y.to_numpy()[:math.ceil(b * nb)].sum() / pb))
    EC = pd.DataFrame(rows)
    EC.to_csv(out / "tables" / "robustness_milestone_budget_windows.csv", index=False)
    order = [x for x in MILESTONE_ORDER if x in set(EC.milestone)]
    summ = EC.groupby(["model", "budget", "milestone"]).early_catch.mean().unstack()[order].reset_index()
    summ.to_csv(out / "tables" / "robustness_milestone_budgets.csv", index=False)
    rows = []
    for (m, b), g in EC.groupby(["model", "budget"]):
        piv = g.pivot_table(index="k", columns="milestone", values="early_catch")
        for o in order:
            if o != "t0" and o in piv.columns:
                rows.append(dict(model=m, budget=b, contrast=f"t0 - {o}", **_ps((piv["t0"] - piv[o]).dropna().to_numpy(), cfg)))
    C = pd.DataFrame(rows); C["p_holm"] = np.nan
    for _, idx in C.groupby(["model", "budget"]).groups.items():
        C.loc[idx, "p_holm"] = holm(C.loc[idx, "p"])
    C.to_csv(out / "tables" / "robustness_milestone_contrasts.csv", index=False)


def recalibration_tables(cfg, out, rd):
    r3 = _read(rd / "recal_metrics.csv")
    if r3 is None:
        return
    info = _read(rd / "recal_info.csv")
    cols = ["ap", "cal_intercept", "citl_diff", "cal_slope", "ece", "brier", "saving_r2", "saving_r5", "saving_r10"]
    r3.groupby(["model", "strategy"])[cols].mean().reset_index().to_csv(out / "tables" / "robustness_recal_strategies.csv", index=False)
    pairs = [("platt_ipw", "platt"), ("platt_ipw_strat", "platt_ipw"), ("platt_ipw_cap10", "platt_ipw"),
             ("platt_ipw_cap50", "platt_ipw"), ("platt_drop_young", "platt"), ("platt_em", "platt"),
             ("platt_ipw_em", "platt_ipw"), ("platt_ipw", "raw"), ("platt_ipw", "static"),
             ("platt_ipw", "platt_drop_young")]
    rows = []
    for m, g in r3.groupby("model"):
        for a, b in pairs:
            A, B = g[g.strategy == a].set_index("k"), g[g.strategy == b].set_index("k")
            for c in ["cal_intercept", "citl_diff", "ece", "brier", "saving_r5", "saving_r10"]:
                rows.append(dict(model=m, contrast=f"{a} - {b}", metric=c, **_ps((A[c] - B[c]).dropna().to_numpy(), cfg)))
    C = pd.DataFrame(rows); C["p_holm"] = np.nan
    for _, idx in C.groupby(["model", "contrast"]).groups.items():
        C.loc[idx, "p_holm"] = holm(C.loc[idx, "p"])
    C.to_csv(out / "tables" / "robustness_recal_contrasts.csv", index=False)
    info.groupby("model").mean(numeric_only=True).reset_index().to_csv(out / "tables" / "robustness_recal_base_rates.csv", index=False)
    pr = _read(rd / "recal_probs.csv.gz")
    strategies = ["raw", "static", "platt", "platt_ipw", "platt_ipw_strat", "platt_drop_young"]
    rows = []
    # r is the effective ratio e * r of Section 4.8, so the grid also covers a contact that averts the loss only with
    # probability e (for example e = 0.25 and r = 2 give 0.5)
    for r in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
        t = 1 / (1 + r)
        for (m, k), g in pr.groupby(["model", "k"]):
            y = g.y.to_numpy(); P = y.sum(); N = len(y) - P; best = min(r * P, N)
            for s in strategies:
                a = g[s].to_numpy() >= t; tp = (a & (y == 1)).sum(); fp = a.sum() - tp
                rows.append(dict(r=r, model=m, k=k, strategy=s, saving=1 - (fp + r * (P - tp)) / best))
    RG = pd.DataFrame(rows)
    RG.groupby(["model", "strategy", "r"]).saving.mean().reset_index().to_csv(out / "tables" / "robustness_cost_ratio_grid.csv", index=False)
    rows = []
    for (m, r), g in RG.groupby(["model", "r"]):
        piv = g.pivot_table(index="k", columns="strategy", values="saving")
        for b in ("platt", "raw", "static", "platt_drop_young"):
            rows.append(dict(model=m, r=r, contrast=f"platt_ipw - {b}", **_ps((piv["platt_ipw"] - piv[b]).to_numpy(), cfg)))
    pd.DataFrame(rows).to_csv(out / "tables" / "robustness_cost_ratio_contrasts.csv", index=False)
    residual_bias_by_stratum(cfg, out, pr)


def residual_bias_by_stratum(cfg, out, pr):
    """Calibration in the large (mean probability minus observed rate) within strata of delivery speed.

    Strata: terciles of the days left on the promise at t0 (known when the order is scored, the variable used by the
    stratified weights) and the realised delivery time (known only afterwards; a diagnostic for Assumption 1(i)).
    """
    F = pd.read_pickle(out / "data" / "features_t0.pkl")[["order_id", "f_remaining_promise_days", "p_transit_days"]]
    d = pr.merge(F, on="order_id", how="left")
    cuts = np.nanquantile(F["f_remaining_promise_days"], [1 / 3, 2 / 3])
    d["promise_tercile"] = np.digitize(d["f_remaining_promise_days"].fillna(np.nanmedian(F["f_remaining_promise_days"])), cuts)
    d["promise_tercile"] = d["promise_tercile"].map({0: "short", 1: "middle", 2: "long"})
    d["delivered_within_7d"] = np.where(d["p_transit_days"] <= 7, "yes", "no")
    strategies = [s for s in ["raw", "static", "platt", "platt_ipw", "platt_ipw_strat", "platt_drop_young"] if s in d]
    rows = []
    for col in ("promise_tercile", "delivered_within_7d"):
        for (m, z), g in d.groupby(["model", col]):
            kk = g.groupby("k")
            obs = kk.y.mean()
            for s in strategies:
                diff = (kk[s].mean() - obs).dropna()
                rows.append(dict(stratifier=col, stratum=z, model=m, strategy=s, n_orders=len(g), base_rate=float(g.y.mean()),
                                 **_ps(diff.to_numpy(), cfg)))
    pd.DataFrame(rows).to_csv(out / "tables" / "robustness_recal_residual_by_stratum.csv", index=False)


def sample_and_smoothing_tables(cfg, out, rd):
    fm, tm = _read(rd / "fixed_metrics.csv"), _read(rd / "tuned_metrics.csv")
    R0 = pd.read_csv(out / "tables" / "rq1_window_metrics.csv")
    if fm is None:
        return
    rows, shares = [], []
    neutral = fm[fm.variant == "neutral0"]
    for m, g in neutral.groupby("model"):
        piv = g.pivot_table(index="k", columns="shortcuts", values="ap")
        if "none" not in piv.columns:
            continue
        for sc in piv.columns:
            if sc != "none":
                d = (piv[sc] - piv["none"]).dropna()
                rows.append(dict(analysis="three-star as y = 0", model=m, condition=sc, ref_ap=piv["none"].mean(),
                                 rel_pct=100 * d.mean() / piv["none"].mean(), **_ps(d.to_numpy(), cfg)))
        if piv.shape[1] >= 12:
            tot, sh = shapley_shares(piv.mean())
            for f, v in sh.items():
                shares.append(dict(analysis="three-star as y = 0", model=m, shortcut=f, total_gain=tot, **v))
    for setting, df in (("fixed configuration", neutral), ("nested tuning", tm[tm.variant == "neutral0"] if tm is not None else None)):
        if df is None or len(df) == 0:
            continue
        for sc in sorted(set(df.shortcuts)):
            x = df[(df.model == "xgb") & (df.shortcuts == sc)].set_index("k").ap
            l = df[(df.model == "lr") & (df.shortcuts == sc)].set_index("k").ap
            if len(x) and len(l):
                rows.append(dict(analysis=f"three-star as y = 0, XGBoost - LR, {setting}", model="xgb - lr", condition=sc,
                                 **_ps((x - l).dropna().to_numpy(), cfg)))
    R0 = R0[R0.model.isin(["xgb", "lr"])]
    sm = pd.concat([fm[fm.variant.str.match(r"^(mu|lp)")], R0[R0.shortcuts.isin(["none", "G", "R+G+P"])].assign(variant="main")])
    for (v, m), g in sm.groupby(["variant", "model"]):
        piv = g.pivot_table(index="k", columns="shortcuts", values="ap")
        for sc in ("G", "R+G+P"):
            if "none" not in piv.columns or sc not in piv.columns:
                continue
            d = (piv[sc] - piv["none"]).dropna()
            rows.append(dict(analysis=f"smoothing / prior: {v}", model=m, condition=sc, ref_ap=piv["none"].mean(),
                             rel_pct=100 * d.mean() / piv["none"].mean(), **_ps(d.to_numpy(), cfg)))
    pd.DataFrame(rows).to_csv(out / "tables" / "robustness_sample_and_smoothing.csv", index=False)
    pd.DataFrame(shares).to_csv(out / "tables" / "robustness_neutral_shapley.csv", index=False)


def summarize(cfg, out, quick=False):
    global _SFX
    _SFX = "_quick" if quick else ""
    rd = out / "robustness"
    for fn in (seeds_table, nested_factorial_table, serving_table, milestone_tables, recalibration_tables,
               sample_and_smoothing_tables):
        fn(cfg, out, rd)
