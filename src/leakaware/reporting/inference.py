"""Paired inference tables for the article: block-bootstrap intervals, Holm-adjusted p-values within each family of
comparisons, and the sensitivity of the intervals to the block length.

The p-value of a paired difference is twice the share of bootstrap replicate means on the far side of zero. As a check
that does not resample, the tables also give a Newey-West (HAC) t-test of the same mean difference (column p_hac).
Families: RQ1 inflation (each condition versus the reference, by learner), RQ1 realised performance, RQ1 learner
comparisons (fixed and tuned separately, by pair of learners), RQ2 milestone contrasts, and RQ3 recalibration contrasts (by learner and
comparison).
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT_P = 20000
BLOCKS = (1, 2, 3, 5, 6)


def boot_means(x, block, n_boot=2000, seed=42):
    x = np.asarray(x, float); n = len(x); block = min(block, n)
    rng = np.random.default_rng(seed); nb = math.ceil(n / block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return x[(st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]].mean(1)


def paired_summary(d, block=3, n_boot=2000, seed=42):
    """Mean, 95% percentile interval and bootstrap p-value of a paired window-level difference."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 3:                        # too few windows for an interval (smoke tests only)
        return dict(mean=float(d.mean()) if len(d) else np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    m = boot_means(d, block, n_boot, seed)
    mp = boot_means(d, block, N_BOOT_P, seed)
    p = max(2 * min((mp <= 0).mean(), (mp >= 0).mean()), 1 / N_BOOT_P)
    return dict(mean=float(d.mean()), ci_lo=float(np.percentile(m, 2.5)), ci_hi=float(np.percentile(m, 97.5)), p=float(p))


def hac_p(d, lag=2):
    """Two-sided p-value of a Newey-West t-test that the mean of an autocorrelated sequence is zero (Bartlett kernel,
    t distribution with n - 1 degrees of freedom); lag 2 matches the block length of 3 windows."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); v = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    if v <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(d.mean()) / math.sqrt(v / n), n - 1))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def _row(family, model, contrast, d, cfg, blocks=True):
    bs = cfg["bootstrap"]
    r = dict(family=family, model=model, contrast=contrast, **paired_summary(d, bs["block_len"], bs["n_boot"], cfg["seed"]))
    r["p_hac"] = hac_p(d, bs["block_len"] - 1)
    if blocks:
        for b in BLOCKS:
            mm = boot_means(d, b, bs["n_boot"], cfg["seed"])
            r[f"ci_lo_block{b}"], r[f"ci_hi_block{b}"] = np.percentile(mm, 2.5), np.percentile(mm, 97.5)
    return r


def _adjust(df, by):
    df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"])
    return df


def paired_contrasts(cfg, out):
    t = out / "tables"
    R = pd.read_csv(t / "rq1_window_metrics.csv")
    rows = []
    for m, g in R.groupby("model"):
        piv = g.pivot_table(index="k", columns="shortcuts", values="ap")
        for sc in piv.columns:
            if sc != "none":
                rows.append(_row("RQ1 inflation, PR-AUC (fixed configuration)", m, f"{sc} - reference",
                                 (piv[sc] - piv["none"]).to_numpy(), cfg))
    G = pd.read_csv(t / "rq1_deployment_windows.csv")
    for m, g in G.groupby("model"):
        ref = R[(R.model == m) & (R.shortcuts == "none")].set_index("k")["ap"]
        for v, h in g.groupby("variant"):
            rows.append(_row("RQ1 realised performance, PR-AUC", m, f"{v} realised - reference",
                             (h.set_index("k")["ap"] - ref).to_numpy(), cfg))
    from ..experiments.rq1_inflation import learner_pairs
    for name, df in (("fixed configuration", R), ("nested tuning", pd.read_csv(t / "rq1b_tuned_window_metrics.csv"))):
        for ma, mb in learner_pairs(set(df.model)):
            for sc, g in df.groupby("shortcuts"):
                a = g[g.model == ma].set_index("k")["ap"]; b = g[g.model == mb].set_index("k")["ap"]
                if len(a) and len(b):
                    rows.append(_row(f"RQ1 learner comparison, {name}", f"{ma} - {mb}", sc, (a - b).dropna().to_numpy(), cfg))
    Q = pd.read_csv(t / "rq2_window_metrics.csv")
    for m, g in Q.groupby("model"):
        piv = g.pivot_table(index="k", columns="milestone", values="early_catch")
        for o in ["purchase", "approval", "t0+3d", "t0+7d", "delivery"]:
            rows.append(_row("RQ2 early catch rate", m, f"t0 - {o}", (piv["t0"] - piv[o]).to_numpy(), cfg))
    df = _adjust(pd.DataFrame(rows), ["family", "model"])
    df.to_csv(t / "supp_paired_contrasts.csv", index=False)

    W = pd.read_csv(t / "rq3_window_metrics.csv")
    rows = []
    metrics = ["cal_intercept", "citl_diff", "cal_slope", "ece", "brier", "saving_r2", "saving_r5", "saving_r10"]
    for m, g in W.groupby("model"):
        A = g[g.strategy == "platt_ipw"].set_index("k")
        for s in sorted(set(g.strategy) - {"platt_ipw"}):
            B = g[g.strategy == s].set_index("k")
            for c in metrics:
                rows.append(_row(f"RQ3 platt_ipw - {s}", m, c, (A[c] - B[c]).to_numpy(), cfg, blocks=False))
        P, S = g[g.strategy == "platt"].set_index("k"), g[g.strategy == "static"].set_index("k")
        for c in metrics:
            rows.append(_row("RQ3 platt - static", m, c, (P[c] - S[c]).to_numpy(), cfg, blocks=False))
    df3 = _adjust(pd.DataFrame(rows), ["family", "model"])
    df3.to_csv(t / "supp_rq3_contrasts.csv", index=False)
    return df, df3
