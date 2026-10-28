"""Uncertainty bands for the net-benefit curves of Fig. 10 (article), from the saved RQ3 predictions.

usage: python tools/net_benefit_bands.py OUTPUTS_DIR EXTRA_DIR

Reads OUTPUTS_DIR/data/rq3_predictions.csv.gz (scores of every evaluation order, written by the rq3 stage) and
OUTPUTS_DIR/tables/rq3_decision_curve.csv. It first recomputes the pooled curves with the function of the pipeline and
stops if they differ from the released table. It then resamples blocks of three adjacent evaluation windows (the
scheme of every interval in the article, same seed and number of replicates) and recomputes the pooled net-benefit
gain over the better of alerting all orders or none in each replicate. Results go to EXTRA_DIR (tables/, figures/), so
the released outputs are not touched. No model is fitted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.evaluation.metrics import block_indices                    # noqa: E402
from leakaware.experiments import rq3_recalibration as rq3                # noqa: E402
from leakaware.reporting import figures as F                              # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
OUT, EXTRA = Path(sys.argv[1]), Path(sys.argv[2])
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
(EXTRA / "figures").mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))
bs = cfg["bootstrap"]

C = pd.read_csv(OUT / "data" / "rq3_predictions.csv.gz")
released = pd.read_csv(OUT / "tables" / "rq3_decision_curve.csv")

# 1. the pooled curves must equal the released table
again = pd.concat([rq3.decision_curve(C, m) for m in ("xgb", "lr")], ignore_index=True)
for d_ in (released, again):
    d_["threshold"] = d_["threshold"].round(10)
chk = released.merge(again, on=["model", "threshold", "strategy"], suffixes=("", "_again"))
assert len(chk) == len(released) == len(again), "threshold grid differs from the released table"
err = float((chk.net_benefit - chk.net_benefit_again).abs().max())
assert err < 1e-12, f"recomputed net benefit differs from the released table (max {err:.2e})"
print(f"pooled curves reproduce rq3_decision_curve.csv (max difference {err:.1e})")

# 2. window-level counts and block bootstrap of the pooled gain
T = np.round(np.linspace(0.02, 0.5, 49), 10)
STRAT = {"platt": "p_platt", "ipw": "p_ipw", "static": "p_static", "prior": "p_prior"}
rows = []
for m in ("xgb", "lr"):
    g = C[C.model == m]
    ks = sorted(g.k.unique())
    n = np.array([(g.k == k).sum() for k in ks], float)
    pos = np.array([g.y[g.k == k].sum() for k in ks], float)
    counts = {}
    for s, col in STRAT.items():
        tp = np.zeros((len(ks), len(T))); fp = np.zeros((len(ks), len(T)))
        for i, k in enumerate(ks):
            gk = g[g.k == k]; y = gk.y.to_numpy(); p = gk[col].to_numpy()
            a = p[:, None] >= T[None, :]
            tp[i] = (a & (y[:, None] == 1)).sum(0); fp[i] = (a & (y[:, None] == 0)).sum(0)
        counts[s] = (tp, fp)
    idx = block_indices(len(ks), bs["n_boot"], bs["block_len"], cfg["seed"])
    odds = T / (1 - T)

    def gain(sel, s):
        tp, fp = counts[s]
        N = n[sel].sum(); P = pos[sel].sum()
        nb = (tp[sel].sum(0) - fp[sel].sum(0) * odds) / N
        base = np.maximum(P / N - (1 - P / N) * odds, 0.0)
        return nb - base

    full = np.arange(len(ks))
    for s in STRAT:
        est = gain(full, s)
        reps = np.array([gain(sel, s) for sel in idx])
        lo, hi = np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)
        for j, t in enumerate(T):
            rows.append(dict(model=m, strategy=s, threshold=t, gain=est[j], ci_lo=lo[j], ci_hi=hi[j]))
    # paired difference latency-weighted minus rolling Platt, same replicates
    est = gain(full, "ipw") - gain(full, "platt")
    reps = np.array([gain(sel, "ipw") - gain(sel, "platt") for sel in idx])
    lo, hi = np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)
    for j, t in enumerate(T):
        rows.append(dict(model=m, strategy="ipw - platt", threshold=t, gain=est[j], ci_lo=lo[j], ci_hi=hi[j]))
B = pd.DataFrame(rows)
B.to_csv(EXTRA / "tables" / "rq3_decision_curve_bands.csv", index=False)

# the gain of the bands equals the released curves minus the better of all or none
for m in ("xgb", "lr"):
    r = released[released.model == m].pivot(index="threshold", columns="strategy", values="net_benefit")
    base = r["alert_all"].clip(lower=0)
    for s in ("platt", "ipw", "static", "prior"):
        b = B[(B.model == m) & (B.strategy == s)].set_index("threshold").gain
        assert float((b - (r[s] - base).reindex(b.index)).abs().max()) < 1e-12

# 3. Fig. 10 with bands, drawn with the style of the pipeline's figure module
F._style()
lab = {"platt": "Rolling Platt", "ipw": "Latency-weighted Platt", "static": "Frozen model + Platt"}
key = {"platt": "platt", "ipw": "platt_ipw", "static": "static"}
fig, axes = F._subplots(2, 1, figsize=(F.COL, 0.7 + 1.45 * 2), sharex=True, sharey=True)
for p, (ax, m) in enumerate(zip(axes, ("xgb", "lr"))):
    g = B[(B.model == m) & (B.threshold >= 0.04) & (B.threshold <= 0.30)]
    for s in ["static", "platt", "ipw"]:
        h = g[g.strategy == s]
        col, ls = F.STRAT_STYLE[key[s]]
        if s != "static":
            ax.fill_between(h.threshold, 100 * h.ci_lo, 100 * h.ci_hi, color=col, alpha=0.15, lw=0)
        ax.plot(h.threshold, 100 * h.gain, color=col, ls=ls, lw=1.1, label=lab[s])
    ax.axhline(0, color=F.INK, lw=0.6)
    for r_ in (10, 5, 3):
        ax.axvline(1 / (1 + r_), color=F.MUTED, lw=0.5, ls=":")
        ax.text(1 / (1 + r_), 1.0, f"r = {r_}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.8, color=F.INK2)
    ax.set_xlim(0.04, 0.30)
    ax.set_title(f"({'ab'[p]}) {F.MODEL_LABEL[m]}", pad=11); F._grid(ax, "y")
    ax.set_ylabel("Net benefit gain (points)")
axes[-1].set_xlabel("Probability threshold $t = 1/(1+r)$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=2)
F._save(fig, EXTRA, "fig13_rq3_net_benefit_bands")
print(f"bands written to {EXTRA / 'tables' / 'rq3_decision_curve_bands.csv'}; figure to {EXTRA / 'figures'}")
