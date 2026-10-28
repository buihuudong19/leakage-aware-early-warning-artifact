"""RQ2 with strictly valid chronology: early catch rates on the orders whose training labels all precede their scoring time.

usage: python tools/rq2_chronology_check.py OUTPUTS_DIR EXTRA_DIR

At purchase and approval the training cut-off of window k is moved back by delta_m, the 95th percentile of
t0 - tau_m(i) (Eq. 11 of the article). For an evaluation order whose tau_m(i) lies before o_k - delta_m, some training
labels can postdate its scoring time. This check drops, in every window, the orders for which this happens at purchase
or at approval (for the other milestones tau_m(i) >= t0_i >= o_k, so no order is affected), and recomputes the early
catch rate of every milestone on the remaining cohort with the same budget share of 10%. The scores are those saved by
the rq2 stage (data/rq2_scores.csv.gz); no model is refitted. As a check of the computation, the early catch rates of
the full cohort must equal tables/rq2_window_metrics.csv.
Output: EXTRA_DIR/tables/rq2_chronology_windows.csv and rq2_chronology_summary.csv.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.config import load_config                                      # noqa: E402
from leakaware.evaluation.splits import rolling_windows                       # noqa: E402
from leakaware.evaluation.metrics import moving_block_bootstrap_mean          # noqa: E402
from leakaware.experiments.rq2_timing import tau, MILESTONES, DAY             # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
OUT, EXTRA = Path(sys.argv[1]), Path(sys.argv[2])
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
cfg = load_config(); bs = cfg["bootstrap"]
F = pd.read_pickle(OUT / "data" / "features_t0.pkl")
D = F[F["y"] >= 0].reset_index(drop=True)
S = pd.read_csv(OUT / "data" / "rq2_scores.csv.gz")
pub = pd.read_csv(OUT / "tables" / "rq2_window_metrics.csv")
W = rolling_windows(cfg)

# shift of the training cut-off, computed as in the rq2 stage
shift = {m: max(0.0, float(np.nanpercentile((D["t0"] - tau(D, m)) / DAY, 95))) for m in MILESTONES}
print("training cut-off shifts (days):", {m: round(v, 2) for m, v in shift.items()}, flush=True)

rows = []
for w in W:
    base = D[(D["t0"] >= w["origin"]) & (D["t0"] < w["end"])]
    valid = np.ones(len(base), bool)
    for m in ("purchase", "approval"):                   # zero shift from hand-over onwards
        valid &= (tau(base, m) >= w["origin"] - pd.Timedelta(days=shift[m])).to_numpy()
    ids_all, ids_ok = set(base["order_id"]), set(base.loc[valid, "order_id"])
    for cohort, ids in (("all", ids_all), ("valid", ids_ok)):
        P = int(base.loc[base["order_id"].isin(ids), "y"].sum()); n = len(ids)
        B = math.ceil(0.10 * n)
        for (m, mod), g in S[S["k"] == w["k"]].groupby(["milestone", "model"], sort=False):
            g = g[g["order_id"].isin(ids)]
            top = np.argsort(-g["score"].to_numpy(), kind="stable")[:B]       # same tie rule as the rq2 stage
            tp = int(g["y"].to_numpy()[top].sum())
            rows.append(dict(cohort=cohort, k=w["k"], milestone=m, model=mod, n=n, pos=P, alerts=B, tp=tp,
                             early_catch=tp / P if P else np.nan))
R = pd.DataFrame(rows)
R.to_csv(EXTRA / "tables" / "rq2_chronology_windows.csv", index=False)

# the full cohort must reproduce the reported early catch rates
chk = R[R.cohort == "all"].merge(pub, on=["milestone", "model", "k"], suffixes=("", "_pub"))
assert len(chk) == len(pub), "windows do not match tables/rq2_window_metrics.csv"
dmax = (chk["early_catch"] - chk["early_catch_pub"]).abs().max()
print(f"full cohort: largest difference from rq2_window_metrics.csv {dmax:.2e}", flush=True)
assert dmax < 1e-12, "the full cohort does not reproduce the reported early catch rates"


def ps(x):
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return m, lo, hi


V = R[R.cohort == "valid"]
A = R[R.cohort == "all"].drop_duplicates("k")
Vk = V.drop_duplicates("k")
summ = []
for mod in sorted(V.model.unique()):
    piv = V[V.model == mod].pivot_table(index="k", columns="milestone", values="early_catch")
    for m in MILESTONES:
        row = dict(model=mod, milestone=m, early_catch=piv[m].mean(),
                   share_orders_kept=Vk.n.sum() / A.n.sum(), share_low_rated_kept=Vk.pos.sum() / A.pos.sum())
        if m != "t0":
            d, lo, hi = ps((piv["t0"] - piv[m]).to_numpy())
            row.update(t0_minus=d, t0_minus_lo=lo, t0_minus_hi=hi, t0_better_share=float((piv["t0"] > piv[m]).mean()))
        summ.append(row)
T = pd.DataFrame(summ)
T.to_csv(EXTRA / "tables" / "rq2_chronology_summary.csv", index=False)
print(T.round(4).to_string(index=False))
