"""Timing of the daily alert policy (RQ4): an end-of-day batch, an online rule and a comparison at equal volume.

usage: python tools/policy_timing_check.py OUTPUTS_DIR EXTRA_DIR

The daily policy of the article ranks all hand-overs of a day, so it can run only once the day is over. This script
uses the scores saved by the rq3 stage (data/rq3_predictions.csv.gz, rolling Platt, both learners) and the order
times of data/features_t0.pkl; no model is refitted.
  1. Reproduction: the daily top-10% alerts recomputed from the saved scores must equal tables/rq3_daily_policy.csv.
  2. End-of-day batch: alerts are sent when the hand-over day ends. Counts the low-rated orders whose review was already
     answered by then, which an end-of-day alert would reach too late.
  3. Online rule: an order is alerted at hand-over if its score reaches the 90th percentile of the scores of the
     hand-overs of the preceding seven days, which uses only scores known at the decision time. The first seven days
     of the first window have no history and are left out of all comparisons.
  4. Equal volume: in each window, a single threshold chosen in hindsight so that, over the days compared, it alerts
     as many orders as the daily rule; it shows how much of the difference between the daily rule and the fixed
     threshold is due to volume. It cannot be deployed.
Precision, recall and the daily alert share are pooled over days with at least 30 hand-overs, as in Fig. 11.
Output: EXTRA_DIR/tables/policy_timing_days.csv and policy_timing_summary.csv.
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

if len(sys.argv) != 3:
    sys.exit(__doc__)
OUT, EXTRA = Path(sys.argv[1]), Path(sys.argv[2])
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
cfg = load_config()
F = pd.read_pickle(OUT / "data" / "features_t0.pkl")
D = F[F["y"] >= 0].reset_index(drop=True)
C = pd.read_csv(OUT / "data" / "rq3_predictions.csv.gz")
pub = pd.read_csv(OUT / "tables" / "rq3_daily_policy.csv", parse_dates=["day"])
W = rolling_windows(cfg)
DAY = pd.Timedelta(days=1)

# order times for the saved scores: the rq3 stage writes, per learner and window, the evaluation orders in the order of D
parts = []
for mod in C["model"].unique():
    Cm = C[C["model"] == mod]
    for w in W:
        ev = ((D["t0"] >= w["origin"]) & (D["t0"] < w["end"])).to_numpy()
        c = Cm[Cm["k"] == w["k"]]
        assert len(c) == ev.sum() and (c["y"].to_numpy() == D.loc[ev, "y"].to_numpy()).all(), f"{mod} k={w['k']}"
        parts.append(pd.DataFrame(dict(model=mod, k=w["k"], y=c["y"].to_numpy(), p=c["p_platt"].to_numpy(),
                                       t0=D.loc[ev, "t0"].to_numpy(), answer=D.loc[ev, "first_answer_ts"].to_numpy())))
X = pd.concat(parts, ignore_index=True)
X["day"] = X["t0"].dt.normalize()

rows = []
for mod, Xm in X.groupby("model"):
    days = sorted(Xm["day"].unique())
    start = days[0] + 7 * DAY
    for (k, day), g in Xm.groupby(["k", "day"]):
        n = len(g); b = max(1, math.ceil(0.10 * n))
        top = g.sort_values("p", ascending=False).head(b)                  # same rule as the rq3 stage
        hist = Xm[(Xm["day"] >= day - 7 * DAY) & (Xm["day"] < day)]["p"]
        thr = np.quantile(hist, 0.90) if day >= start else np.nan
        on = g[g["p"] >= thr] if day >= start else g.iloc[0:0]
        late = top["answer"] < day + DAY                                    # answered before the end of the day
        rows.append(dict(model=mod, k=k, day=day, n=n, pos=int(g["y"].sum()), in_comparison=bool(day >= start),
                         topk_alerts=b, topk_tp=int(top["y"].sum()),
                         topk_tp_answered_same_day=int((top["y"].astype(bool) & late).sum()),
                         pos_answered_same_day=int((g["y"].astype(bool) & (g["answer"] < day + DAY)).sum()),
                         online_alerts=len(on), online_tp=int(on["y"].sum())))
R = pd.DataFrame(rows)

# 1. reproduction of the daily rule
chk = R.merge(pub, on=["model", "k", "day"], suffixes=("", "_pub"))
assert len(chk) == len(pub) == len(R), "days do not match tables/rq3_daily_policy.csv"
bad = (chk["topk_tp"] != chk["topk_tp_pub"]) | (chk["topk_alerts"] != chk["topk_alerts_pub"])
print(f"daily top-10% recomputed from the saved scores: {int(bad.sum())} of {len(chk)} days differ", flush=True)
assert not bad.any(), "the daily rule does not reproduce tables/rq3_daily_policy.csv"
R = R.merge(pub[["model", "k", "day", "thr_alerts", "thr_tp"]], on=["model", "k", "day"])

# 4. equal volume: per window, one threshold alerting as many orders as the daily rule (hindsight)
comp = R[R.in_comparison & (R.n >= 30)][["model", "k", "day"]]            # the days that enter the summary
eq = []
for (mod, k), g in X.merge(comp, on=["model", "k", "day"]).groupby(["model", "k"]):
    b = int(R[(R.model == mod) & (R.k == k) & R.in_comparison & (R.n >= 30)]["topk_alerts"].sum())
    top = g.sort_values("p", ascending=False).head(b)
    eq.append(top.groupby("day").agg(eqvol_alerts=("y", "size"), eqvol_tp=("y", "sum")).reset_index().assign(model=mod, k=k))
R = R.merge(pd.concat(eq), on=["model", "k", "day"], how="left").fillna({"eqvol_alerts": 0, "eqvol_tp": 0})
R.to_csv(EXTRA / "tables" / "policy_timing_days.csv", index=False)

summ = []
for mod, h in R[R.in_comparison & (R.n >= 30)].groupby("model"):
    a_ = R[R.model == mod]                                   # all evaluation days
    row = dict(model=mod, n_days=len(h), all_pos=int(a_.pos.sum()), all_pos_answered_same_day=int(a_.pos_answered_same_day.sum()),
               all_topk_tp=int(a_.topk_tp.sum()), all_topk_tp_answered_same_day=int(a_.topk_tp_answered_same_day.sum()),
               pos_answered_same_day=int(h.pos_answered_same_day.sum()), pos=int(h.pos.sum()),
               topk_tp_answered_same_day=int(h.topk_tp_answered_same_day.sum()))
    for rule in ("topk", "thr", "online", "eqvol"):
        a, tp = h[f"{rule}_alerts"], h[f"{rule}_tp"]
        share = a / h.n
        row.update({f"{rule}_share": a.sum() / h.n.sum(), f"{rule}_precision": tp.sum() / a.sum(),
                    f"{rule}_recall": tp.sum() / h.pos.sum(), f"{rule}_cv": share.std() / share.mean(),
                    f"{rule}_max_share": share.max()})
    summ.append(row)
T = pd.DataFrame(summ)
T.to_csv(EXTRA / "tables" / "policy_timing_summary.csv", index=False)
print(T.set_index("model").T.round(4).to_string())
