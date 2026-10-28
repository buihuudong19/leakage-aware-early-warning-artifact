"""Summaries of the seed check (tools/extra_seeds.py) and the sample check (tools/sample_variants.py).

usage: python tools/summarise_extra.py OUTPUTS_DIR EXTRA_DIR [--allow-fewer-windows]

Seed check. For every seed found in EXTRA_DIR/tables (seed{S}_rq3_window_metrics.csv, seed{S}_rq2_window_metrics.csv)
and for the reported run (OUTPUTS_DIR/tables, seed 42), the XGBoost quantities on which RQ2 to RQ4 rest are computed
as means over the evaluation windows: early catch rate by milestone and its difference from hand-over, and, for
rolling Platt and latency-weighted Platt, the calibration intercept, calibration-in-the-large and the cost saving at
r = 5 and r = 10, with the weighted - unweighted differences. Output: tables/seed_check_by_seed.csv (one row per seed
and quantity) and tables/seed_check_summary.csv (mean, SD, minimum and maximum over the refitted seeds, the value of
the reported run, and whether the sign of the reported value holds for every seed).

Sample check. For every variant (sample_{name}_window_metrics.csv) the reference PR-AUC, the relative inflation of
post-hand-over features (P) and of all shortcuts (R+G+P) and the XGBoost - LR difference under the reference protocol
and the random split, with 95% moving-block bootstrap intervals over windows. Output: tables/sample_check_summary.csv.
For the variants with RQ2 and RQ3/RQ4 results (sample_{name}_rq2/rq3_window_metrics.csv), the quantities of the seed
check are computed for both learners. Output: tables/sample_check_rq24.csv (one row per variant, learner, quantity).

Every result file must cover the evaluation windows of the reported run; files of --quick test runs are refused
unless --allow-fewer-windows is given.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.config import load_config                                      # noqa: E402
from leakaware.evaluation.metrics import moving_block_bootstrap_mean          # noqa: E402

ARGS = [x for x in sys.argv[1:] if x != "--allow-fewer-windows"]
FEWER_OK = len(ARGS) < len(sys.argv) - 1
if len(ARGS) != 2:
    sys.exit(__doc__)
OUT, EX = Path(ARGS[0]) / "tables", Path(ARGS[1]) / "tables"
cfg = load_config(); bs = cfg["bootstrap"]
MIL = ["purchase", "approval", "t0", "t0+3d", "t0+7d", "delivery"]


def rq3_q(R, model="xgb"):
    R = R[R["model"] == model]
    g = R.groupby("strategy")[["cal_intercept", "citl_diff", "saving_r5", "saving_r10", "ap"]].mean()
    q = {"rq3_ap_raw": g.loc["raw", "ap"]}
    for s, lab in (("platt", "platt"), ("platt_ipw", "ipw")):
        for m in ("cal_intercept", "citl_diff", "saving_r5", "saving_r10"):
            q[f"rq3_{m}_{lab}"] = g.loc[s, m]
    for m in ("cal_intercept", "citl_diff", "saving_r5", "saving_r10"):
        q[f"rq3_{m}_ipw_minus_platt"] = g.loc["platt_ipw", m] - g.loc["platt", m]
    return q


def rq2_q(R, model="xgb"):
    R = R[R["model"] == model]
    p = R.pivot_table(index="k", columns="milestone", values="early_catch")
    q = {f"rq2_ec_{m}": p[m].mean() for m in MIL if m in p}
    for m in MIL:
        if m != "t0" and m in p:
            q[f"rq2_ec_t0_minus_{m}"] = (p["t0"] - p[m]).mean()
    q["rq2_t0_highest"] = float(all(q["rq2_ec_t0"] >= q[f"rq2_ec_{m}"] for m in MIL if m != "t0" and m in p))
    best = max(q.get("rq2_ec_t0", -1), q.get("rq2_ec_t0+3d", -1))
    q["rq2_t0_or_3d_highest"] = float(all(best >= q[f"rq2_ec_{m}"] for m in MIL if m in p))
    return q


def check_windows(R, f, ref_file):
    k, k_ref = R["k"].nunique(), pd.read_csv(OUT / ref_file, usecols=["k"])["k"].nunique()
    if k != k_ref and not FEWER_OK:
        sys.exit(f"{f.name} covers {k} windows, the reported run {k_ref} (a --quick test run?)")


rows = []
ref = {**rq3_q(pd.read_csv(OUT / "rq3_window_metrics.csv")), **rq2_q(pd.read_csv(OUT / "rq2_window_metrics.csv"))}
rows += [dict(seed="reported", quantity=k, value=v) for k, v in ref.items()]
seeds = sorted({int(re.match(r"seed(\d+)_", p.name).group(1)) for p in EX.glob("seed*_window_metrics.csv")})
for s in seeds:
    for part, fn in (("rq3", rq3_q), ("rq2", rq2_q)):
        f = EX / f"seed{s}_{part}_window_metrics.csv"
        if f.is_file():
            R = pd.read_csv(f)
            check_windows(R, f, f"{part}_window_metrics.csv")
            rows += [dict(seed=str(s), quantity=k, value=v) for k, v in fn(R).items()]
if seeds:
    B = pd.DataFrame(rows)
    B.to_csv(EX / "seed_check_by_seed.csv", index=False)
    refit = B[B["seed"] != "reported"]
    S = refit.groupby("quantity")["value"].agg(n_seeds="count", mean="mean", sd="std", min="min", max="max")
    S["reported"] = pd.Series(ref)
    S["same_sign_all_seeds"] = [bool((np.sign(refit.loc[refit.quantity == q, "value"]) == np.sign(ref[q])).all())
                                for q in S.index]
    S.reset_index().to_csv(EX / "seed_check_summary.csv", index=False)
    print(S.round(4).to_string())


def ps(x):
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return m, lo, hi


summ = []
for f in sorted(p for p in EX.glob("sample_*_window_metrics.csv") if "_rq2_" not in p.name and "_rq3_" not in p.name):
    R = pd.read_csv(f)
    check_windows(R, f, "rq1_window_metrics.csv")
    piv = R.pivot_table(index=["model", "k"], columns="shortcuts", values="ap")
    row = dict(variant=R["variant"].iloc[0], n_windows=piv.loc["xgb"].shape[0])
    sizes = EX / f"sample_{row['variant']}_size.json"
    if sizes.is_file():
        row.update(json.loads(sizes.read_text(encoding="utf-8")))
    for mm in ("xgb", "lr"):
        p = piv.loc[mm]
        row[f"ref_ap_{mm}"] = p["none"].mean()
        for sc in ("P", "R+G+P"):
            row[f"infl_{sc}_{mm}_pct"] = 100 * (p[sc] - p["none"]).mean() / p["none"].mean()
    for sc in ("none", "R"):
        m, lo, hi = ps((piv.loc["xgb"][sc] - piv.loc["lr"][sc]).to_numpy())
        row.update({f"xgb_minus_lr_{sc}": m, f"xgb_minus_lr_{sc}_lo": lo, f"xgb_minus_lr_{sc}_hi": hi})
    summ.append(row)
if summ:
    T = pd.DataFrame(summ)
    T.to_csv(EX / "sample_check_summary.csv", index=False)
    print(T.set_index("variant").T.round(4).to_string())

rows24 = []
for f in sorted(EX.glob("sample_*_rq3_window_metrics.csv")) + sorted(EX.glob("sample_*_rq2_window_metrics.csv")):
    R = pd.read_csv(f)
    part = "rq3" if "_rq3_" in f.name else "rq2"
    check_windows(R, f, f"{part}_window_metrics.csv")
    for mod in ("xgb", "lr"):
        q = (rq3_q if part == "rq3" else rq2_q)(R, mod)
        rows24 += [dict(variant=R["variant"].iloc[0], model=mod, quantity=k, value=v) for k, v in q.items()]
if rows24:
    for mod in ("xgb", "lr"):                                   # the reported run, for comparison
        for part, fn in (("rq3", rq3_q), ("rq2", rq2_q)):
            q = fn(pd.read_csv(OUT / f"{part}_window_metrics.csv"), mod)
            rows24 += [dict(variant="reported", model=mod, quantity=k, value=v) for k, v in q.items()]
    T24 = pd.DataFrame(rows24)
    T24.to_csv(EX / "sample_check_rq24.csv", index=False)
    print(T24.pivot_table(index="quantity", columns=["variant", "model"], values="value").round(4).to_string())
