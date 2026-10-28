"""Sensitivity of RQ1 to the sample conditions that the article does not vary (delivered orders; a review exists).

usage: python tools/sample_variants.py DATA_DIR OUTPUTS_DIR EXTRA_DIR [--quick] [--variants main,undelivered,...]

Variants of the analysis sample, built with the released sample and feature code:
  main          the sample of the article (rebuilt as a check: its size must equal tables/sample_notes.json)
  undelivered   adds orders that were handed over to the carrier but never delivered (status 'shipped' or
                'canceled' after hand-over) and have a review; their label is real
  unreviewed0   bounding case: delivered orders without any review are added as satisfied (y = 0), with the label
                taken to arrive at delivery plus the median delivery-to-answer time of high ratings. This is an
                assumption, not a label: it shows how far the results could move if every silent customer were
                satisfied; the synthetic reviews also enter the as-of history features (seller, category
                and state rating rates) of later orders
  both          undelivered and unreviewed0 together
For each variant the RQ1 shortcut conditions none, NP, G, P, R and R+G+P are run with the fixed configurations for
XGBoost and LR (released function run_factorial), and the script reports the reference PR-AUC, the inflation of
post-hand-over features and of all shortcuts, and the XGBoost - LR difference under the reference protocol and the
random split. For every variant other than main, the RQ2 milestones and the RQ3/RQ4 recalibration strategies are also
run for both learners with the stage functions of the released code and the configurations that the reported run
selected in each window (tables/rq2_chosen_params.csv, tables/rq3_chosen_params.csv); --no-rq24 skips them.
Results go to EXTRA_DIR/tables; the released outputs are not touched. A part whose window file already exists is not
refitted, so an interrupted run can be restarted with the same command.
"""
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.config import load_config                                      # noqa: E402
from leakaware.data.io import load_raw                                        # noqa: E402
from leakaware.data.sample import build_sample                                # noqa: E402
from leakaware.data.features import build_features, BASE_CAT                  # noqa: E402
from leakaware.evaluation.metrics import moving_block_bootstrap_mean          # noqa: E402
from leakaware.experiments import rq1_inflation as rq1                        # noqa: E402
from leakaware.experiments import rq2_timing as rq2                           # noqa: E402
from leakaware.experiments import rq3_recalibration as rq3                    # noqa: E402
from leakaware.evaluation.splits import rolling_windows                       # noqa: E402

import argparse
_ap = argparse.ArgumentParser(usage=__doc__)
_ap.add_argument("data"); _ap.add_argument("out"); _ap.add_argument("extra")
_ap.add_argument("--quick", action="store_true")
_ap.add_argument("--variants", default="main,undelivered,unreviewed0,both")
_ap.add_argument("--no-rq24", action="store_true", help="run RQ1 only")
_a = _ap.parse_args()
DATA, OUT, EXTRA, QUICK = Path(_a.data), Path(_a.out), Path(_a.extra), _a.quick
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
cfg = load_config(); cfg["data_dir"] = DATA
if QUICK:
    cfg["rolling"]["step_days"] = 140
bs = cfg["bootstrap"]
raw0, _ = load_raw(DATA)
COND = ["none", "NP", "G", "P", "R", "R+G+P"]


def variant_raw(name):
    raw = {k: v.copy() for k, v in raw0.items()}
    o, r = raw["orders"], raw["reviews"]
    if name in ("undelivered", "both"):
        sel = (o["order_status"] != "delivered") & o["order_delivered_carrier_date"].notna()
        o.loc[sel, "order_status"] = "delivered"          # passes rule R1; the delivery time stays missing
    if name in ("unreviewed0", "both"):
        dl = o[(raw0["orders"]["order_status"] == "delivered") & o["order_delivered_carrier_date"].notna()
               & o["order_delivered_customer_date"].notna()]
        silent = dl[~dl["order_id"].isin(r["order_id"])]
        hi = r[r["review_score"] >= 4].merge(dl[["order_id", "order_delivered_customer_date"]], on="order_id")
        gap = (hi["review_answer_timestamp"] - hi["order_delivered_customer_date"]).median()
        add = pd.DataFrame({"review_id": "synthetic_" + silent["order_id"], "order_id": silent["order_id"],
                            "review_score": 5,
                            "review_creation_date": silent["order_delivered_customer_date"] + gap,
                            "review_answer_timestamp": silent["order_delivered_customer_date"] + gap})
        for c in r.columns:
            if c not in add.columns:
                add[c] = np.nan
        raw["reviews"] = pd.concat([r, add[r.columns]], ignore_index=True)
    return raw


W = rolling_windows(cfg)
P2 = {(r.milestone, r.model, r.k): ast.literal_eval(r.params)
      for r in pd.read_csv(OUT / "tables" / "rq2_chosen_params.csv").itertuples()}
P3 = {(r.model, r.k): ast.literal_eval(r.params) for r in pd.read_csv(OUT / "tables" / "rq3_chosen_params.csv").itertuples()}


def fixed_selector(key):
    """Replacement for the inner tuning: the configuration the reported run selected for this learner and window.
    The stages call it once per window and learner, in window order; a counter per learner gives the window."""
    count = {}

    def select(mname, *args):
        i = count.get(mname, 0); count[mname] = i + 1
        return key(mname, W[i]["k"]), float("nan")
    return select, count


def windows_ok(f):
    k = pd.read_csv(f, usecols=["k"])["k"].nunique()
    if k != len(W):
        sys.exit(f"{f} has {k} windows, this run has {len(W)}: move the file away (an earlier --quick run?) and restart")
    print(f"{f.name} exists ({k} windows), reused", flush=True)


def run_rq24(name, F, raw):
    f3 = EXTRA / "tables" / f"sample_{name}_rq3_window_metrics.csv"
    if f3.is_file():
        windows_ok(f3)
    else:
        rq3._inner_select, cnt = fixed_selector(lambda mname, k: P3[(mname, k)])
        R3, Dd3, _, _ = rq3.run_rq3(F, cfg, models=("xgb", "lr"))
        assert cnt == {"xgb": len(W), "lr": len(W)}, f"RQ3 selections {cnt} for {len(W)} windows"
        Dd3.assign(variant=name).to_csv(EXTRA / "tables" / f"sample_{name}_rq3_daily_policy.csv", index=False)
        R3.assign(variant=name).to_csv(f3, index=False)
    f2 = EXTRA / "tables" / f"sample_{name}_rq2_window_metrics.csv"
    if f2.is_file():
        windows_ok(f2)
    else:
        frames = []
        for m in rq2.MILESTONES:
            fm = EXTRA / "tables" / f"sample_{name}_rq2_part_{m}.csv"            # one file per milestone, for restarts
            if fm.is_file():
                windows_ok(fm); frames.append(pd.read_csv(fm)); continue
            rq2._inner_select, cnt = fixed_selector(lambda mname, k, _m=m: P2[(_m, mname, k)])
            R2, _, _, _ = rq2.run_rq2(F, raw, cfg, models=("xgb", "lr"), milestones=[m])
            assert cnt == {"xgb": len(W), "lr": len(W)}, f"RQ2 {m}: selections {cnt} for {len(W)} windows"
            R2.to_csv(fm, index=False); frames.append(R2)
        pd.concat(frames).assign(variant=name).to_csv(f2, index=False)


def ps(x):
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return dict(mean=m, ci_lo=lo, ci_hi=hi)


notes = json.loads((OUT / "tables" / "sample_notes.json").read_text(encoding="utf-8"))
summary = []
for name in _a.variants.split(","):
    t = time.time()
    raw = variant_raw(name)
    S, flow, nt = build_sample(raw, cfg)
    if name == "main":
        assert nt["primary_n"] == notes["primary_n"] and nt["primary_pos"] == notes["primary_pos"], "main sample differs"
    F = build_features(S, raw, cfg)
    D = F[F["y"] >= 0].reset_index(drop=True)
    (EXTRA / "tables" / f"sample_{name}_size.json").write_text(
        json.dumps(dict(n=len(D), pos=int(D["y"].sum()))), encoding="utf-8")
    fw = EXTRA / "tables" / f"sample_{name}_window_metrics.csv"
    if fw.is_file():                                     # finished in an earlier, interrupted run
        R = pd.read_csv(fw)
        windows_ok(fw)
    else:
        vocab = {c: sorted(F[c].unique()) for c in BASE_CAT}
        R, _ = rq1.run_factorial(D, vocab, cfg, models=("xgb", "lr"), only=COND)
        R.assign(variant=name).to_csv(fw, index=False)
    if name == "main" and not QUICK:
        # the main variant repeats the reported RQ1 fits: logistic regression is deterministic and must agree with
        # tables/rq1_window_metrics.csv; XGBoost agrees on the platform of the reported run only
        pub = pd.read_csv(OUT / "tables" / "rq1_window_metrics.csv")
        cmp_ = R.merge(pub, on=["model", "shortcuts", "k"], suffixes=("", "_pub"))
        for mm in ("lr", "xgb"):
            dmax = (cmp_.loc[cmp_.model == mm, "ap"] - cmp_.loc[cmp_.model == mm, "ap_pub"]).abs().max()
            print(f"main, {mm}: largest PR-AUC difference from the reported run {dmax:.2e} "
                  f"({len(cmp_[cmp_.model == mm])} window fits)", flush=True)
        assert len(cmp_) == len(R), "main: windows do not match the reported run"
        assert (cmp_.loc[cmp_.model == "lr", "ap"] - cmp_.loc[cmp_.model == "lr", "ap_pub"]).abs().max() < 1e-6, \
            "main: logistic regression does not reproduce the reported run"
    piv = R.pivot_table(index=["model", "k"], columns="shortcuts", values="ap")
    row = dict(variant=name, n=len(D), pos=int(D["y"].sum()), base_rate=float(D["y"].mean()))
    for mm in ("xgb", "lr"):
        p = piv.loc[mm]
        row[f"ref_ap_{mm}"] = float(p["none"].mean())
        for sc in ("P", "R+G+P"):
            d = ps((p[sc] - p["none"]).to_numpy())
            row[f"infl_{sc}_{mm}_pct"] = 100 * d["mean"] / p["none"].mean()
    for sc in ("none", "R"):
        d = ps((piv.loc["xgb"][sc] - piv.loc["lr"][sc]).to_numpy())
        row.update({f"xgb_minus_lr_{sc}": d["mean"], f"xgb_minus_lr_{sc}_lo": d["ci_lo"], f"xgb_minus_lr_{sc}_hi": d["ci_hi"]})
    summary.append(row)
    if name != "main" and not _a.no_rq24:
        run_rq24(name, F, raw)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: n = {len(D)} ({time.time() - t:.0f}s)", flush=True)
pd.DataFrame(summary).to_csv(EXTRA / "tables" / f"sample_variants_summary_{_a.variants.replace(',', '_')}.csv", index=False)
print(pd.DataFrame(summary).T.to_string())
