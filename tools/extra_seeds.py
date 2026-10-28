"""Seed sensitivity of RQ2 (milestones) and RQ3/RQ4 (recalibration and alert policies) for XGBoost.

usage: python tools/extra_seeds.py OUTPUTS_DIR EXTRA_DIR --seeds 43 44 45 [--nested] [--what rq2,rq3] [--quick] [--data DIR]

Logistic regression is deterministic under the temporal protocol (no random folds, no subsampling), so only XGBoost
depends on the seed, through row and column subsampling and, with --nested, through the inner tuning. By default
the configuration that the reported run selected in every window (tables/rq2_chosen_params.csv,
tables/rq3_chosen_params.csv) is refitted with each seed, which isolates the randomness of the fit; with --nested
the inner tuning is repeated for each seed, as in the main stages. The stage functions of the released code are
called unchanged; only the seed of the configuration is replaced. Results go to EXTRA_DIR/tables; the released
outputs are not touched. A seed and part whose result file already exists is skipped, so an interrupted run can be
restarted with the same command.
"""
import argparse
import ast
import copy
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.data.io import load_raw                                       # noqa: E402
from leakaware.experiments import rq2_timing as rq2                          # noqa: E402
from leakaware.experiments import rq3_recalibration as rq3                   # noqa: E402
from leakaware.config import load_config                                     # noqa: E402
from leakaware.evaluation.splits import rolling_windows                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("extra")
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--nested", action="store_true")
ap.add_argument("--what", default="rq2,rq3")
ap.add_argument("--quick", action="store_true", help="three windows only, to test the script")
ap.add_argument("--data", default=None, help="folder with the nine raw Olist tables (default: data_dir of config.yaml)")
a = ap.parse_args()
OUT, EXTRA = Path(a.out), Path(a.extra)
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
cfg0 = load_config()
if a.data:
    cfg0["data_dir"] = Path(a.data)
F = pd.read_pickle(OUT / "data" / "features_t0.pkl")
if a.quick:
    cfg0["rolling"]["step_days"] = 140

if not a.nested:
    ch2 = pd.read_csv(OUT / "tables" / "rq2_chosen_params.csv")
    ch3 = pd.read_csv(OUT / "tables" / "rq3_chosen_params.csv")
    P2 = {(r.milestone, r.model, r.k): ast.literal_eval(r.params) for r in ch2.itertuples()}
    P3 = {(r.model, r.k): ast.literal_eval(r.params) for r in ch3.itertuples()}


def fixed_select(P, key_fn):
    """Replacement for the inner tuning: returns the configuration the reported run selected."""
    state = {}

    def select(mname, Ds, ys, splits, num, cat, vocab, cfg):
        return P[key_fn(state, mname)], float("nan")
    return select, state


what = a.what.split(",")
K = len(rolling_windows(cfg0))


def check_reused(f):
    """A result file left by an earlier run is reused only if it covers the windows of this run (not --quick)."""
    k = pd.read_csv(f, usecols=["k"])["k"].nunique()
    if k != K:
        sys.exit(f"{f} has {k} windows, this run has {K}: move the file away (an earlier --quick run?) and restart")
    print(f"{f.name} exists ({k} windows), skipped", flush=True)


for seed in a.seeds:
    cfg = copy.deepcopy(cfg0); cfg["seed"] = seed
    t = time.time()
    f3 = EXTRA / "tables" / f"seed{seed}_rq3_window_metrics.csv"
    f2 = EXTRA / "tables" / f"seed{seed}_rq2_window_metrics.csv"
    if "rq3" in what and f3.is_file():
        check_reused(f3)
    elif "rq3" in what:
        if not a.nested:
            # run_rq3 loops over windows in order; the current window index is tracked through the call count
            W = rq3.rolling_windows(cfg); calls = {"i": 0}

            def sel3(mname, Ds, ys, splits, num, cat, vocab, cfg_, _W=W, _c=calls):
                k = _W[_c["i"] % len(_W)]["k"]; _c["i"] += 1
                return P3[(mname, k)], float("nan")
            rq3._inner_select = sel3
        R, Dd, C, Ch = rq3.run_rq3(F, cfg, models=("xgb",))
        if not a.nested:
            assert calls["i"] == len(W), f"RQ3: {calls['i']} selections for {len(W)} windows"
        Dd.assign(seed=seed).to_csv(EXTRA / "tables" / f"seed{seed}_rq3_daily_policy.csv", index=False)
        R.assign(seed=seed).to_csv(f3, index=False)           # written last: its presence marks a finished part
    if "rq2" in what and f2.is_file():
        check_reused(f2)
    elif "rq2" in what:
        raw, _ = load_raw(cfg["data_dir"], verify=False)
        frames = []
        for m in rq2.MILESTONES:
            if not a.nested:
                W = rq2.rolling_windows(cfg); calls = {"i": 0}

                def sel2(mname, Xtr, ytr, splits, num, cat, vocab, cfg_, _W=W, _c=calls, _m=m):
                    k = _W[_c["i"] % len(_W)]["k"]; _c["i"] += 1
                    return P2[(_m, mname, k)], float("nan")
                rq2._inner_select = sel2
            R2, L2, Ch2, Sc2 = rq2.run_rq2(F, raw, cfg, models=("xgb",), milestones=[m])
            if not a.nested:
                assert calls["i"] == len(W), f"RQ2 {m}: {calls['i']} selections for {len(W)} windows"
            frames.append(R2)
        pd.concat(frames).assign(seed=seed).to_csv(f2, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] seed {seed} done ({time.time() - t:.0f}s)", flush=True)
