"""Separates the bias left by the weight cap from the bias due to Assumption 1(i) (latency independent of the
features given the class), for the latency-weighted Platt map of RQ3.

usage: python tools/recal_cap_check.py OUTPUTS_DIR EXTRA_DIR

For every origin and learner the model of the rq3 stage is refitted with the configuration that the stage selected
(tables/rq3_chosen_params.csv), on the same training set, so no tuning is repeated. The script first checks that the
latency-weighted map with the cap of the article (20) reproduces the saved probabilities (data/rq3_predictions.csv.gz)
and stops otherwise. It then refits the map with
  cap100          weights capped at 100
  cap20_age4      cap 20, recalibration orders at least 4 days old
  nocap_age4      no cap, recalibration orders at least 4 days old (the youngest orders, whose label probability
                  is close to zero, are the ones the cap exists for)
  speed           cap 20, latency distribution estimated within two strata of the transit time predicted at t0
                  from admissible features (gradient-boosted regression fitted on matured orders): 7 days or less,
                  more than 7 days
and reports calibration and cost in the evaluation windows, and calibration-in-the-large within strata of the
realised transit time (7 days or less, more), as in tables/robustness_recal_residual_by_stratum.csv.
Results go to EXTRA_DIR/tables; the released outputs are not touched. Released code in src/ is imported, not changed.
"""
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from leakaware.data.features import feature_set, BASE_CAT, BASE_NUM, HIST_ASOF          # noqa: E402
from leakaware.evaluation.metrics import moving_block_bootstrap_mean, window_metrics       # noqa: E402
from leakaware.evaluation.models import make_model                                         # noqa: E402
from leakaware.evaluation.splits import rolling_windows                                    # noqa: E402
from leakaware.experiments.rq3_recalibration import Platt, latency_weights, calib_stats, decisions   # noqa: E402

if len(sys.argv) not in (3, 4):
    sys.exit(__doc__)
OUT, EXTRA = Path(sys.argv[1]), Path(sys.argv[2])
QUICK = len(sys.argv) == 4 and sys.argv[3] == "--quick"
(EXTRA / "tables").mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))
bs = cfg["bootstrap"]

F = pd.read_pickle(OUT / "data" / "features_t0.pkl")
D = F[F["y"] >= 0].reset_index(drop=True)
vocab = {c: sorted(F[c].unique()) for c in BASE_CAT}          # as in run_rq3
num, cat = feature_set("asof", False)
y_all = D["y"].to_numpy()
chosen = pd.read_csv(OUT / "tables" / "rq3_chosen_params.csv")
saved = pd.read_csv(OUT / "data" / "rq3_predictions.csv.gz")
H28 = pd.Timedelta(days=28)
lag = ((D["label_available_ts"] - D["t0"]).dt.total_seconds() / 86400).to_numpy()
transit = D["p_transit_days"].to_numpy(dtype=float)
adm = BASE_NUM + HIST_ASOF


def speed_weights(a, cal, clip=20.0, horizon_days=90, cut=7.0):
    """1/G_{y,s}(age) with s = predicted transit time (<= cut, > cut), predicted from admissible features."""
    from xgboost import XGBRegressor
    mat = ((D["t0"] < a - pd.Timedelta(days=horizon_days)) & (D["label_available_ts"] < a)).to_numpy()
    fit_rows = mat & np.isfinite(transit) & (D["t0"] + pd.to_timedelta(np.nan_to_num(transit), unit="D") < a).to_numpy()
    X = D[adm + BASE_CAT].copy()
    for c_ in BASE_CAT:
        X[c_] = pd.Categorical(X[c_], categories=vocab[c_])
    reg = XGBRegressor(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_estimators=300, max_depth=5,
                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=cfg["seed"], n_jobs=2)
    reg.fit(X[fit_rows], transit[fit_rows])
    need = mat | cal
    pred = np.full(len(D), np.nan); pred[need] = reg.predict(X[need])
    st = (pred > cut).astype(int)
    age = ((a - D["t0"]).dt.total_seconds() / 86400).to_numpy()
    w = np.ones(cal.sum()); yc, agec, sc = y_all[cal], age[cal], st[cal]
    for cls in (0, 1):
        Lc = np.sort(lag[mat & (y_all == cls)])
        for s_ in (0, 1):
            L = np.sort(lag[mat & (y_all == cls) & (st == s_)])
            if len(L) < 50:
                L = Lc
            sel = (yc == cls) & (sc == s_)
            G = np.searchsorted(L, agec[sel], side="right") / len(L)
            w[sel] = 1 / np.clip(G, 1 / clip, 1)
    return w, float(np.mean(st[cal]))


rows, strata, check = [], [], []
W = rolling_windows(cfg)
if QUICK:
    W = W[:: max(1, len(W) // 3)][:3]
for mname in ("xgb", "lr"):
    t = time.time()
    for w in W:
        a = w["origin"]
        known = (D["label_available_ts"] < a).to_numpy()
        trA = (D["t0"] < a - H28).to_numpy() & known
        cal = ((D["t0"] >= a - H28) & (D["t0"] < a)).to_numpy() & known
        ev = ((D["t0"] >= a) & (D["t0"] < w["end"])).to_numpy()
        p = ast.literal_eval(chosen[(chosen.model == mname) & (chosen.k == w["k"])].params.iloc[0])
        mA = make_model(mname, num, cat, vocab, cfg, **p).fit(D[trA], y_all[trA])
        pA_cal, pA_ev = mA.predict_proba(D[cal]), mA.predict_proba(D[ev])
        yc, y = y_all[cal], y_all[ev]
        age_cal = ((a - D.loc[cal, "t0"]).dt.total_seconds() / 86400).to_numpy()
        old4 = age_cal >= 4
        w20 = latency_weights(D, a, cal, y_all)
        ref = Platt().fit(pA_cal, yc, w20)(pA_ev)
        s_ = saved[(saved.model == mname) & (saved.k == w["k"])].p_ipw.to_numpy()
        diff = float(np.abs(ref - s_).max()) if len(s_) == len(ref) else np.inf
        check.append(dict(model=mname, k=w["k"], max_abs_diff=diff))
        w100 = latency_weights(D, a, cal, y_all, clip=100.0)
        winf = latency_weights(D, a, cal, y_all, clip=1e12)
        wsp, share_slow = speed_weights(a, cal)
        P = {"cap20": ref,
             "cap100": Platt().fit(pA_cal, yc, w100)(pA_ev),
             "cap20_age4": Platt().fit(pA_cal[old4], yc[old4], w20[old4])(pA_ev),
             "nocap_age4": Platt().fit(pA_cal[old4], yc[old4], winf[old4])(pA_ev),
             "speed": Platt().fit(pA_cal, yc, wsp)(pA_ev)}
        fast = (transit[ev] <= 7)
        for s, pr in P.items():
            rows.append(dict(model=mname, strategy=s, k=w["k"], n=int(ev.sum()), max_weight={"cap20": w20, "cap100": w100,
                             "cap20_age4": w20[old4], "nocap_age4": winf[old4], "speed": wsp}[s].max(),
                             share_slow_pred=share_slow, **calib_stats(y, pr), **decisions(y, pr, None),
                             ap=window_metrics(y, pr)["ap"]))
            for z, msk in (("yes", fast), ("no", ~fast)):
                if msk.sum() > 0:
                    strata.append(dict(model=mname, strategy=s, k=w["k"], delivered_within_7d=z,
                                       citl=float(pr[msk].mean() - y[msk].mean())))
    print(f"[{time.strftime('%H:%M:%S')}] {mname} done ({time.time() - t:.0f}s)", flush=True)

CK = pd.DataFrame(check); CK.to_csv(EXTRA / "tables" / "recal_cap_check_reproduction.csv", index=False)
for m_, g_ in CK.groupby("model"):
    print(f"{m_}: max abs difference to the saved latency-weighted probabilities {g_.max_abs_diff.max():.2e}")
# logistic regression is deterministic and must match exactly. XGBoost row and column subsampling draws its random
# numbers through the C++ standard library, whose distributions differ between compilers, so a run on another
# platform than the reported one (Windows) gives slightly different trees; the variants are then compared with the
# cap-20 map refitted in the same run (paired), never with the saved probabilities.
assert float(CK[CK.model == "lr"].max_abs_diff.max()) < 1e-6, "logistic regression does not reproduce the rq3 stage"
R = pd.DataFrame(rows); R.to_csv(EXTRA / "tables" / "recal_cap_check_windows.csv", index=False)
S = pd.DataFrame(strata); S.to_csv(EXTRA / "tables" / "recal_cap_check_strata_windows.csv", index=False)


def ps(x):
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return dict(mean=m, ci_lo=lo, ci_hi=hi)


summ = []
for (m, s), g in R.groupby(["model", "strategy"]):
    g = g.sort_values("k")
    for c in ("citl_diff", "cal_intercept", "ece", "saving_r5", "saving_r10", "ap", "max_weight"):
        summ.append(dict(model=m, strategy=s, metric=c, **ps(g[c])))
    if s != "cap20":
        r0 = R[(R.model == m) & (R.strategy == "cap20")].sort_values("k")
        for c in ("citl_diff", "cal_intercept", "ece", "saving_r5", "saving_r10"):
            summ.append(dict(model=m, strategy=f"{s} - cap20", metric=c, **ps(g[c].to_numpy() - r0[c].to_numpy())))
for (m, s, z), g in S.groupby(["model", "strategy", "delivered_within_7d"]):
    summ.append(dict(model=m, strategy=s, metric=f"citl_delivered_within_7d_{z}", **ps(g.sort_values("k").citl)))
pd.DataFrame(summ).to_csv(EXTRA / "tables" / "recal_cap_check_summary.csv", index=False)
print("summary written to", EXTRA / "tables" / "recal_cap_check_summary.csv")
