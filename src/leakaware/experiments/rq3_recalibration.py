"""RQ3: recalibration and alert policies under outcome-dependent label latency.

At hand-over (as-of features, reference protocol), every rolling origin compares nine probability strategies:
  static            : model and Platt map fitted ONCE at the first origin and then frozen
  raw               : model refitted at every origin, raw probabilities
  platt             : refitted; Platt map fitted on the labelled orders of the last 28 days (unseen by the model)
  isotonic          : as platt, with isotonic regression
  prior_recent      : raw probabilities shifted to the base rate of the labelled orders of the last 28 days
  prior_em          : raw probabilities shifted to a current base rate estimated by EM (Saerens et al., 2002) on the
                      orders handed over in the last 14 days, which are still unlabelled at the origin
  platt_ipw         : Platt map weighted by 1/G_y(order age); G_y estimated on all matured orders (> 90 days)
  platt_ipw_roll    : as platt_ipw, with G_y estimated on a rolling 180-day window of matured orders
  platt_ipw_nowcast : platt_ipw_roll followed by a prior shift from the weighted 28-day base rate to a 14-day nowcast
                      estimated by IPW on the labels already received
Hyper-parameters are tuned within each origin: inner training on t0 < a - 56 days (labels before a - 56 days), inner
validation on t0 in [a - 56, a - 28) days (labels before a). The last 28 days used for recalibration are never used
for tuning. Decisions: Bayes threshold t* = 1/(1 + r) with r = c_FN/c_FP in {2, 5, 10}; a daily top-10% budget;
decision curves.
"""
import math
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from ..data.features import feature_set, BASE_CAT
from ..evaluation.metrics import window_metrics, moving_block_bootstrap_mean
from ..evaluation.models import make_model
from .rq1_inflation import _inner_select
from ..evaluation.splits import rolling_windows

STRATEGIES = ["static", "raw", "platt", "isotonic", "prior_recent", "prior_em", "platt_ipw", "platt_ipw_roll",
              "platt_ipw_nowcast"]
STRAT_LABEL = {"static": "Frozen model + Platt map", "raw": "Refit, raw probabilities",
               "platt": "Refit + rolling Platt (28 d)", "isotonic": "Refit + rolling isotonic (28 d)",
               "prior_recent": "Refit + prior shift to last-28-day base rate",
               "prior_em": "Refit + EM prior shift (recent orders, labels unused)",
               "platt_ipw": "Refit + latency-weighted Platt (expanding latency model)",
               "platt_ipw_roll": "Refit + latency-weighted Platt (rolling latency model)",
               "platt_ipw_nowcast": "Refit + latency-weighted Platt + 14-day nowcast prior"}
RATIOS = [2, 5, 10]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y, w=None):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y, sample_weight=w); return self

    def __call__(self, p):
        return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y); return self

    def __call__(self, p):
        return self.m.predict(p)


def em_prior(p, pi_train, n_iter=200, tol=1e-7):
    """Saerens, Latinne and Decaestecker (2002): estimate a new base rate from the probabilities of unlabelled data."""
    pi = pi_train
    for _ in range(n_iter):
        a = (pi / pi_train) * p
        b = ((1 - pi) / (1 - pi_train)) * (1 - p)
        q = a / (a + b)
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def prior_shift(p, pi_train, pi_new):
    a = (pi_new / pi_train) * p
    b = ((1 - pi_new) / (1 - pi_train)) * (1 - p)
    return a / (a + b)


def latency_weights(D, a, cal, y_all, horizon_days=90, clip=20.0, window_days=None):
    """Inverse-probability weights for 'the label has already arrived', by order age and class.

    Low ratings arrive later than high ratings, so the labelled orders of the last 28 days under-represent class 1.
    G_y(d) = P(label latency <= d | y) is estimated on matured orders (t0 < origin - 90 days, label known before the
    origin). The weight of order i in the recalibration set is 1 / G_{y_i}(a - t0_i), capped at `clip`.
    """
    mat = ((D["t0"] < a - pd.Timedelta(days=horizon_days)) & (D["label_available_ts"] < a)).to_numpy()
    if window_days is not None:
        mat = mat & (D["t0"] >= a - pd.Timedelta(days=horizon_days + window_days)).to_numpy()   # no in-place update (arrays are read-only in pandas 3)
    lag = ((D["label_available_ts"] - D["t0"]).dt.total_seconds() / 86400).to_numpy()
    age = ((a - D["t0"]).dt.total_seconds() / 86400).to_numpy()
    w = np.ones(cal.sum())
    yc, agec = y_all[cal], age[cal]
    for cls in (0, 1):
        L = np.sort(lag[mat & (y_all == cls)])
        if len(L) < 50:
            continue
        G = np.searchsorted(L, agec[yc == cls], side="right") / len(L)
        w[yc == cls] = 1 / np.clip(G, 1 / clip, 1)
    return w


def calib_stats(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df.p.rank(method="first"), 10, labels=False)
    g = df.groupby("b").agg(n=("y", "size"), py=("y", "mean"), pp=("p", "mean"))
    ece = float((g.n / g.n.sum() * (g.py - g.pp).abs()).sum())
    lp = _logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0][0]
    # calibration intercept: intercept with the slope fixed at 1 (offset), solved by root finding
    from scipy.optimize import brentq
    from scipy.special import expit
    f = lambda a_: float((y - expit(a_ + lp)).sum())
    a = brentq(f, -20, 20) if f(-20) * f(20) < 0 else np.nan
    return dict(mean_pred=float(p.mean()), obs=float(y.mean()), citl_diff=float(p.mean() - y.mean()), ece=ece,
                cal_intercept=float(a), cal_slope=float(slope), brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def decisions(y, p, day):
    y = np.asarray(y, int); P, N = int(y.sum()), int(len(y) - y.sum()); out = {}
    for r in RATIOS:
        t = 1 / (1 + r)
        a = p >= t
        tp = int((a & (y == 1)).sum()); fp = int(a.sum() - tp)
        cost = fp + r * (P - tp); best = min(r * P, N)
        # best threshold chosen in hindsight on the window itself (an upper bound that cannot be deployed)
        qs = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
        costs = [((p >= q) & (y == 0)).sum() + r * (P - ((p >= q) & (y == 1)).sum()) for q in qs]
        out.update({f"alert_rate_r{r}": float(a.mean()), f"cost_r{r}": cost, f"best_baseline_r{r}": best,
                    f"saving_r{r}": 1 - cost / best, f"saving_oracle_r{r}": 1 - min(min(costs), best) / best})
    return out


def run_rq3(F, cfg, models=("xgb", "lr")):
    D = F[F["y"] >= 0].reset_index(drop=True)
    vocab = {c: sorted(F[c].unique()) for c in BASE_CAT}
    num, cat = feature_set("asof", False)
    W = rolling_windows(cfg)
    y_all = D["y"].to_numpy()
    rows, daily, curves, chosen = [], [], [], []
    H28, H14, H56 = pd.Timedelta(days=28), pd.Timedelta(days=14), pd.Timedelta(days=56)
    for mname in models:
        t = time.time()
        static = None
        for w in W:
            a = w["origin"]
            known = (D["label_available_ts"] < a).to_numpy()
            trA = (D["t0"] < a - H28).to_numpy() & known
            cal = ((D["t0"] >= a - H28) & (D["t0"] < a)).to_numpy() & known
            full = (D["t0"] < a).to_numpy() & known
            unl = ((D["t0"] >= a - H14) & (D["t0"] < a)).to_numpy()
            ev = ((D["t0"] >= a) & (D["t0"] < w["end"])).to_numpy()
            # nested tuning (touches neither the recalibration set nor the evaluation window)
            sel = (D["t0"] < a - H28).to_numpy() & known
            Ds = D[sel].reset_index(drop=True); ys = y_all[sel]
            itr = ((Ds["t0"] < a - H56) & (Ds["label_available_ts"] < a - H56)).to_numpy()
            iva = (Ds["t0"] >= a - H56).to_numpy()
            p, s_in = _inner_select(mname, Ds, ys, [(itr, iva)], num, cat, vocab, cfg)
            chosen.append(dict(model=mname, k=w["k"], params=str(p), inner_ap=s_in))
            mA = make_model(mname, num, cat, vocab, cfg, **p).fit(D[trA], y_all[trA])
            pA_cal, pA_ev = mA.predict_proba(D[cal]), mA.predict_proba(D[ev])
            mF = make_model(mname, num, cat, vocab, cfg, **p).fit(D[full], y_all[full])
            pF_ev, pF_unl = mF.predict_proba(D[ev]), mF.predict_proba(D[unl])
            pi_tr = y_all[full].mean()
            pi_hat = em_prior(pF_unl, pi_tr)
            if static is None:
                static = (mA, Platt().fit(pA_cal, y_all[cal]))
            wl = latency_weights(D, a, cal, y_all)
            pi_ipw = float((wl * y_all[cal]).sum() / wl.sum())
            wr = latency_weights(D, a, cal, y_all, window_days=180)
            pi_ipw_roll = float((wr * y_all[cal]).sum() / wr.sum())
            rec = ((D["t0"] >= a - H14) & (D["t0"] < a)).to_numpy() & known
            wn = latency_weights(D, a, rec, y_all, window_days=180)
            pi_now = float((wn * y_all[rec]).sum() / wn.sum()) if rec.sum() >= 30 else pi_ipw_roll
            p_roll = Platt().fit(pA_cal, y_all[cal], wr)(pA_ev)
            probs = {"static": static[1](static[0].predict_proba(D[ev])), "raw": pF_ev,
                     "platt_ipw": Platt().fit(pA_cal, y_all[cal], wl)(pA_ev),
                     "platt_ipw_roll": p_roll,
                     "platt_ipw_nowcast": prior_shift(p_roll, pi_ipw_roll, pi_now),
                     "platt": Platt().fit(pA_cal, y_all[cal])(pA_ev), "isotonic": Iso().fit(pA_cal, y_all[cal])(pA_ev),
                     "prior_recent": prior_shift(pF_ev, pi_tr, float(y_all[cal].mean())),
                     "prior_em": prior_shift(pF_ev, pi_tr, pi_hat)}
            y = y_all[ev]
            for s, pr in probs.items():
                rows.append(dict(model=mname, strategy=s, k=w["k"], origin=a, n=int(ev.sum()), pi_train=pi_tr,
                                 pi_em=pi_hat, pi_recent_labeled=float(y_all[cal].mean()), pi_recent_ipw=pi_ipw,
                                 pi_recent_ipw_roll=pi_ipw_roll, pi_nowcast=pi_now, n_recent_labeled=int(rec.sum()),
                                 **calib_stats(y, pr), **decisions(y, pr, None), ap=window_metrics(y, pr)["ap"]))
            # daily policy (rolling Platt): top 10% of each day versus a fixed threshold that alerts 10% of the recalibration set
            pc = probs["platt"]
            thr10 = np.quantile(Platt().fit(pA_cal, y_all[cal])(pA_cal), 0.90)
            dd = pd.DataFrame(dict(day=D.loc[ev, "t0"].dt.normalize().to_numpy(), y=y, p=pc))
            for day, g in dd.groupby("day"):
                k = max(1, math.ceil(0.10 * len(g)))
                top = g.sort_values("p", ascending=False).head(k)
                fx = g[g.p >= thr10]
                daily.append(dict(model=mname, k=w["k"], day=day, n=len(g), pos=int(g.y.sum()),
                                  topk_alerts=k, topk_tp=int(top.y.sum()),
                                  thr_alerts=len(fx), thr_tp=int(fx.y.sum())))
            curves.append(pd.DataFrame(dict(model=mname, k=w["k"], y=y, p_platt=pc, p_prior=probs["prior_em"],
                                            p_ipw=probs["platt_ipw"], p_static=probs["static"])))
        print(f"[{time.strftime('%H:%M:%S')}] RQ3 {mname} done ({time.time()-t:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(daily), pd.concat(curves), pd.DataFrame(chosen)


def summarize_rq3(R, cfg):
    bs = cfg["bootstrap"]; out = []
    cols = ["citl_diff", "ece", "cal_intercept", "cal_slope", "brier", "logloss", "ap"] + \
           [f"saving_r{r}" for r in RATIOS] + [f"saving_oracle_r{r}" for r in RATIOS] + [f"alert_rate_r{r}" for r in RATIOS]
    for (m, s), g in R.groupby(["model", "strategy"]):
        row = dict(model=m, strategy=s)
        for c in cols:
            mean, lo, hi, _ = moving_block_bootstrap_mean(g[c].to_numpy(), bs["n_boot"], bs["block_len"], cfg["seed"])
            row.update({c: mean, f"{c}_lo": lo, f"{c}_hi": hi})
        row["abs_citl"] = g["citl_diff"].abs().mean()
        out.append(row)
    return pd.DataFrame(out)


def decision_curve(C, model="xgb"):
    g = C[C.model == model]
    y = g.y.to_numpy(); n = len(y); pi = y.mean(); rows = []
    for t in np.linspace(0.02, 0.5, 49):
        for col in ["p_platt", "p_ipw", "p_prior", "p_static"]:
            a = g[col].to_numpy() >= t
            tp = (a & (y == 1)).sum(); fp = (a & (y == 0)).sum()
            rows.append(dict(model=model, threshold=t, strategy=col[2:], net_benefit=tp / n - fp / n * t / (1 - t)))
        rows.append(dict(model=model, threshold=t, strategy="alert_all", net_benefit=pi - (1 - pi) * t / (1 - t)))
    return pd.DataFrame(rows)


def paired_strategy(R, cfg, a="platt_ipw", b="platt", cols=("citl_diff", "cal_intercept", "ece", "brier", "saving_r5", "saving_r10")):
    bs = cfg["bootstrap"]; rows = []
    for m, g in R.groupby("model"):
        A = g[g.strategy == a].set_index("k"); B = g[g.strategy == b].set_index("k")
        for c in cols:
            if c == "citl_diff":
                d = (A[c].abs() - B[c].abs()).to_numpy(); name = "abs_citl"
            else:
                d = (A[c] - B[c]).to_numpy(); name = c
            mean, lo, hi, _ = moving_block_bootstrap_mean(d, bs["n_boot"], bs["block_len"], cfg["seed"])
            rows.append(dict(model=m, comparison=f"{a} - {b}", metric=name, diff=mean, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)
