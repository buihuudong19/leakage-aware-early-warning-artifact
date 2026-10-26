"""RQ2: how does the usefulness of a warning change across the order lifecycle?

Prediction milestones tau_m: purchase, payment approval, carrier hand-over (t0), t0 + 3 days, t0 + 7 days and delivery.
At each milestone only information known at tau_m is used, including seller and category histories as of tau_m.

Unlike milestone-wise discrimination, the evaluation accounts for feedback that arrives before the warning: the longer
the platform waits for information, the more customers (mostly dissatisfied ones) have already reviewed, and those
orders can no longer be recovered.
  * The cohort of a window is every order with t0 in the window; it is the same at every milestone, so comparisons
    are paired.
  * At milestone m only orders still WITHOUT feedback at tau_m are scored; orders already reviewed count as missed.
  * The alert budget is fixed at 10% of the cohort.
  * Main metric: the share of low-rated orders flagged before the customer answers (early catch rate), together with
    PR-AUC among the orders that can still be scored.
"""
import math
import time

import numpy as np
import pandas as pd

from ..data.features import order_attributes, _asof_cum, _asof_global, BASE_CAT
from ..evaluation.metrics import window_metrics
from ..evaluation.models import make_model
from .rq1_inflation import _inner_select
from ..evaluation.splits import rolling_windows

DAY = pd.Timedelta(days=1)
MILESTONES = ["purchase", "approval", "t0", "t0+3d", "t0+7d", "delivery"]
MILESTONE_LABEL = {"purchase": "Order placed", "approval": "Payment approved", "t0": "Carrier hand-over (t0)",
                   "t0+1d": "t0 + 1 day", "t0+2d": "t0 + 2 days", "t0+3d": "t0 + 3 days", "t0+7d": "t0 + 7 days", "delivery": "Delivered"}
PURCHASE_NUM = ["f_promised_days", "f_purchase_dow", "f_purchase_hour", "f_purchase_month", "f_freight_ratio",
                "f_same_state", "f_n_items", "f_n_products", "f_n_sellers", "f_price_sum", "f_price_max",
                "f_freight_sum", "f_weight_g", "f_volume_cm3", "f_photos_qty", "f_name_len", "f_desc_len",
                "f_pay_value", "f_n_payments", "f_installments", "f_distance_km"]
HIST = ["m_seller_n", "m_seller_lowrate", "m_cat_n", "m_cat_lowrate", "m_cstate_n", "m_cstate_lowrate",
        "m_sstate_n", "m_sstate_lowrate", "m_seller_late_n", "m_seller_laterate", "m_seller_proc_mean",
        "h_seller_vol30", "h_customer_prior_orders"]
STAGE = ["s_approved", "s_approval_hours", "s_handed_over", "s_processing_days", "s_handover_after_approval",
         "s_remaining_promise_t0", "s_elapsed_since_t0", "s_delivered", "s_delivery_days", "s_delay_vs_estimate",
         "s_remaining_promise_now", "s_overdue"]
NUM = PURCHASE_NUM + HIST + STAGE


def tau(F, m):
    if m == "purchase":
        return F["purchase_ts"]
    if m == "approval":
        return F["approved_ts"].fillna(F["purchase_ts"])
    if m == "t0":
        return F["t0"]
    if m == "t0+1d":                     # finer milestones used only in the robustness analysis
        return F["t0"] + 1 * DAY
    if m == "t0+2d":
        return F["t0"] + 2 * DAY
    if m == "t0+3d":
        return F["t0"] + 3 * DAY
    if m == "t0+7d":
        return F["t0"] + 7 * DAY
    if m == "delivery":
        return F["delivered_ts"].fillna(F["t0"] + 30 * DAY)
    raise ValueError(m)


def milestone_features(F, raw, cfg, m, attrs=None, events=None):
    """Feature matrix at milestone m for every order of F (as of tau_m)."""
    mm = cfg["features"]["smoothing_m"]
    X = F[["order_id", "y", "t0", "purchase_ts", "label_available_ts", "first_answer_ts", "delivered_ts"] + PURCHASE_NUM
          + BASE_CAT + ["seller_id", "category", "customer_state", "seller_state", "h_seller_vol30",
                        "h_customer_prior_orders"]].copy()
    X["tau"] = tau(F, m).to_numpy()
    X["_qid"] = np.arange(len(X))
    ev, late_ev, proc_ev = events
    gn, gs = _asof_global(X, ev, "tau", "avail", "low")
    prior = np.where(gn > 0, gs / np.maximum(gn, 1), 0.13)
    for key, name in [("seller_id", "seller"), ("category", "cat"), ("customer_state", "cstate"), ("seller_state", "sstate")]:
        n, s = _asof_cum(X, ev, key, "tau", "avail", "low")
        X[f"m_{name}_n"], X[f"m_{name}_lowrate"] = n, (s + mm * prior) / (n + mm)
    n, s = _asof_cum(X, late_ev, "seller_id", "tau", "late_time", "late")
    X["m_seller_late_n"], X["m_seller_laterate"] = n, (s + 5 * 0.08) / (n + 5)
    n, s = _asof_cum(X, proc_ev, "seller_id", "tau", "t0", "proc_days")
    X["m_seller_proc_mean"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)

    t = X["tau"]
    appr = F["approved_ts"]
    approved = appr.notna() & (appr <= t)
    X["s_approved"] = approved.astype(int)
    X["s_approval_hours"] = np.where(approved, (appr - F["purchase_ts"]) / pd.Timedelta(hours=1), np.nan)
    handed = F["t0"] <= t
    X["s_handed_over"] = handed.astype(int)
    X["s_processing_days"] = np.where(handed, (F["t0"] - F["purchase_ts"]) / DAY, np.nan)
    X["s_handover_after_approval"] = np.where(handed & approved, (F["t0"] - appr) / DAY, np.nan)
    X["s_remaining_promise_t0"] = np.where(handed, (F["estimated_ts"] - F["t0"]) / DAY, np.nan)
    X["s_elapsed_since_t0"] = np.where(handed, (t - F["t0"]) / DAY, np.nan)
    delivered = F["delivered_ts"].notna() & (F["delivered_ts"] <= t)
    X["s_delivered"] = delivered.astype(int)
    X["s_delivery_days"] = np.where(delivered, (F["delivered_ts"] - F["purchase_ts"]) / DAY, np.nan)
    X["s_delay_vs_estimate"] = np.where(delivered, (F["delivered_ts"] - F["estimated_ts"]) / DAY, np.nan)
    X["s_remaining_promise_now"] = (F["estimated_ts"] - t) / DAY
    X["s_overdue"] = ((t > F["estimated_ts"]) & ~delivered).astype(int)
    return X


def build_events(raw):
    attrs = order_attributes(raw)
    rv = raw["reviews"].groupby("order_id").agg(score=("review_score", "min"), avail=("review_answer_timestamp", "max"))
    ev = attrs.set_index("order_id").join(rv, how="inner").reset_index()
    ev["low"] = (ev["score"] <= 2).astype(float)
    late_ev = attrs[attrs["delivered_ts"].notna() & attrs["t0"].notna()].copy()
    late_ev["late"] = (late_ev["delivered_ts"].dt.normalize() > late_ev["estimated_ts"]).astype(float)
    late_ev["late_time"] = np.maximum(late_ev["delivered_ts"], late_ev["t0"] + pd.Timedelta(seconds=1))
    proc_ev = attrs[attrs["t0"].notna()].copy()
    proc_ev["proc_days"] = (proc_ev["t0"] - proc_ev["purchase_ts"]) / DAY
    return attrs, (ev, late_ev, proc_ev)


def run_rq2(F, raw, cfg, models=("xgb", "lr"), milestones=MILESTONES):
    """Hyper-parameters are tuned within each (milestone, window): the inner training set holds training orders whose tau
    and label precede (cut-off - 28 days); the inner validation set is the last 28 days before the cut-off. Evaluation
    windows are never used. Returns window metrics, milestone diagnostics, selected configurations and the scores of the
    orders still open at each milestone (used for alert budgets other than 10%)."""
    D = F[F["y"] >= 0].reset_index(drop=True)
    H = pd.Timedelta(days=28)
    chosen = []
    vocab = {c: sorted(F[c].unique()) for c in BASE_CAT}
    attrs, events = build_events(raw)
    W = rolling_windows(cfg)
    rows, lead_rows, scores = [], [], []
    for m in milestones:
        t = time.time()
        X = milestone_features(D, raw, cfg, m, attrs, events)
        # training cut-off moved back by the 95th percentile of (t0 - tau_m) so that no label postdates the scoring time
        lag = (X["t0"] - X["tau"]) / DAY
        shift = max(0.0, float(np.nanpercentile(lag, 95)))
        eligible_all = X["first_answer_ts"] > X["tau"]
        lead = (X["label_available_ts"] - X["tau"]) / DAY
        lead_rows.append(dict(milestone=m, train_cutoff_shift_days=shift,
                              median_days_to_label=float(lead[eligible_all].median()),
                              share_feedback_before=float(1 - eligible_all.mean()),
                              share_pos_feedback_before=float(1 - eligible_all[X["y"] == 1].mean()),
                              share_undelivered=float((X["s_delivered"] == 0).mean())))
        for w in W:
            cut = w["origin"] - pd.Timedelta(days=shift)
            # train on the population that is scored: orders still without a review at tau_m (as-of features of an order
            # reviewed before tau_m would contain its own label); at purchase and t0 every order of the sample qualifies
            tr = ((X["tau"] < cut) & (X["label_available_ts"] < cut) & eligible_all).to_numpy()
            base = ((X["t0"] >= w["origin"]) & (X["t0"] < w["end"])).to_numpy()
            ev = base & eligible_all.to_numpy()
            yb = X["y"].to_numpy()[base]
            P_base, n_base = int(yb.sum()), int(base.sum())
            k = math.ceil(0.10 * n_base)
            Xtr = X[tr].reset_index(drop=True); ytr = X["y"].to_numpy()[tr]
            itr = ((Xtr["tau"] < cut - H) & (Xtr["label_available_ts"] < cut - H)).to_numpy()
            iva = (Xtr["tau"] >= cut - H).to_numpy()
            for mname in models:
                p, s_in = _inner_select(mname, Xtr, ytr, [(itr, iva)], NUM, BASE_CAT, vocab, cfg)
                chosen.append(dict(milestone=m, model=mname, k=w["k"], params=str(p), inner_ap=s_in))
                mdl = make_model(mname, NUM, BASE_CAT, vocab, cfg, **p).fit(Xtr, ytr)
                s = mdl.predict_proba(X[ev])
                y = X["y"].to_numpy()[ev]
                met = window_metrics(y, s) if y.sum() > 0 and y.sum() < len(y) else {}
                order = np.argsort(-s, kind="stable")[:k]
                tp = int(y[order].sum())
                scores.append(pd.DataFrame(dict(milestone=m, model=mname, k=w["k"], order_id=X["order_id"].to_numpy()[ev],
                                                y=y, score=s)))
                rows.append(dict(milestone=m, model=mname, k=w["k"], origin=w["origin"], n_base=n_base, pos_base=P_base,
                                 n_eligible=int(ev.sum()), pos_eligible=int(y.sum()), alerts=k, tp=tp,
                                 early_catch=tp / P_base, precision_budget=tp / k, n_train=int(tr.sum()), **met))
        print(f"[{time.strftime('%H:%M:%S')}] RQ2 milestone {m} done ({time.time()-t:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(lead_rows), pd.DataFrame(chosen), pd.concat(scores, ignore_index=True)


def summarize_rq2(R, L, cfg):
    from ..evaluation.metrics import moving_block_bootstrap_mean
    bs = cfg["bootstrap"]; out = []
    for (mname, m), g in R.groupby(["model", "milestone"]):
        row = dict(model=mname, milestone=m)
        for met in ["early_catch", "precision_budget", "ap", "ap_norm", "auc"]:
            mean, lo, hi, _ = moving_block_bootstrap_mean(g[met].to_numpy(), bs["n_boot"], bs["block_len"], cfg["seed"])
            row.update({met: mean, f"{met}_lo": lo, f"{met}_hi": hi})
        row["share_pos_still_open"] = g.pos_eligible.sum() / g.pos_base.sum()
        out.append(row)
    S = pd.DataFrame(out).merge(L, on="milestone")
    S["order"] = S["milestone"].map({m: i for i, m in enumerate(MILESTONES)})
    return S.sort_values(["model", "order"])


def paired_vs_t0(R, cfg):
    """Paired window-level differences: early catch rate at t0 minus that at each other milestone."""
    from ..evaluation.metrics import moving_block_bootstrap_mean
    bs = cfg["bootstrap"]; rows = []
    for mname, g in R.groupby("model"):
        piv = g.pivot_table(index="k", columns="milestone", values="early_catch")
        for other in [m for m in MILESTONES if m != "t0"]:
            d = (piv["t0"] - piv[other]).to_numpy()
            mean, lo, hi, _ = moving_block_bootstrap_mean(d, bs["n_boot"], bs["block_len"], cfg["seed"])
            rows.append(dict(model=mname, comparison=f"t0 - {other}", metric="early_catch", diff=mean, ci_lo=lo,
                             ci_hi=hi, t0_better_share=float((d > 0).mean())))
    return pd.DataFrame(rows)
