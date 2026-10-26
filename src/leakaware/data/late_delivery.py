"""Second task: predicting late delivery at payment approval (a check of the RQ1 findings on the same platform).

Design
  * Population: delivered orders with purchase, approval, delivery and promised dates and at least one item.
    No review is required, so the population differs from that of the main task.
  * Prediction time tau_i = payment approval. The column "t0" holds tau_i so that the rolling-origin and purge
    functions of RQ1 can be reused; the actual hand-over time is kept in "handover_ts".
  * Label y_i = 1 if the delivery date is later than the promised date (the late flag of the main task).
  * Label availability l_i = min(delivery, promised date + 1 day): a late order is known to be late once the promised
    date has passed, an on-time order once it is delivered, so label latency also depends on the outcome.
  * Features admissible at tau: order attributes at purchase, approval hours, promised delivery time, and as-of
    histories of the seller, category and states (known late rates, mean processing time, order volume) and the
    number of earlier orders of the customer.
  * The four RQ1 shortcuts:
      R  - random five-fold split
      NP - no purge on l
      G  - late rates and processing times by seller, category and state over the FULL period (own and future orders)
      P  - variables recorded after tau: the carrier hand-over (processing days, days left on the promise at hand-over,
           ...), the kind of input that late-delivery studies on Olist commonly use
"""
import numpy as np
import pandas as pd

from .features import order_attributes, _asof_cum, _asof_global

DAY = pd.Timedelta(days=1)

T2_NUM = ["f_approval_hours", "f_promised_days", "f_purchase_dow", "f_purchase_hour", "f_purchase_month",
          "f_approval_dow", "f_freight_ratio", "f_same_state", "f_n_items", "f_n_products", "f_n_sellers",
          "f_price_sum", "f_price_max", "f_freight_sum", "f_weight_g", "f_volume_cm3", "f_photos_qty", "f_name_len",
          "f_desc_len", "f_pay_value", "f_n_payments", "f_installments", "f_distance_km"]
T2_CAT = ["c_category", "c_customer_state", "c_seller_state", "c_pay_type"]
T2_HIST_CORE = ["seller_late_n", "seller_laterate", "cat_late_n", "cat_laterate", "cstate_late_n", "cstate_laterate",
                "sstate_late_n", "sstate_laterate", "seller_proc_n", "seller_proc_mean", "seller_vol30",
                "customer_prior_orders"]
T2_ASOF = ["h_" + c for c in T2_HIST_CORE]
T2_GLOBAL = ["gs_" + c for c in T2_HIST_CORE]
T2_GLOBAL_RAW = ["gr_" + c for c in T2_HIST_CORE]
T2_POST = ["p_processing_days", "p_handover_after_approval", "p_remaining_promise_at_handover", "p_handover_dow"]
T2_COLS = {"asof": T2_ASOF, "global": T2_GLOBAL, "global_raw": T2_GLOBAL_RAW, "post": T2_POST}


def t2_feature_set(agg="asof", post=False):
    hist = {"asof": T2_ASOF, "global": T2_GLOBAL, "global_raw": T2_GLOBAL_RAW, "none": []}[agg]
    return T2_NUM + hist + (T2_POST if post else []), T2_CAT


def build_task2(raw, cfg):
    """Return the data frame of the second task and its sample-flow table."""
    m = cfg["features"]["smoothing_m"]
    A = order_attributes(raw)
    flow = [("All orders in the orders table", len(A))]
    S = A[A["order_status"] == "delivered"]
    flow.append(("delivered", len(S)))
    S = S[S["approved_ts"].notna() & S["delivered_ts"].notna() & S["estimated_ts"].notna() & S["purchase_ts"].notna()]
    flow.append(("approval, delivery and promised dates recorded", len(S)))
    S = S[S["seller_id"].notna()]
    flow.append(("at least one item / seller (analysis sample)", len(S)))
    S = S.reset_index(drop=True).copy()
    S["handover_ts"] = S["t0"]
    S["tau"] = S["approved_ts"]
    S["y"] = (S["delivered_ts"].dt.normalize() > S["estimated_ts"]).astype(int)
    S["label_available_ts"] = np.minimum(S["delivered_ts"], S["estimated_ts"].dt.normalize() + DAY)
    S["t0"] = S["tau"]                       # prediction time, stored as t0 so that the RQ1 split functions can be reused
    S["_qid"] = np.arange(len(S))

    # ---------------- features admissible at the approval time
    S["f_approval_hours"] = (S["approved_ts"] - S["purchase_ts"]) / pd.Timedelta(hours=1)
    S["f_promised_days"] = (S["estimated_ts"] - S["purchase_ts"]) / DAY
    S["f_purchase_dow"] = S["purchase_ts"].dt.dayofweek
    S["f_purchase_hour"] = S["purchase_ts"].dt.hour
    S["f_purchase_month"] = S["purchase_ts"].dt.month
    S["f_approval_dow"] = S["approved_ts"].dt.dayofweek
    S["f_freight_ratio"] = S["freight_sum"] / S["price_sum"].replace(0, np.nan)
    S["f_same_state"] = (S["customer_state"] == S["seller_state"]).astype(int)
    for c in ["n_items", "n_products", "n_sellers", "price_sum", "price_max", "freight_sum", "weight_g", "volume_cm3",
              "photos_qty", "name_len", "desc_len", "pay_value", "n_payments", "installments", "distance_km"]:
        S["f_" + c] = S[c]
    for c in ["category", "customer_state", "seller_state", "pay_type"]:
        S["c_" + c] = S[c].fillna("unknown").astype(str)

    # ---------------- history events (all delivered orders in the orders table)
    late_ev = A[A["delivered_ts"].notna() & A["estimated_ts"].notna()].copy()
    late_ev["late"] = (late_ev["delivered_ts"].dt.normalize() > late_ev["estimated_ts"]).astype(float)
    late_ev["late_time"] = np.minimum(late_ev["delivered_ts"], late_ev["estimated_ts"].dt.normalize() + DAY)
    proc_ev = A[A["t0"].notna()].copy()
    proc_ev["proc_days"] = (proc_ev["t0"] - proc_ev["purchase_ts"]) / DAY

    gn, gs = _asof_global(S, late_ev, "tau", "late_time", "late")
    prior = np.where(gn > 0, gs / np.maximum(gn, 1), 0.08)
    for key, name in [("seller_id", "seller"), ("category", "cat"), ("customer_state", "cstate"), ("seller_state", "sstate")]:
        n, s = _asof_cum(S, late_ev, key, "tau", "late_time", "late")
        S[f"h_{name}_late_n"] = n
        S[f"h_{name}_laterate"] = (s + m * prior) / (n + m)
    n, s = _asof_cum(S, proc_ev, "seller_id", "tau", "t0", "proc_days")
    S["h_seller_proc_n"] = n
    S["h_seller_proc_mean"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    pe = A[["seller_id", "purchase_ts"]].copy()
    n_now, _ = _asof_cum(S, pe, "seller_id", "purchase_ts", "purchase_ts")
    S2 = S[["_qid", "seller_id"]].copy()
    S2["q30"] = S["purchase_ts"] - pd.Timedelta(days=cfg["features"]["seller_volume_window_days"])
    n_30, _ = _asof_cum(S2, pe, "seller_id", "q30", "purchase_ts")
    S["h_seller_vol30"] = n_now - n_30
    ce = A[["customer_uid", "purchase_ts"]].copy()
    S["h_customer_prior_orders"], _ = _asof_cum(S, ce, "customer_uid", "purchase_ts", "purchase_ts")

    # ---------------- G: full period, own order and future orders included
    p_all = late_ev["late"].mean()
    for key, name in [("seller_id", "seller"), ("category", "cat"), ("customer_state", "cstate"), ("seller_state", "sstate")]:
        tot = late_ev.groupby(key)["late"].agg(["size", "sum"])
        n_k = S[key].map(tot["size"]).fillna(0).to_numpy()
        s_k = S[key].map(tot["sum"]).fillna(0).to_numpy()
        S[f"gs_{name}_late_n"], S[f"gs_{name}_laterate"] = n_k, (s_k + m * p_all) / (n_k + m)
        S[f"gr_{name}_late_n"], S[f"gr_{name}_laterate"] = n_k, np.where(n_k > 0, s_k / np.maximum(n_k, 1), p_all)
    tp = proc_ev.groupby("seller_id")["proc_days"].agg(["size", "sum"])
    nk, sk = S["seller_id"].map(tp["size"]).fillna(0).to_numpy(), S["seller_id"].map(tp["sum"]).fillna(0).to_numpy()
    for pre in ("gs_", "gr_"):
        S[pre + "seller_proc_n"] = nk
        S[pre + "seller_proc_mean"] = np.where(nk > 0, sk / np.maximum(nk, 1), np.nan)
    tv = A.groupby("seller_id").size()
    tc = A.groupby("customer_uid").size()
    for pre in ("gs_", "gr_"):
        S[pre + "seller_vol30"] = S["seller_id"].map(tv).fillna(0).to_numpy() * 30 / 700.0
        S[pre + "customer_prior_orders"] = S["customer_uid"].map(tc).fillna(1).to_numpy() - 1

    # ---------------- P: variables recorded after approval (carrier hand-over)
    S["p_processing_days"] = (S["handover_ts"] - S["purchase_ts"]) / DAY
    S["p_handover_after_approval"] = (S["handover_ts"] - S["approved_ts"]) / DAY
    S["p_remaining_promise_at_handover"] = (S["estimated_ts"] - S["handover_ts"]) / DAY
    S["p_handover_dow"] = S["handover_ts"].dt.dayofweek

    flow.append(("of which delivered late (y = 1)", int(S["y"].sum())))
    S = S.drop(columns=["_qid"]).copy()
    return S, pd.DataFrame(flow, columns=["step", "n_orders"])
