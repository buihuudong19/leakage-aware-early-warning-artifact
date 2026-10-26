"""Feature store at the prediction time t0 (as-of features) and the leaky variants used for RQ1.

Feature groups
  BASE            : order information known at or before t0
  HIST_ASOF       : seller / category / state histories computed only from events before each order's t0
  HIST_GLOBAL     : the same quantities computed over the FULL period (future orders and the order itself included),
                    smoothed by an m-estimate; the main, conservative version of the common full-period shortcut
  HIST_GLOBAL_RAW : as above but raw means without smoothing, as in most public notebooks
  HIST_GLOBAL_LOO : full period, leave-one-out; known to reveal the label to tree models
  POST            : variables known only after delivery (actual delivery time, delay against the promise, late flag)
"""
import numpy as np
import pandas as pd

REGION = {**{s: "Norte" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
          **{s: "Nordeste" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
          **{s: "Centro-Oeste" for s in ["DF", "GO", "MT", "MS"]},
          **{s: "Sudeste" for s in ["ES", "MG", "RJ", "SP"]},
          **{s: "Sul" for s in ["PR", "RS", "SC"]}}
DAY = pd.Timedelta(days=1)


def _haversine(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = r2 - r1, np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def order_attributes(raw):
    """Order-level attributes for ALL orders (keys for the history events and inputs to the features)."""
    o = raw["orders"].rename(columns={"order_purchase_timestamp": "purchase_ts", "order_approved_at": "approved_ts",
                                      "order_delivered_carrier_date": "t0", "order_delivered_customer_date": "delivered_ts",
                                      "order_estimated_delivery_date": "estimated_ts"})
    it = raw["items"].merge(raw["products"], on="product_id", how="left") \
        .merge(raw["translation"], on="product_category_name", how="left") \
        .merge(raw["sellers"], on="seller_id", how="left") \
        .sort_values(["order_id", "order_item_id"], kind="stable")
    it["volume_cm3"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    g = it.groupby("order_id")
    first = g.first()
    items = pd.DataFrame({
        "n_items": g.size(), "n_products": g["product_id"].nunique(), "n_sellers": g["seller_id"].nunique(),
        "price_sum": g["price"].sum(), "price_max": g["price"].max(), "freight_sum": g["freight_value"].sum(),
        "seller_id": first["seller_id"], "seller_state": first["seller_state"],
        "seller_zip": first["seller_zip_code_prefix"],
        "category": first["product_category_name_english"].fillna("unknown"),
        "weight_g": first["product_weight_g"], "volume_cm3": first["volume_cm3"],
        "photos_qty": first["product_photos_qty"], "name_len": first["product_name_lenght"],
        "desc_len": first["product_description_lenght"],
    })
    pay = raw["payments"]
    pg = pay.groupby("order_id")
    main_type = pay.sort_values(["order_id", "payment_value"], ascending=[True, False]).groupby("order_id")["payment_type"].first()
    pays = pd.DataFrame({"pay_value": pg["payment_value"].sum(), "n_payments": pg.size(),
                         "installments": pg["payment_installments"].max(), "pay_type": main_type})
    c = raw["customers"].set_index("customer_id")
    df = o.set_index("order_id").join(items, how="left").join(pays, how="left")
    df["customer_state"] = df["customer_id"].map(c["customer_state"])
    df["customer_zip"] = df["customer_id"].map(c["customer_zip_code_prefix"])
    df["customer_uid"] = df["customer_id"].map(c["customer_unique_id"])
    geo = raw["geo"].groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    df["distance_km"] = _haversine(df["customer_zip"].map(geo["geolocation_lat"]), df["customer_zip"].map(geo["geolocation_lng"]),
                                   df["seller_zip"].map(geo["geolocation_lat"]), df["seller_zip"].map(geo["geolocation_lng"]))
    df["customer_region"] = df["customer_state"].map(REGION)
    df["seller_region"] = df["seller_state"].map(REGION)
    return df.reset_index()


def _asof_cum(queries, events, key, qtime, etime, value=None):
    """For each query, the number of events of the same key strictly before the query time and the sum of their values."""
    ev = events[[key, etime] + ([value] if value else [])].dropna(subset=[key, etime]).sort_values(etime, kind="stable").copy()
    ev["_n"] = ev.groupby(key).cumcount() + 1
    ev["_s"] = ev.groupby(key)[value].cumsum() if value else 0.0
    q = queries[["_qid", key, qtime]].copy()
    q[key] = q[key].fillna("__NA__")
    ev[key] = ev[key].astype(q[key].dtype) if ev[key].dtype != q[key].dtype else ev[key]
    q = q.dropna(subset=[qtime]).sort_values(qtime, kind="stable")
    m = pd.merge_asof(q, ev[[key, etime, "_n", "_s"]].rename(columns={etime: "_et"}), left_on=qtime, right_on="_et",
                      by=key, direction="backward", allow_exact_matches=False)
    m = m.set_index("_qid").reindex(queries["_qid"])
    return m["_n"].fillna(0).to_numpy(), m["_s"].fillna(0).to_numpy()


def _asof_global(queries, events, qtime, etime, value):
    ev = events[[etime, value]].dropna().sort_values(etime, kind="stable").copy()
    ev["_n"] = np.arange(1, len(ev) + 1)
    ev["_s"] = ev[value].cumsum()
    q = queries[["_qid", qtime]].dropna().sort_values(qtime, kind="stable")
    m = pd.merge_asof(q, ev[[etime, "_n", "_s"]].rename(columns={etime: "_et"}), left_on=qtime, right_on="_et",
                      direction="backward", allow_exact_matches=False).set_index("_qid").reindex(queries["_qid"])
    return m["_n"].fillna(0).to_numpy(), m["_s"].fillna(0).to_numpy()


def build_features(sample, raw, cfg):
    m = cfg["features"]["smoothing_m"]
    attrs = order_attributes(raw)
    A = attrs.set_index("order_id")
    S = sample.copy()
    S["_qid"] = np.arange(len(S))
    for col in ["seller_id", "seller_state", "category", "customer_state", "customer_region", "seller_region",
                "customer_uid", "n_items", "n_products", "n_sellers", "price_sum", "price_max", "freight_sum",
                "weight_g", "volume_cm3", "photos_qty", "name_len", "desc_len", "pay_value", "n_payments",
                "installments", "pay_type", "distance_km"]:
        S[col] = S["order_id"].map(A[col])

    # ---------------- BASE (known at or before t0)
    appr_ok = S["approved_ts"].notna() & (S["approved_ts"] <= S["t0"])
    S["f_approval_hours"] = np.where(appr_ok, (S["approved_ts"] - S["purchase_ts"]) / pd.Timedelta(hours=1), np.nan)
    S["f_approval_after_t0"] = (S["approved_ts"] > S["t0"]).astype(int)
    S["f_processing_days"] = (S["t0"] - S["purchase_ts"]) / DAY
    S["f_promised_days"] = (S["estimated_ts"] - S["purchase_ts"]) / DAY
    S["f_remaining_promise_days"] = (S["estimated_ts"] - S["t0"]) / DAY
    S["f_purchase_dow"] = S["purchase_ts"].dt.dayofweek
    S["f_purchase_hour"] = S["purchase_ts"].dt.hour
    S["f_purchase_month"] = S["purchase_ts"].dt.month
    S["f_t0_dow"] = S["t0"].dt.dayofweek
    S["f_freight_ratio"] = S["freight_sum"] / S["price_sum"].replace(0, np.nan)
    S["f_same_state"] = (S["customer_state"] == S["seller_state"]).astype(int)
    for c in ["n_items", "n_products", "n_sellers", "price_sum", "price_max", "freight_sum", "weight_g", "volume_cm3",
              "photos_qty", "name_len", "desc_len", "pay_value", "n_payments", "installments", "distance_km"]:
        S["f_" + c] = S[c]
    for c in ["category", "customer_state", "seller_state", "pay_type"]:
        S["c_" + c] = S[c].fillna("unknown").astype(str)

    # ---------------- history events (from ALL reviewed / delivered orders, not only the sample)
    rv = raw["reviews"].groupby("order_id").agg(score=("review_score", "min"), avail=("review_answer_timestamp", "max"))
    ev = attrs.set_index("order_id").join(rv, how="inner").reset_index()
    ev["low"] = (ev["score"] <= 2).astype(float)
    late_ev = attrs[attrs["delivered_ts"].notna() & attrs["t0"].notna()].copy()
    late_ev["late"] = (late_ev["delivered_ts"].dt.normalize() > late_ev["estimated_ts"]).astype(float)
    late_ev["late_time"] = np.maximum(late_ev["delivered_ts"], late_ev["t0"] + pd.Timedelta(seconds=1))
    proc_ev = attrs[attrs["t0"].notna()].copy()
    proc_ev["proc_days"] = (proc_ev["t0"] - proc_ev["purchase_ts"]) / DAY

    # platform-wide as-of rate (prior of the m-estimate), also restricted to information before t0
    gn, gs = _asof_global(S, ev, "t0", "avail", "low")
    prior = np.where(gn > 0, gs / np.maximum(gn, 1), 0.13)
    for key, name in [("seller_id", "seller"), ("category", "cat"), ("customer_state", "cstate"), ("seller_state", "sstate")]:
        n, s = _asof_cum(S, ev, key, "t0", "avail", "low")
        S[f"h_{name}_n"] = n
        S[f"h_{name}_lowrate"] = (s + m * prior) / (n + m)
    n, s = _asof_cum(S, late_ev, "seller_id", "t0", "late_time", "late")
    S["h_seller_late_n"] = n
    S["h_seller_laterate"] = (s + 5 * 0.08) / (n + 5)
    n, s = _asof_cum(S, proc_ev, "seller_id", "t0", "t0", "proc_days")
    S["h_seller_proc_mean"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    # recent order volume of the seller (by purchase time) and the number of earlier orders of the customer
    pe = attrs[["seller_id", "purchase_ts"]].copy()
    n_now, _ = _asof_cum(S, pe, "seller_id", "purchase_ts", "purchase_ts")
    S2 = S[["_qid", "seller_id"]].copy()
    S2["q30"] = S["purchase_ts"] - pd.Timedelta(days=cfg["features"]["seller_volume_window_days"])
    n_30, _ = _asof_cum(S2, pe, "seller_id", "q30", "purchase_ts")
    S["h_seller_vol30"] = n_now - n_30
    ce = attrs[["customer_uid", "purchase_ts"]].copy()
    S["h_customer_prior_orders"], _ = _asof_cum(S, ce, "customer_uid", "purchase_ts", "purchase_ts")

    # ---------------- HIST_GLOBAL (leaky: computed over the full period, including future orders)
    p_all = ev["low"].mean()
    S["_own_low"] = S["order_id"].map(ev.set_index("order_id")["low"]).fillna(0).to_numpy()
    S["_own_in"] = S["order_id"].isin(ev["order_id"]).astype(int).to_numpy()
    for key, name in [("seller_id", "seller"), ("category", "cat"), ("customer_state", "cstate"), ("seller_state", "sstate")]:
        tot = ev.groupby(key)["low"].agg(["size", "sum"])
        n_k = S[key].map(tot["size"]).fillna(0).to_numpy()
        s_k = S[key].map(tot["sum"]).fillna(0).to_numpy()
        n_loo, s_loo = n_k - S["_own_in"], s_k - S["_own_low"]
        S[f"g_{name}_n"] = n_loo
        S[f"g_{name}_lowrate"] = (s_loo + m * p_all) / (n_loo + m)
        S[f"gs_{name}_n"] = n_k
        S[f"gs_{name}_lowrate"] = (s_k + m * p_all) / (n_k + m)
        S[f"gr_{name}_n"] = n_k
        S[f"gr_{name}_lowrate"] = np.where(n_k > 0, s_k / np.maximum(n_k, 1), p_all)
    tl = late_ev.groupby("seller_id")["late"].agg(["size", "sum"])
    own_late = S["order_id"].map(late_ev.set_index("order_id")["late"]).fillna(0).to_numpy()
    own_late_in = S["order_id"].isin(late_ev["order_id"]).astype(int).to_numpy()
    nk, sk = S["seller_id"].map(tl["size"]).fillna(0).to_numpy(), S["seller_id"].map(tl["sum"]).fillna(0).to_numpy()
    S["g_seller_late_n"], S["g_seller_laterate"] = nk - own_late_in, (sk - own_late + 5 * 0.08) / (nk - own_late_in + 5)
    S["gs_seller_late_n"], S["gs_seller_laterate"] = nk, (sk + 5 * 0.08) / (nk + 5)
    S["gr_seller_late_n"], S["gr_seller_laterate"] = nk, np.where(nk > 0, sk / np.maximum(nk, 1), 0.08)
    tp = proc_ev.groupby("seller_id")["proc_days"].agg(["size", "sum"])
    nk, sk = S["seller_id"].map(tp["size"]).fillna(0).to_numpy(), S["seller_id"].map(tp["sum"]).fillna(0).to_numpy()
    own_p = S["f_processing_days"].to_numpy()
    S["g_seller_proc_mean"] = np.where(nk > 1, (sk - own_p) / np.maximum(nk - 1, 1), np.nan)
    S["gs_seller_proc_mean"] = np.where(nk > 0, sk / np.maximum(nk, 1), np.nan)
    S["gr_seller_proc_mean"] = S["gs_seller_proc_mean"]
    tv = attrs.groupby("seller_id").size()
    S["g_seller_vol30"] = S["seller_id"].map(tv).fillna(0).to_numpy() * 30 / 700.0   # full-period volume scaled to 30 days
    S["gs_seller_vol30"] = S["g_seller_vol30"]
    S["gr_seller_vol30"] = S["g_seller_vol30"]
    tc = attrs.groupby("customer_uid").size()
    S["g_customer_prior_orders"] = S["customer_uid"].map(tc).fillna(1).to_numpy() - 1
    S["gs_customer_prior_orders"] = S["g_customer_prior_orders"]
    S["gr_customer_prior_orders"] = S["g_customer_prior_orders"]

    S = S.copy()
    # ---------------- POST (known only after delivery)
    S["p_delivery_days"] = (S["delivered_ts"] - S["purchase_ts"]) / DAY
    S["p_transit_days"] = (S["delivered_ts"] - S["t0"]) / DAY
    S["p_delay_vs_estimate"] = (S["delivered_ts"] - S["estimated_ts"]) / DAY
    S["p_late"] = (S["delivered_ts"].dt.normalize() > S["estimated_ts"]).astype(int)
    S = S.drop(columns=["_qid", "_own_low", "_own_in"])
    return S


BASE_NUM = ["f_approval_hours", "f_approval_after_t0", "f_processing_days", "f_promised_days", "f_remaining_promise_days",
            "f_purchase_dow", "f_purchase_hour", "f_purchase_month", "f_t0_dow", "f_freight_ratio", "f_same_state",
            "f_n_items", "f_n_products", "f_n_sellers", "f_price_sum", "f_price_max", "f_freight_sum", "f_weight_g",
            "f_volume_cm3", "f_photos_qty", "f_name_len", "f_desc_len", "f_pay_value", "f_n_payments",
            "f_installments", "f_distance_km"]
BASE_CAT = ["c_category", "c_customer_state", "c_seller_state", "c_pay_type"]
HIST_CORE = ["seller_n", "seller_lowrate", "cat_n", "cat_lowrate", "cstate_n", "cstate_lowrate", "sstate_n",
             "sstate_lowrate", "seller_late_n", "seller_laterate", "seller_proc_mean", "seller_vol30",
             "customer_prior_orders"]
HIST_ASOF = ["h_" + c for c in HIST_CORE]
HIST_GLOBAL_LOO = ["g_" + c for c in HIST_CORE]      # full period, leave-one-out
HIST_GLOBAL = ["gs_" + c for c in HIST_CORE]          # full period, own order included, m-estimate smoothing (main, conservative variant)
HIST_GLOBAL_RAW = ["gr_" + c for c in HIST_CORE]      # full period, own order included, raw mean (common practice)
POST = ["p_delivery_days", "p_transit_days", "p_delay_vs_estimate", "p_late"]


def feature_set(agg="asof", post=False):
    hist = {"asof": HIST_ASOF, "global": HIST_GLOBAL, "global_raw": HIST_GLOBAL_RAW, "global_loo": HIST_GLOBAL_LOO,
            "none": []}[agg]
    num = BASE_NUM + hist + (POST if post else [])
    return num, BASE_CAT
