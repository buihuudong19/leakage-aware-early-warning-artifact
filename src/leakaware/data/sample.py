"""Order-level analysis sample built with explicit rules, together with the sample-flow table.

Rules (parameters in config.yaml):
  R1  order_status == 'delivered'
  R2  order_delivered_carrier_date is recorded (it defines the prediction time t0)
  R3  at least one review
  R4  orders with several reviews: score = minimum (worst review); label availability = last answer
  R5  drop orders with any review answered before t0 (feedback already existed at prediction time)
  R6  main label: y = 1 for scores {1, 2}, y = 0 for {4, 5}; three-star orders are kept with is_neutral = 1
"""
import numpy as np
import pandas as pd


def aggregate_reviews(reviews, score_rule="min"):
    r = reviews.copy()
    g = r.groupby("order_id")
    agg = pd.DataFrame({
        "n_reviews": g.size(),
        "review_score": g["review_score"].agg(score_rule),
        "first_answer_ts": g["review_answer_timestamp"].min(),
        "last_answer_ts": g["review_answer_timestamp"].max(),
        "first_creation_ts": g["review_creation_date"].min(),
    }).reset_index()
    return agg


def build_sample(raw, cfg):
    sc = cfg["sample"]
    o = raw["orders"].copy()
    flow = [("All orders in the orders table", len(o))]
    o = o[o["order_status"] == "delivered"]
    flow.append(("R1: status 'delivered'", len(o)))
    o = o[o["order_delivered_carrier_date"].notna()]
    flow.append(("R2: carrier hand-over time (t0) recorded", len(o)))
    rev = aggregate_reviews(raw["reviews"], sc["multi_review_score"])
    o = o.merge(rev, on="order_id", how="inner")
    flow.append(("R3: at least one review", len(o)))
    n_multi = int((o["n_reviews"] > 1).sum())
    first_col = {"first_answer": "first_answer_ts", "last_answer": "last_answer_ts"}[sc["exclude_if_feedback_before_t0"]]
    before = o[first_col] < o["order_delivered_carrier_date"]
    n_before, n_before_low = int(before.sum()), int((before & o["review_score"].isin(sc["low_scores"])).sum())
    o = o[~before]
    flow.append(("R5: no review answered before t0", len(o)))
    avail_col = {"first_answer": "first_answer_ts", "last_answer": "last_answer_ts"}[sc["label_available_ts"]]
    o["label_available_ts"] = o[avail_col]
    o["is_neutral"] = (o["review_score"] == 3).astype(int)
    o["y"] = np.where(o["review_score"].isin(sc["low_scores"]), 1,
                      np.where(o["review_score"].isin(sc["high_scores"]), 0, -1))
    o = o.rename(columns={"order_purchase_timestamp": "purchase_ts", "order_approved_at": "approved_ts",
                          "order_delivered_carrier_date": "t0", "order_delivered_customer_date": "delivered_ts",
                          "order_estimated_delivery_date": "estimated_ts"})
    primary = o[o["is_neutral"] == 0]
    flow.append(("R6: main analysis sample (three-star orders removed)", len(primary)))
    flow_df = pd.DataFrame(flow, columns=["step", "n_orders"])
    flow_df["removed"] = (-flow_df["n_orders"].diff()).fillna(0).astype(int)
    notes = dict(orders_multi_review=n_multi, excluded_feedback_before_t0=n_before,
                 excluded_feedback_before_t0_low=n_before_low,
                 neutral_3star=int(o["is_neutral"].sum()),
                 primary_n=int(len(primary)), primary_pos=int(primary["y"].sum()),
                 primary_prevalence=float(primary["y"].mean()))
    keep = ["order_id", "customer_id", "purchase_ts", "approved_ts", "t0", "delivered_ts", "estimated_ts",
            "n_reviews", "review_score", "first_answer_ts", "last_answer_ts", "first_creation_ts",
            "label_available_ts", "is_neutral", "y"]
    return o[keep].sort_values("t0").reset_index(drop=True), flow_df, notes
