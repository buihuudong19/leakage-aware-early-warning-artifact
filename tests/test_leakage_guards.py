"""Tests of the leakage guards of the pipeline.

    python -m pytest tests -q        (or: python tests/test_leakage_guards.py)

Requires the feature file: run `python run_pipeline.py --stage sample` first.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakaware.config import load_config                                    # noqa: E402
from leakaware.data.io import load_raw, load_features                       # noqa: E402
from leakaware.data.features import (feature_set, POST, HIST_GLOBAL, HIST_GLOBAL_RAW,  # noqa: E402
                                     HIST_GLOBAL_LOO)
from leakaware.evaluation.splits import rolling_windows, temporal_train_mask, eval_mask  # noqa: E402

CFG = load_config()
F = load_features(CFG["out_dir"])


def test_sample_rules():
    assert F["order_id"].is_unique
    assert (F["first_answer_ts"] >= F["t0"]).all(), "an order with feedback before t0 remains"
    assert set(F["y"].unique()) <= {-1, 0, 1}
    assert ((F["y"] == -1) == (F["review_score"] == 3)).all()


def test_asof_seller_history_matches_bruteforce():
    raw, _ = load_raw(CFG["data_dir"], verify=False)
    rv = raw["reviews"].groupby("order_id").agg(score=("review_score", "min"), avail=("review_answer_timestamp", "max"))
    it = raw["items"].sort_values(["order_id", "order_item_id"]).groupby("order_id")["seller_id"].first()
    ev = rv.join(it, how="inner").reset_index()
    ev["low"] = (ev["score"] <= 2).astype(float)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(F), 200, replace=False):
        r = F.iloc[i]
        past = ev[(ev["seller_id"] == r["seller_id"]) & (ev["avail"] < r["t0"])]
        assert len(past) == r["h_seller_n"], (r["order_id"], len(past), r["h_seller_n"])


def test_no_future_label_in_purged_training():
    D = F[F["y"] >= 0]
    for w in rolling_windows(CFG):
        tr = temporal_train_mask(D, w["origin"], purge=True)
        assert (D.loc[tr, "label_available_ts"] < w["origin"]).all()
        assert (D.loc[tr, "t0"] < w["origin"]).all()
        ev = eval_mask(D, w)
        assert not (tr & ev).any()


def test_reference_features_exclude_leaky_sets():
    num, cat = feature_set("asof", post=False)
    used = set(num + cat)
    leaky = set(POST) | set(HIST_GLOBAL) | set(HIST_GLOBAL_RAW) | set(HIST_GLOBAL_LOO)
    assert not used & leaky, sorted(used & leaky)
    assert all(c in F.columns for c in used)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
