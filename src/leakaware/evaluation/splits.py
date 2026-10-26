"""Data splits: rolling origins on t0 (with or without the label-availability purge) and random K-fold splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def rolling_windows(cfg):
    r = cfg["rolling"]
    origins = pd.date_range(r["first_origin"], r["last_origin"], freq=f"{r['step_days']}D")
    H = pd.Timedelta(days=r["horizon_days"])
    return [dict(k=i + 1, origin=a, end=a + H) for i, a in enumerate(origins)]


def temporal_train_mask(D, origin, purge=True):
    """Training mask: orders with t0 < origin; with the purge, only orders whose label was available before the origin."""
    m = D["t0"] < origin
    if purge:
        m &= D["label_available_ts"] < origin
    return m.to_numpy()


def eval_mask(D, w):
    return ((D["t0"] >= w["origin"]) & (D["t0"] < w["end"])).to_numpy()


def random_folds(D, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(D), int)
    for i, (_, te) in enumerate(skf.split(np.zeros(len(D)), D["y"].to_numpy())):
        fold[te] = i
    return fold
