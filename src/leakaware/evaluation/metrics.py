"""Evaluation metrics and the moving-block bootstrap."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def precision_at_k(y, s, frac=0.10):
    y = np.asarray(y); k = max(1, math.ceil(len(y) * frac))
    idx = np.argsort(-np.asarray(s), kind="stable")[:k]
    return y[idx].mean()


def window_metrics(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    p = y.mean()
    ap = average_precision_score(y, s)
    return dict(n=len(y), pos=int(y.sum()), prevalence=p, ap=ap, ap_norm=(ap - p) / (1 - p),
                auc=roc_auc_score(y, s), p_at_10=precision_at_k(y, s, 0.10))


def best_f1_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 200)))
    f = [f1_score(y, s >= t, zero_division=0) for t in qs]
    i = int(np.argmax(f))
    return float(qs[i]), float(f[i])


def moving_block_bootstrap_mean(x, n_boot=2000, block=3, seed=42):
    """Mean of a time-ordered series with a 95% moving-block bootstrap interval."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.array([])
    block = min(block, n)
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), means


def block_indices(n, n_boot=2000, block=3, seed=42):
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
