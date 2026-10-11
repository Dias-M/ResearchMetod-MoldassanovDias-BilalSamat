"""Metrics at a fixed operating point (target FPR)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def threshold_at_fpr(ham_scores, target_fpr: float) -> float:
    """Smallest threshold t such that the share of ham with score > t is <= target_fpr.

    Computed on VALIDATION ham of the original data; the same t is then reused for every
    other condition, so the only thing that changes between conditions is the phishing text.
    """
    ham = np.sort(np.asarray(ham_scores, dtype=float))
    k = int(np.floor(target_fpr * len(ham)))      # number of ham allowed above t
    if k == 0:
        return float(ham[-1])                      # nothing above the max ham score
    return float(ham[-(k + 1)])                    # k scores strictly above this one (ties aside)


def evaluate(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    ham = y == 0
    out = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(pred[ham].mean()) if ham.any() else float("nan"),
    }
    out["roc_auc"] = float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")
    return out
