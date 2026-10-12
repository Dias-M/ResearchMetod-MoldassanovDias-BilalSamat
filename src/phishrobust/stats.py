"""Statistical tests for H0/H1 of RQ1-RQ3."""
from __future__ import annotations

from math import comb

import numpy as np


def mcnemar_exact(orig_detected, para_detected) -> dict:
    """Exact (binomial) McNemar test on paired detections of the same phishing e-mails.

    b = detected originally but missed after paraphrasing; c = the opposite.
    Two-sided p-value; also returns the one-sided p for H1: detection drops (b > c).
    """
    o = np.asarray(orig_detected).astype(bool)
    p = np.asarray(para_detected).astype(bool)
    b = int((o & ~p).sum())
    c = int((~o & p).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "p_one_sided": 1.0}
    tail_b = sum(comb(n, k) for k in range(b, n + 1)) / 2**n          # P(X >= b)
    tail_small = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return {"b": b, "c": c, "p_two_sided": min(1.0, 2 * tail_small), "p_one_sided": tail_b}


def paired_bootstrap_delta_recall(orig_detected, para_detected, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Delta = recall(original) - recall(paraphrased); resample e-mails (pairs) with replacement."""
    o = np.asarray(orig_detected).astype(float)
    p = np.asarray(para_detected).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(o), size=(n_boot, len(o)))
    deltas = o[idx].mean(axis=1) - p[idx].mean(axis=1)
    return {
        "delta": float(o.mean() - p.mean()),
        "ci_low": float(np.quantile(deltas, alpha / 2)),
        "ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def logit_rq2(df):
    """RQ2: detected ~ version + model + dataset + log(length), cluster-robust SE by e-mail.

    df columns: detected (0/1), version ('original'/'paraphrased'), model, source, length, email_id.
    H0: beta(version=paraphrased) = 0; H1: beta < 0.
    """
    import statsmodels.formula.api as smf  # imported lazily: only needed for RQ2

    d = df.copy()
    d["log_len"] = np.log(d["length"].clip(lower=1))
    d["paraphrased"] = (d["version"] == "paraphrased").astype(int)
    fit = smf.logit("detected ~ paraphrased + C(model) + C(source) + log_len", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["email_id"].astype("category").cat.codes}
    )
    beta, se = fit.params["paraphrased"], fit.bse["paraphrased"]
    p_two = float(fit.pvalues["paraphrased"])
    p_one = p_two / 2 if beta < 0 else 1 - p_two / 2
    return {"beta": float(beta), "se": float(se), "odds_ratio": float(np.exp(beta)), "p_one_sided": float(p_one)}
