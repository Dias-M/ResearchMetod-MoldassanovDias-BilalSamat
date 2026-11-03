"""Mathematical model of the experiment: a location-shift model of detector scores.

Setting. A detector gives every e-mail a score s. A mail is flagged if s > t, where t is fixed on
validation ham at FPR = 1% (config: target_fpr). For one detector the scores of the ORIGINAL phishing
e-mails are modelled as s ~ N(mu, sigma^2). Then
    recall_orig = P(s > t) = 1 - Phi((t - mu) / sigma)            (1)
LLM paraphrasing is modelled as a pure LOCATION shift of the score by delta (mean paired difference
s_para - s_orig), with the same sigma:
    recall_para = 1 - Phi((t - mu - delta) / sigma)               (2)
Prediction (2) uses only the first two moments of the original scores, the threshold and the paired mean
shift; it is NOT fitted to the paraphrased recall, so comparing it with the measured recall is a real test
of the model. Its assumptions (normal scores, shift only, no change of spread) are checked in the notebook.

Useful by-product, the break-even shift that costs d percentage points of recall:
    delta*(d) = (t - mu) - sigma * Phi^-1(1 - (recall_orig - d))   (negative = score goes down)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_normal(scores) -> tuple[float, float]:
    x = np.asarray(scores, dtype=float)
    return float(x.mean()), float(x.std(ddof=1))


def recall_normal(mu: float, sigma: float, t: float) -> float:
    """Eq. (1)/(2): P(N(mu, sigma^2) > t)."""
    return float(1.0 - norm.cdf((t - mu) / sigma))


def predicted_recall_after_shift(mu: float, sigma: float, t: float, delta: float) -> float:
    return recall_normal(mu + delta, sigma, t)


def breakeven_shift(mu: float, sigma: float, t: float, drop_pp: float = 2.0) -> float:
    """Score shift that lowers the modelled recall by `drop_pp` percentage points."""
    r0 = recall_normal(mu, sigma, t)
    target = max(r0 - drop_pp / 100.0, 1e-9)
    return float((t - mu) - sigma * norm.ppf(1.0 - target))


def check_cells(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, seed): measured vs modelled recall.

    `pred` = predictions_flagged.csv (columns seed, model, condition, version, email_id, score, threshold).
    """
    rows = []
    for (model, cond, seed), g in pred.groupby(["model", "condition", "seed"], sort=False):
        o = g[g.version == "original"].set_index("email_id")["score"]
        p = g[g.version == "paraphrased"].set_index("email_id")["score"]
        ids = o.index.intersection(p.index)
        if len(ids) < 10:
            continue
        o, p = o.loc[ids], p.loc[ids]
        t = float(g["threshold"].iloc[0])
        mu, sd = fit_normal(o)
        diff = p - o
        delta = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append(dict(
            model=model, condition=cond, seed=seed, n_pairs=len(ids), threshold=t, mu=mu, sigma=sd, delta=delta, delta_ci_lo=delta - 1.96 * se, delta_ci_hi=delta + 1.96 * se,
            median_margin_orig=float((o - t).median()),
            sigma_ratio_para_orig=float(p.std(ddof=1) / sd) if sd > 0 else np.nan,
            recall_orig_measured=float((o > t).mean()), recall_orig_model=recall_normal(mu, sd, t),
            recall_para_measured=float((p > t).mean()),
            recall_para_model=predicted_recall_after_shift(mu, sd, t, delta),
            breakeven_shift_2pp=breakeven_shift(mu, sd, t, 2.0),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_error_pp"] = 100 * (out["recall_para_model"] - out["recall_para_measured"]).abs()
    return out


def summarise_check(cells: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (model, condition)."""
    cols = ["recall_orig_measured", "recall_orig_model", "recall_para_measured", "recall_para_model",
            "abs_error_pp", "delta", "delta_ci_lo", "delta_ci_hi", "median_margin_orig", "sigma_ratio_para_orig", "breakeven_shift_2pp"]
    return cells.groupby(["model", "condition"], sort=False)[cols].mean().reset_index()
