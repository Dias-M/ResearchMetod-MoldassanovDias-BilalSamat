"""Step 1-2 of the studio: clean benchmark outputs and aggregate them.

Input : results/metrics.csv, results/metrics_aug.csv, results/predictions*.csv
Output: data/processed/metrics_summary.csv      (one row per model x condition x text version)
        data/processed/paired_delta.csv         (paraphrased minus original recall, per model x condition)
        data/processed/predictions_flagged.csv  (per-email scores with a non-destructive IQR outlier flag)
        data/processed/model_check.csv          (location-shift model vs measured recall, see src/signal_model.py)

Notes
- "warm-up" does not exist in this project (no latency logs), so there is nothing to drop.
  The equivalent here: only the first-class rows of the benchmark are kept; nothing is deleted.
- The error guardrail of the studio is the realised FPR: it is reported, never filtered out.
- Outliers are FLAGGED (1.5 x IQR), never removed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from signal_model import check_cells, summarise_check

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 10_000
FPR_LIMIT = 0.02  # guardrail from docs/methodology_card.md: realised FPR must stay <= 2%


def load_cfg() -> dict:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_all(res: Path, stem: str) -> pd.DataFrame:
    frames = [pd.read_csv(res / f) for f in (f"{stem}.csv", f"{stem}_aug.csv") if (res / f).exists()]
    if not frames:
        raise SystemExit(f"No {stem}*.csv in {res}. Run `python src/benchmark.py --sample` first.")
    df = pd.concat(frames, ignore_index=True)
    df["condition"] = np.where(df["augment"] == 1, "augmented", "baseline")
    return df


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> tuple[float, float]:
    """95% bootstrap CI of the mean over seeds. Undefined (NaN) for fewer than 2 seeds."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion k/n (sampling uncertainty over the test e-mails)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def summarise(metrics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (model, cond, version), g in metrics.groupby(["model", "condition", "version"], sort=False):
        row = {"model": model, "condition": cond, "version": version, "n_seeds": g["seed"].nunique(),
               "n_phish": int(g["n_phish"].iloc[0]), "n_ham": int(g["n_ham"].iloc[0])}
        for col in ("recall", "fpr", "f1", "precision", "roc_auc"):
            x = g[col].to_numpy()
            row[f"{col}_mean"] = x.mean()
            row[f"{col}_sd"] = x.std(ddof=1) if len(x) > 1 else float("nan")
            row[f"{col}_p50"] = np.median(x)
            row[f"{col}_p25"], row[f"{col}_p75"] = np.percentile(x, [25, 75])
        row["recall_ci_lo"], row["recall_ci_hi"] = boot_ci(g["recall"].to_numpy(), rng)
        row["fpr_ci_lo"], row["fpr_ci_hi"] = boot_ci(g["fpr"].to_numpy(), rng)
        row["recall_wilson_lo"], row["recall_wilson_hi"] = wilson(row["recall_mean"] * row["n_phish"], row["n_phish"])
        row["fpr_guardrail_ok"] = bool(row["fpr_mean"] <= FPR_LIMIT)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta(metrics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    wide = metrics.pivot_table(index=["model", "condition", "seed"], columns="version", values="recall").reset_index()
    wide["delta_recall"] = wide["paraphrased"] - wide["original"]
    rows = []
    for (model, cond), g in wide.groupby(["model", "condition"], sort=False):
        lo, hi = boot_ci(g["delta_recall"].to_numpy(), rng)
        rows.append({"model": model, "condition": cond, "n_seeds": len(g),
                     "delta_recall_mean": g["delta_recall"].mean(), "delta_ci_lo": lo, "delta_ci_hi": hi})
    return pd.DataFrame(rows)


def flag_outliers(pred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    thr = metrics.drop_duplicates(["seed", "model", "augment"])[["seed", "model", "augment", "threshold"]]
    df = pred.merge(thr, on=["seed", "model", "augment"], how="left")
    df["margin"] = df["score"] - df["threshold"]  # > 0 means detected
    df["condition"] = np.where(df["augment"] == 1, "augmented", "baseline")
    flags = []
    for _, g in df.groupby(["model", "condition", "version"]):
        q1, q3 = g["margin"].quantile([0.25, 0.75])
        iqr = q3 - q1
        flags.append(((g["margin"] < q1 - 1.5 * iqr) | (g["margin"] > q3 + 1.5 * iqr)).rename("is_outlier"))
    df["is_outlier"] = pd.concat(flags).reindex(df.index)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap itself")
    args = ap.parse_args()
    cfg = load_cfg()
    res, proc = ROOT / cfg["paths"]["results"], ROOT / cfg["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    metrics = read_all(res, "metrics")
    pred = read_all(res, "predictions")

    summary = summarise(metrics, rng)
    delta = paired_delta(metrics, rng)
    flagged = flag_outliers(pred, metrics)

    cells = check_cells(flagged)
    summary.to_csv(proc / "metrics_summary.csv", index=False)
    if len(cells):
        cells.to_csv(proc / "model_check_cells.csv", index=False)
        summarise_check(cells).to_csv(proc / "model_check.csv", index=False)
    delta.to_csv(proc / "paired_delta.csv", index=False)
    flagged.to_csv(proc / "predictions_flagged.csv", index=False)

    print(f"seeds per cell: {sorted(int(v) for v in summary['n_seeds'].unique())}")
    if summary["n_seeds"].min() < 2:
        print("WARNING: fewer than 2 seeds -> bootstrap CI is NaN. Run with several seeds (config.yaml: seeds).")
    print(f"outliers flagged (kept in file): {int(flagged['is_outlier'].sum())} of {len(flagged)}")
    bad = summary[~summary["fpr_guardrail_ok"]]
    if len(bad):
        print(f"WARNING: realised FPR above {FPR_LIMIT:.0%} for {len(bad)} cell(s); see metrics_summary.csv")
    print(f"wrote {proc / 'metrics_summary.csv'}")
