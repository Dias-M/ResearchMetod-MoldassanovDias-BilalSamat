"""Step 3-4 of the studio: publication-ready figures from data/processed/*.csv.

Figure 1  Recall at FPR = 1% on original vs paraphrased phishing, one panel per training condition,
          with 95% bootstrap CI over seeds.
Figure 2  Distribution of the detector margin (score minus threshold) per e-mail, original vs paraphrased.

The three rules of scientific graphics:
  1. Axes: names with units, y starts at 0 for rates, readable ticks and font size.
  2. Variance: 95% CI error bars (Fig. 1); box + individual points, medians and quartiles (Fig. 2).
  3. Colour: colour-blind-safe palette plus distinct markers and line styles (readable in grayscale).
Output: reports/figures/*.pdf (vector) and *.png (300 DPI).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["o", "s", "^", "D"]
LINES = ["-", "--", ":", "-."]
LABELS = {"tfidf_lr": "TF-IDF + LR", "distilbert": "DistilBERT", "zeroshot_llm": "Zero-shot LLM",
          "original": "Original", "paraphrased": "Paraphrased",
          "baseline": "Trained on original only", "augmented": "Trained with paraphrase augmentation"}


def nice(x: str) -> str:
    return LABELS.get(x, x)


def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.set_palette(sns.color_palette("colorblind"))
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")             # vector, for LaTeX / the paper
    fig.savefig(out / f"{name}.png", dpi=300)    # raster, for web / GitHub
    plt.close(fig)
    print(f"wrote {out / name}.pdf and .png")


def fig_recall(summary: pd.DataFrame, target_fpr: float, out: Path) -> None:
    conds = [c for c in ("baseline", "augmented") if c in set(summary["condition"])]
    models = list(dict.fromkeys(summary["model"]))
    pal = sns.color_palette("colorblind", len(models))
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.4), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for i, m in enumerate(models):
            g = summary[(summary.model == m) & (summary.condition == cond)].set_index("version")
            if g.empty:
                continue
            x = np.arange(2)
            y = [g.loc[v, "recall_mean"] for v in ("original", "paraphrased")]
            lo = [g.loc[v, "recall_ci_lo"] for v in ("original", "paraphrased")]
            hi = [g.loc[v, "recall_ci_hi"] for v in ("original", "paraphrased")]
            if "recall_wilson_lo" in g and np.nanmax(np.array(hi) - np.array(lo)) < 1e-9:   # seeds identical (deterministic detector)
                lo = [g.loc[v, "recall_wilson_lo"] for v in ("original", "paraphrased")]
                hi = [g.loc[v, "recall_wilson_hi"] for v in ("original", "paraphrased")]
            yerr = None if np.isnan(lo).any() else [np.array(y) - lo, np.array(hi) - np.array(y)]
            ax.errorbar(x + (i - (len(models) - 1) / 2) * 0.04, y, yerr=yerr, color=pal[i], marker=MARKERS[i % 4],
                        linestyle=LINES[i % 4], capsize=4, elinewidth=1.4, markersize=8, linewidth=2,
                        label=nice(m), alpha=0.95)
        ax.set_xticks([0, 1], [nice("original"), nice("paraphrased")])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Phishing text version")
        ax.set_title(nice(cond), fontsize=12)
    axes[0][0].set_ylabel(f"Recall at FPR = {target_fpr:.0%}\n(share of phishing detected)")
    axes[0][-1].legend(title="Detector", loc="lower left", frameon=True)
    n_seeds = int(summary["n_seeds"].min())
    wilson_used = "recall_wilson_lo" in summary and (summary["recall_ci_hi"] - summary["recall_ci_lo"]).max() < 1e-9
    note = ("95% Wilson CI over test e-mails; seeds identical" if wilson_used
            else "95% bootstrap CI over seeds" if n_seeds >= 2 else "no CI: single seed")
    fig.suptitle(f"Recall before and after LLM paraphrasing ({note}, {n_seeds} seeds)", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig1_recall_original_vs_paraphrased")


def fig_margin(flagged: pd.DataFrame, out: Path) -> None:
    models = list(dict.fromkeys(flagged["model"]))
    df = flagged[flagged.condition == "baseline"].copy()
    if df.empty:
        df = flagged.copy()
    df["Version"] = df["version"].map(nice)
    df["Detector"] = df["model"].map(nice)
    dots = df.sample(frac=1, random_state=0).groupby(["Detector", "Version"]).head(150)
    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(models) + 3), 4.6))
    order = [nice("original"), nice("paraphrased")]
    sns.boxplot(data=df, x="Detector", y="margin", hue="Version", hue_order=order, ax=ax, width=0.6,
                fliersize=0, palette=sns.color_palette("colorblind", 2), boxprops={"alpha": 0.55})
    sns.stripplot(data=dots, x="Detector", y="margin", hue="Version", hue_order=order, ax=ax, dodge=True,
                  size=3.5, alpha=0.6, linewidth=0.4, edgecolor="black", palette=sns.color_palette("colorblind", 2),
                  legend=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.text(1.01, 0, "threshold\n(FPR = 1%)", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Score minus threshold\n(> 0 = detected)")
    ax.legend(title="Phishing text", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True)
    ax.set_title("Score margin per phishing e-mail\nbox: median and quartiles, dots: e-mails", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig2_score_margin_distribution")


def fig_model_check(check: pd.DataFrame, out: Path) -> None:
    """Fig. 3: recall of paraphrased e-mails, measured vs predicted by the location-shift model."""
    models = list(dict.fromkeys(check["model"]))
    cond = "baseline" if "baseline" in set(check["condition"]) else check["condition"].iloc[0]
    g = check[check["condition"] == cond].set_index("model").loc[models]
    pal = sns.color_palette("colorblind", 2)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(5.5, 2.2 * len(models) + 2.5), 4.4))
    ax.bar(x - 0.19, g["recall_para_measured"], 0.36, color=pal[0], edgecolor="black", label="Measured")
    ax.bar(x + 0.19, g["recall_para_model"], 0.36, color=pal[1], edgecolor="black", hatch="//",
           label="Model (location shift)")
    for xi, (m, mo) in enumerate(zip(g["recall_para_measured"], g["recall_para_model"])):
        ax.text(xi - 0.19, m + 0.015, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, mo + 0.015, f"{mo:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([nice(m) for m in models])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Recall of paraphrased phishing\n(at FPR = 1%, mean over seeds)")
    ax.set_title("Model check: predicted vs measured recall", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True)
    fig.tight_layout()
    save(fig, out, "fig3_model_check")


if __name__ == "__main__":
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = ROOT / cfg["paths"]["processed"]
    out = ROOT / "reports" / "figures"
    for needed in ("metrics_summary.csv", "predictions_flagged.csv"):
        if not (proc / needed).exists():
            raise SystemExit(f"{proc / needed} missing. Run `python src/process_data.py` first.")
    set_style()
    fig_recall(pd.read_csv(proc / "metrics_summary.csv"), cfg["target_fpr"], out)
    fig_margin(pd.read_csv(proc / "predictions_flagged.csv"), out)
    if (proc / "model_check.csv").exists():
        fig_model_check(pd.read_csv(proc / "model_check.csv"), out)
