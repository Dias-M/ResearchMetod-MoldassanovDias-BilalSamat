"""RQ1-RQ3 tests -> results/summary.md and results/summary.csv."""
import pandas as pd

from _common import load_config, p
from phishrobust.stats import holm, logit_rq2, mcnemar_exact, paired_bootstrap_delta_recall

def _table(df):
    try:
        return df.round(4).to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return df.round(4).to_string(index=False)


if __name__ == "__main__":
    cfg = load_config()
    res = p(cfg, "results")
    pred = pd.read_csv(res / "predictions.csv")
    met = pd.read_csv(res / "metrics.csv")
    st = cfg["stats"]
    first_seed = pred.seed.min()          # paired tests use one run; seed variability is reported separately
    rows = []

    for model, g in pred[pred.seed == first_seed].groupby("model"):
        w = g.pivot(index="email_id", columns="version", values="detected")
        mc = mcnemar_exact(w["original"], w["paraphrased"])
        bs = paired_bootstrap_delta_recall(w["original"], w["paraphrased"], st["bootstrap_n"], first_seed, st["alpha"])
        sd = met[(met.model == model)].pivot(index="seed", columns="version", values="recall")
        d = sd["original"] - sd["paraphrased"]
        rows.append({"model": model, "n_pairs": len(w), "recall_orig": w["original"].mean(), "recall_para": w["paraphrased"].mean(),
                     "delta": bs["delta"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "b": mc["b"], "c": mc["c"],
                     "p_one_sided": mc["p_one_sided"], "delta_sd_over_seeds": float(d.std(ddof=0))})
    s = pd.DataFrame(rows)
    s["p_holm"] = holm(s.p_one_sided.tolist())
    s["reject_H0_RQ1"] = s.p_holm < st["alpha"]
    s.to_csv(res / "summary.csv", index=False)

    lines = ["# Results summary", "", "## RQ1 (paired, one-sided, Holm-adjusted across detectors)", "", _table(s)]
    try:
        r2 = logit_rq2(pred[pred.seed == first_seed].assign(detected=lambda d: d.detected.astype(int)))
        lines += ["", "## RQ2 (logit, cluster-robust SE by e-mail)", "", str({k: round(v, 4) for k, v in r2.items()})]
    except ImportError:
        lines += ["", "## RQ2", "", "skipped: statsmodels not installed"]

    aug_path = res / "metrics_aug.csv"
    if aug_path.exists():
        a = pd.read_csv(aug_path)
        base_o = met[met.version == "original"].groupby("model")[["recall", "fpr"]].mean()
        base_p = met[met.version == "paraphrased"].groupby("model")[["recall", "fpr"]].mean()
        aug_p = a[a.version == "paraphrased"].groupby("model")[["recall", "fpr"]].mean()
        r3 = pd.DataFrame({"recall_base_orig": base_o.recall, "recall_base_para": base_p.recall, "recall_aug_para": aug_p.recall,
                           "d_fpr_pp": (aug_p.fpr - base_p.fpr) * 100})
        r3["H1_RQ3_holds"] = (r3.recall_aug_para >= r3.recall_base_orig - 0.02) & (r3.d_fpr_pp <= 1.0)
        lines += ["", "## RQ3 (augmentation)", "", r3.round(4).to_string()]
    (res / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
