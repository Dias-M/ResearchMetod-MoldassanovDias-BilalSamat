"""Build reports/TASK3_report.pdf from the REAL results of this repository.

Every number is read from data/processed/*.csv and results/summary.csv; nothing is typed by hand.
Refuses to run on a toy run (fewer than 2 seeds or fewer than 30 phishing pairs), so the report cannot
be produced from the 10-row sample by accident.

  python src/make_report.py --group "XX-000" --repo "https://github.com/<user>/<repo>"
"""
import argparse
from pathlib import Path

import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "reports" / "figures"
NAMES = {"tfidf_lr": "TF-IDF + LR", "distilbert": "DistilBERT", "zeroshot_llm": "Zero-shot LLM", "zero_shot_llm": "Zero-shot LLM"}
nm = lambda m: NAMES.get(m, m)


def need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run the experiment and `python src/process_data.py` first (docs/task3_runbook.md).")
    return pd.read_csv(path)


def img(name: str, width_cm: float):
    from PIL import Image as PI
    w, h = PI.open(FIG / f"{name}.png").size
    return Image(str(FIG / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w)


def main(group: str, repo: str) -> None:
    summ, delta, check = need(PROC / "metrics_summary.csv"), need(PROC / "paired_delta.csv"), need(PROC / "model_check.csv")
    flagged = need(PROC / "predictions_flagged.csv")
    if summ["n_seeds"].min() < 2 or summ["n_phish"].min() < 30:
        raise SystemExit("Too small to report: need >= 2 seeds and >= 30 matched phishing e-mails per cell. Run the full experiment.")
    rq1 = pd.read_csv(RES / "summary.csv") if (RES / "summary.csv").exists() else None

    ss = getSampleStyleSheet()
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
    C = ParagraphStyle("c", parent=B, fontSize=8.5, textColor=colors.HexColor("#333333"))
    T = ParagraphStyle("t", parent=ss["Title"], fontSize=17, leading=20)
    P = lambda t, st=B: Paragraph(t, st)
    tab = lambda rows, w: _table(rows, w)

    base = summ[summ.condition == "baseline"]
    n_ph, n_ham, n_seed = int(base.n_phish.iloc[0]), int(base.n_ham.iloc[0]), int(base.n_seeds.iloc[0])
    models = list(dict.fromkeys(summ.model))
    el = [P("TASK 3: Research Results, Data Analysis and Scientific Visualization", T),
          P("Robustness of text-based phishing detectors to LLM paraphrasing (Cybersecurity track)"),
          P(f"Authors: Moldassanov Dias, Bilal Samat | Group: {group}<br/>GitHub repository: {repo}"),
          P("1. Goal and summary", H),
          P(f"We measure how much LLM paraphrasing of phishing e-mails lowers the recall of text detectors at a fixed false-positive rate of 1%. "
            f"Paired design: the same {n_ph} test phishing e-mails are scored in original and paraphrased form by the same detector, "
            f"{n_seed} seeds, {n_ham} legitimate test e-mails; the threshold is fixed on validation ham of the original data."),
          ]
    # main result sentences, computed
    lines = []
    for m in models:
        r = base[(base.model == m)].set_index("version")
        d = delta[(delta.model == m) & (delta.condition == "baseline")].iloc[0]
        lines.append(f"{nm(m)}: recall {r.loc['original','recall_mean']:.3f} on original vs {r.loc['paraphrased','recall_mean']:.3f} on paraphrased "
                     f"(paired change {d.delta_recall_mean*100:+.1f} p.p., 95% CI over seeds [{d.delta_ci_lo*100:+.1f}, {d.delta_ci_hi*100:+.1f}])")
    el += [P("<b>Main result.</b> " + "; ".join(lines) + "."), P("2. Method", H),
           P("<b>Data hygiene.</b> Deduplication by SHA-1 of normalised text, removal of conflicting labels, stratified 70/15/15 split by label and source, "
             "spam excluded from the negative class (src/phishrobust/data.py). <b>Outliers</b> are flagged by the 1.5 x IQR rule on the score margin "
             f"(score minus threshold) and kept: {int(flagged.is_outlier.sum())} of {len(flagged)} rows flagged. There is no warm-up phase in this "
             "project (no latency measurements), so nothing is dropped for warm-up."),
           P("<b>Statistics.</b> Mean over seeds with a 95% bootstrap CI (10,000 resamples) over seeds; exact McNemar test on the paired detections "
             "(one-sided, H1: detection drops), Holm correction across detectors; guardrail: realised FPR must not exceed 2%."),
           P("<b>Mathematical model.</b> Scores of original phishing are modelled as normal, paraphrasing as a pure location shift "
             "(docs/domain_model.md, section 5; src/signal_model.py). The prediction is not fitted to the paraphrased recall."),
           P("3. Results", H)]
    rows = [["Detector", "Training", "Version", "Recall, % [95% CI]", "FPR, %", "F1", "Guardrail"]]
    for _, r in summ.iterrows():
        degenerate = (r.recall_ci_hi - r.recall_ci_lo) < 1e-9 and "recall_wilson_lo" in summ
        ci = (f"[{r.recall_wilson_lo*100:.1f}, {r.recall_wilson_hi*100:.1f}]" if degenerate else f"[{r.recall_ci_lo*100:.1f}, {r.recall_ci_hi*100:.1f}]")
        rows.append([nm(r.model), "augmented" if r.condition == "augmented" else "plain", r.version, f"{r.recall_mean*100:.1f} {ci}",
                     f"{r.fpr_mean*100:.2f}", f"{r.f1_mean:.3f}", "ok" if r.fpr_guardrail_ok else "FAIL"])
    el += [tab(rows, [2.8, 2.0, 2.4, 4.2, 1.6, 1.5, 1.8]), P("Table 1. Metrics at the fixed threshold (FPR target 1%), mean over seeds." + (" Seeds give identical numbers (deterministic detector, section 6), so the interval is the 95% Wilson interval over the test e-mails." if (summ.recall_sd.fillna(0) < 1e-9).all() else ""), C), Spacer(1, 6),
           img("fig1_recall_original_vs_paraphrased", 15), P("Figure 1. Recall at FPR = 1%, original vs paraphrased; error bars: 95% CI (bootstrap over seeds; Wilson over e-mails if all seeds are identical).", C)]
    if rq1 is not None:
        rows = [["Detector", "Pairs", "Recall orig", "Recall para", "Change, p.p. [95% CI]", "Lost / gained", "Upper 95% bound on lost share", "p (Holm)", "H0 rejected"]]
        for _, r in rq1.iterrows():
            ub = beta.ppf(0.95, r.b + 1, r.n_pairs - r.b) if r.b < r.n_pairs else 1.0
            rows.append([nm(r.model), int(r.n_pairs), f"{r.recall_orig:.3f}", f"{r.recall_para:.3f}",
                         f"{(0.0 - r.delta)*100:+.1f} [{(0.0 - r.ci_high)*100:+.1f}, {(0.0 - r.ci_low)*100:+.1f}]", f"{int(r.b)} / {int(r.c)}", f"{ub*100:.1f}%",
                         f"{r.p_holm:.4f}", "yes" if r.reject_H0_RQ1 else "no"])
        el += [Spacer(1, 6), tab(rows, [2.1, 1.1, 1.6, 1.6, 3.0, 1.6, 2.6, 1.5, 1.7]),
               P("Table 2. RQ1: paired exact McNemar test, one-sided, Holm-adjusted across detectors. Lost = detected in original but missed after "
                 "paraphrase; gained = the reverse. The bound is the one-sided Clopper-Pearson 95% upper limit of the share of e-mails that get lost.", C)]
    el += [img("fig2_score_margin_distribution", 13), P("Figure 2. Score margin of every phishing e-mail (box: median and quartiles, dots: e-mails). Values above 0 are detected.", C)]
    cb = check[check.condition == "baseline"] if (check.condition == "baseline").any() else check
    rows = [["Detector", "Recall orig (meas. / model)", "Recall para (meas. / model)", "|error|, p.p.", "Score shift [95% CI]", "Spread ratio", "Median margin"]]
    for _, r in cb.iterrows():
        rows.append([nm(r.model), f"{r.recall_orig_measured:.3f} / {r.recall_orig_model:.3f}", f"{r.recall_para_measured:.3f} / {r.recall_para_model:.3f}",
                     f"{r.abs_error_pp:.1f}", f"{r.delta:+.2f} [{r.delta_ci_lo:+.2f}, {r.delta_ci_hi:+.2f}]", f"{r.sigma_ratio_para_orig:.2f}", f"{r.median_margin_orig:.2f}"])
    el += [P("4. Mathematical model vs measurement", H), tab(rows, [2.2, 3.1, 3.1, 1.5, 3.2, 1.4, 1.6]),
           P("Table 3. Location-shift model (not fitted to the paraphrased recall). Spread ratio = SD of paraphrased scores / SD of original scores; "
             "values far from 1 or a large error mean that the pure-shift or normality assumption does not hold for that detector. Median margin = median of "
           "(score - threshold) of original phishing: how far the typical e-mail sits above the decision threshold.", C),
           img("fig3_model_check", 12), P("Figure 3. Measured vs predicted recall of paraphrased e-mails.", C),
           P("5. Discussion", H)]
    worst = cb.loc[cb.abs_error_pp.idxmax()]
    for _, r in cb.iterrows():
        lost = (rq1[rq1.model == r.model].b.iloc[0] if rq1 is not None else None)
        el += [P(f"<b>{nm(r.model)}.</b> Paraphrasing moves the score by {r.delta:+.2f} (95% CI [{r.delta_ci_lo:+.2f}, {r.delta_ci_hi:+.2f}]), "
                 f"while the median original e-mail sits {r.median_margin_orig:.2f} above the threshold; the model says a shift of {r.breakeven_shift_2pp:+.2f} "
                 f"would be needed to cost 2 p.p. of recall. " + (f"{int(lost)} e-mails were lost, so no drop of recall is visible. A score shift without a recall drop "
                 "means the detector has a large safety margin on this data (ceiling effect), not that paraphrasing is harmless in general." if lost == 0 else ""))]
    el += [P(f"The largest gap between model and measurement is {worst.abs_error_pp:.1f} p.p. ({nm(worst.model)}). "
             "Where it is small, a paraphrase acts mainly as a shift of the score; where it is large, paraphrasing also changes the shape of the score "
             "distribution and the simple model should not be used for that detector."),
           P("<b>Limitations.</b> Phishing and legitimate e-mail come from different sources and periods (source/label confound); only paraphrases accepted by the "
             "quality filter enter the pairs; meaning drift of paraphrases is not manually audited; paired tests use the first seed only. "
             "Run-specific limits (paraphraser, detectors, seeds, data) are listed in section 6."),
           ]
    notes = ROOT / "docs" / "run_notes.md"
    if notes.exists():
        el.append(P("6. Notes on this run (data, deviations from the plan)", H))
        for para in notes.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                el.append(P(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", para.replace("&", "&amp;").replace("<", "&lt;"))))
    el += [P("7. Domain model (STRIDE)", H),
           P("STRIDE matrix of the mail gateway, attack tree for the goal 'phishing reaches the inbox' and metric collection points MP-1..MP-4 are in "
             "docs/domain_model.md. Threat T2b (evasion by adversarial rewriting) is the one tested experimentally."),
           P("8. Repository and reproduction", H),
           P("docs/task3_runbook.md lists the exact commands: download data, prepare, paraphrase, train and evaluate with 5 seeds (plain and augmented), "
             "statistics, processing, figures, this report. Environment: exact pins in requirements.txt, Docker image python:3.11.9-slim.")]
    out = ROOT / "reports" / "TASK3_report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="TASK 3 report").build(el)
    print("wrote", out)


def _table(rows, widths_cm):
    hs = ParagraphStyle("th", fontSize=7.5, leading=9, alignment=1)
    rows = [[Paragraph(str(c), hs) for c in rows[0]]] + rows[1:]
    t = Table(rows, repeatRows=1, colWidths=[w * cm for w in widths_cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="[fill in]")
    ap.add_argument("--repo", default="https://github.com/Dias-M/ResearchMetod-MoldassanovDias-BilalSamat")
    a = ap.parse_args()
    main(a.group, a.repo)
