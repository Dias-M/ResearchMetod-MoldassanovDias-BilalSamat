# Methodology Passport Card

## 1. Project information

| | |
|---|---|
| Title | Robustness of Text-Based Phishing Detectors to LLM Paraphrasing |
| Authors | Dias Moldassanov, Bilal Samat |
| Repository | https://github.com/Dias-M/ResearchMetod-MoldassanovDias-BilalSamat |

## 2. Research type

**Quantitative.** The study is a controlled computational experiment with a paired design: the same phishing
e-mails are scored by the same detector in their original and paraphrased form, and the outcome is a numeric
metric (recall at a fixed false-positive rate). The question is "how much", so it is answered with
effect sizes, confidence intervals and significance tests rather than with interviews or text analysis.

## 3. Hypotheses

Main hypothesis (RQ1, tested per detector: TF-IDF + LR, DistilBERT, zero-shot LLM):

- **H0:** LLM paraphrasing does not change recall at FPR = 1%: recall(original) − recall(paraphrased) = 0.
- **H1:** Paraphrasing lowers recall: recall(original) − recall(paraphrased) > 0.

Causal hypothesis (RQ2, controlling for detector, dataset and e-mail length):

- **H0:** β(paraphrased) = 0 in a logistic regression of the detection outcome.
- **H1:** β(paraphrased) < 0.

Optimisation hypothesis (RQ3):

- **H0:** After training on paraphrased e-mails, recall stays more than 2 percentage points (p.p.) below the baseline, or FPR rises by more than 1 p.p.
- **H1:** Recall is within 2 p.p. of the baseline and FPR rises by at most 1 p.p.

## 4. Variable matrix

| Type | Variable | Values / unit |
|------|----------|---------------|
| Independent | Text version | original / LLM-paraphrased |
| Independent | Detector | TF-IDF + logistic regression, DistilBERT, zero-shot LLM |
| Independent | Training regime (RQ3) | plain / augmented with paraphrases |
| Dependent | Recall at FPR = 1% (primary) | share of phishing e-mails detected, 0–1 |
| Dependent | F1, precision, ROC-AUC, realised FPR | 0–1 |
| Controlled | Datasets and train/validation/test split | fixed split, one seed for the split |
| Controlled | Decision threshold rule | set on validation ham of the original data, reused for all conditions |
| Controlled | Paraphraser | one open-weights LLM, fixed prompt, temperature and seed (`config/config.yaml`) |
| Controlled | Random seeds | 42–46 for model initialisation and batch order |
| Controlled | Software | Python 3.11.9, Docker image `python:3.11.9-slim`, exact library versions in `requirements.txt` |
| Controlled | Hardware | CPU/GPU model recorded in `results/benchmark.log` for every run |
| Covariate | E-mail length, dataset source | characters; categorical |

## 5. Metrics and baseline

**Primary metric.** Recall at FPR = 1%. A mail filter must keep false alarms rare, so detectors are compared at a
fixed, low false-positive rate. The threshold is chosen on validation ham of the original data and reused for the
paraphrased data, so only the phishing text differs between conditions.

**Secondary metrics.** F1, precision, ROC-AUC.

**Guardrail metrics** (a run is valid only if all hold; thresholds are proposed and may be adjusted with the supervisor):

| Guardrail | Limit |
|-----------|-------|
| Realised FPR on test ham | ≤ 2% (target 1%) |
| Paraphrase acceptance by the quality filter | ≥ 70% of phishing e-mails; 100% of URLs preserved |
| Seed-to-seed standard deviation of recall (DistilBERT) | ≤ 2 p.p. |
| Pipeline integrity | `python src/benchmark.py` finishes with `BENCHMARK OK`, no errors |

**Baseline.** Each detector's recall on the original (unparaphrased) test e-mails is the reference for its own
drop. TF-IDF + logistic regression trained on original data is the classical baseline model (B0 in the README).
RQ3 compares the augmented detectors with the plain ones on the same paraphrased test set.
