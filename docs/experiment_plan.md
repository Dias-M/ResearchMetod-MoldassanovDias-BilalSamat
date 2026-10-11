# Experiment Plan

Status: **draft for supervisor approval** (Task 2, weeks 4-6).

| | |
|---|---|
| Student |Dias Moldassanov, Bilal Samat |
| Supervisor / reviewer | Aisultan Shoiynbek |
| Approved on | 11.10.2026 |

## 1. Goal and hypotheses

Goal: quantify the robustness of text-based phishing detectors to LLM paraphrasing and test whether
paraphrase-augmented training removes the loss. Hypotheses are inherited from the Task 1 methodology card.

| RQ | H0 | H1 | Test |
|----|----|----|------|
| RQ1 | delta recall = 0 for each detector | delta recall > 0 | paired bootstrap CI + exact McNemar (one-sided), Holm-adjusted over detectors |
| RQ2 | beta(paraphrased) = 0 | beta < 0 | logistic regression, cluster-robust SE by e-mail |
| RQ3 | after augmentation recall < baseline - 2 p.p. or FPR rises by more than 1 p.p. | recall >= baseline - 2 p.p. and FPR rise <= 1 p.p. | descriptive with paired bootstrap CI; equivalence-style margin |

Variables. Independent: text version (original / paraphrased), detector type, training regime (plain / augmented).
Dependent: recall at FPR = 1%, F1, ROC-AUC. Controlled: dataset/source, e-mail length, split, seeds, threshold rule, paraphraser model and prompt.

## 2. Data

Positive class (phishing) comes from phishing corpora only. Negative class is legitimate mail only.
Spam is excluded because spam is not phishing and would add label noise (it can be added later as a sensitivity analysis).
Sources are listed in `data/README.md`.

Cleaning (`phishrobust.data.prepare`): strip HTML, normalise whitespace, drop texts shorter than 30 characters,
truncate to 4000 characters, remove exact duplicates and any text that appears with both labels.
Split (`assign_splits`): 70/15/15, stratified by label x source, one fixed split seed. Every unique text is in exactly one split.

## 3. Evaluation protocol

1. Train each detector on the **original** training set (plain regime).
2. Score the validation set. Choose the threshold `t` as the score that leaves at most 1% of **validation ham** above it.
3. Build two test sets that differ only in the phishing text: `original` = test ham + test phishing; `paraphrased` = the same test ham + the paraphrased versions of the same phishing e-mails.
4. Apply the **same** `t` to both sets. Ham is never changed, so FPR stays comparable and any change in recall is caused by the phishing text.
5. Use only matched pairs (phishing e-mails whose paraphrase passed the quality filter).
6. Repeat for seeds 42-46 (affects DistilBERT initialisation and batch order; TF-IDF+LR is deterministic).
7. RQ3: add paraphrased **training** phishing to the training set, repeat steps 1-5.

## 4. Detectors

| Name | Type | Training |
|------|------|----------|
| `tfidf_lr` | TF-IDF (1-2 grams) + logistic regression | scikit-learn |
| `distilbert` | `distilbert-base-uncased`, fine-tuned 2 epochs | PyTorch loop |
| `zero_shot_llm` | `Qwen/Qwen2.5-1.5B-Instruct`, score = logit(Yes) - logit(No) | none |

## 5. Paraphrasing

Open-weights LLM run locally (`Qwen/Qwen2.5-1.5B-Instruct`, temperature 0.7, top-p 0.9). The prompt asks for a rewrite with the same meaning,
tone and structure and with URLs, addresses and numbers unchanged. A paraphrase is accepted only if it differs from the original,
has 0.5-1.5x the original length and keeps every URL. The acceptance rate is reported.
For RQ3 generalisation, a second, different paraphraser model is used for the test paraphrases (planned for W5; model to be chosen).

## 6. Statistics

- RQ1: delta = recall(original) - recall(paraphrased) per detector, 95% paired bootstrap CI (10,000 resamples of e-mails), exact McNemar (b = detected before and missed after, c = opposite), one-sided, Holm-adjusted over detectors. Seed-to-seed standard deviation of delta is reported next to it.
- RQ2: `detected ~ paraphrased + C(model) + C(source) + log(length)`, standard errors clustered by e-mail; report odds ratio and one-sided p.
- RQ3: recall and FPR of augmented detectors on the paraphrased test set vs. the plain detectors on the original test set.
- alpha = 0.05. Everything above is in `scripts/05_stats.py` and `src/phishrobust/stats.py`.

## 7. Reproducibility

All parameters live in `config/config.yaml`. Seeds are fixed. Model names are pinned in the config.
Environment: `Dockerfile` + `requirements.txt` (minimum versions; pin exact versions with `pip freeze` after the first successful run).
Raw data are not redistributed; scripts rebuild them. `make test` and `make smoke` verify the setup.

## 8. Threats to validity

- **Source/label confound.** If phishing comes from one corpus and ham from others, a detector can learn corpus style instead of phishing. Mitigation: add a second, modern phishing corpus; report results per source; the paired original-vs-paraphrased design still isolates the paraphrasing effect on the same e-mails.
- **Old data.** Nazario (2000s-2010s), Enron (2001) and SpamAssassin (2003) differ from current mail. Results describe robustness on these corpora, not on today's traffic.
- **Weak paraphraser.** A 1.5B model may produce poorer text than current attackers' tools. Results are a lower bound on the effect.
- **Meaning drift.** Paraphrases may change intent. Mitigation: URL/length filter now; manual check of a random sample of 50 in W5.
- **Small zero-shot model.** Zero-shot scores depend on the prompt. The prompt is fixed in code and reported.
- **Multiple tests.** Holm correction over detectors. RQ2 and RQ3 are secondary.

## 9. Compute budget

TF-IDF+LR: seconds on CPU. DistilBERT and Qwen-1.5B: a free Colab/Kaggle GPU is enough; on CPU use `--max-test` to subsample the test set.
Paraphrasing is the slowest step; it checkpoints after every batch and resumes.

## 10. Timeline

| Week | Work | Output |
|------|------|--------|
| W4 | Environment, data download, cleaning, split, choose second phishing corpus | `data/processed/emails.csv`, data statistics table |
| W5 | Paraphrasing (val, test), manual quality check of 50 samples, baseline detectors | `paraphrases.csv`, baseline metrics |
| W6 | Freeze protocol, dry run of all scripts, plan approval | tagged `v0.2-task2` release |

Task 3 (weeks 7-9) executes steps 3-7 of the protocol and the statistics.

## 11. Task 2 checklist

- [x] Repository with README, Docker/Jupyter environment, config
- [x] Data scripts (download, clean, split) and a synthetic sample
- [x] Pipeline scripts 01-05 and unit tests
- [x] Experiment plan (this file)
- [ ] Real data downloaded and described
- [ ] Docker image built once; exact versions pinned
- [ ] Supervisor approval recorded above
