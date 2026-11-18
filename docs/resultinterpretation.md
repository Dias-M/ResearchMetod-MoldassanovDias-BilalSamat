# Lab 10: Results Interpretation, Limitations and Draft Completion

Project: Robustness of Text-Based Phishing Detectors to LLM Paraphrasing
Authors: Dias Moldassanov, Bilal Samat. Supervisor: Aisultan Shoiynbek.

All numbers below come from `results/summary.md`, `data/processed/metrics_summary.csv`,
`data/processed/model_check.csv` and `docs/run_notes.md`. Items marked **TODO** need the authors' input.

---

## 0. Results vs Discussion (rule used in this paper)

| Section | Question | Allowed content | Not allowed |
|---|---|---|---|
| Results | What happened? | numbers, intervals, tests | "because", "suggests", opinions |
| Discussion | Why, and what does it mean? | explanations, comparison with baselines, implications | new numbers not shown in Results |

Hedging rules: write "results suggest", "within the tested envelope", "in this corpus"; never "proves" or "shows that paraphrasing is harmless".

---

## 1. Discussion

**1.1 Main finding.** At a threshold fixed at FPR = 1% on validation ham, TF-IDF + logistic regression detected 95 of 97 phishing e-mails both in original and in paraphrased form (recall 0.979 vs 0.979). No e-mail was lost and none was gained (b = c = 0; one-sided exact McNemar p = 1.0 after Holm correction), so H0 of RQ1 is not rejected. The one-sided 95% upper bound on the share of e-mails lost to paraphrasing is 3.0%. Within the tested envelope, the data therefore give no evidence that LLM paraphrasing lowers recall, and they bound the possible loss at about 3% of e-mails.

**1.2 Score shift without recall loss.** Paraphrasing did lower the detector score by 0.53 on average (95% CI [-0.67, -0.39]), so the paraphraser changes what the detector sees. The recall stayed unchanged because the median original e-mail lies 4.27 score units above the threshold, while the location-shift model puts the shift that costs 2 percentage points of recall at about -1.10. The observed shift is roughly half of that break-even value. This suggests that the detector had a large safety margin on this data, not that paraphrasing is harmless in general.

**1.3 Model versus measurement.** The location-shift model, which uses only original scores, the threshold and the mean paired shift, predicted a paraphrased recall of 0.992 against the measured 0.979 (absolute error 1.2 p.p.). The ratio of standard deviations (paraphrased / original) is 0.98, which supports the "location change only" assumption. The model overestimates recall slightly; normality of scores is an approximation and would be weaker for saturated scores such as DistilBERT probabilities.

**1.4 Why the margin is so large (likely ceiling effect).** ROC-AUC is about 0.9997, and legitimate e-mail (2008 mailing lists and personal mail) differs strongly in style from phishing (modern marketing and crypto-wallet HTML e-mail). The detector may rely partly on corpus style rather than on phishing semantics. Two label giveaways were removed before splitting (the anonymiser string "phishing@pot" and four-digit years), and the result did not change, but residual style differences remain. A small paraphrasing effect could be hidden by this ceiling.

**1.5 Relation to baselines in the Introduction.** The Introduction motivates the study with reports that LLM-written or LLM-rewritten phishing can evade detectors. Our result for TF-IDF + LR on this corpus does not reproduce a recall drop. A plausible reading is that classical bag-of-words detectors trained on easy, source-separated data keep a wide margin, whereas the evasion effects reported for harder or same-source data may be larger. **TODO:** insert `\cite{...}` keys of the baseline papers from the Introduction and compare their reported recall drops with our 0.0 p.p. [upper bound 3.0%].

**1.6 Status of research questions.**

| RQ | Status | Evidence |
|---|---|---|
| RQ1 (recall drop) | Run for TF-IDF + LR only; H0 not rejected | Table 2; DistilBERT and zero-shot LLM not run |
| RQ2 (logistic regression, cluster-robust SE) | Not run | `statsmodels` was not installed in the run environment |
| RQ3 (paraphrase augmentation) | Not run | needs paraphrases of the training set |

---

## 2. Limitations and Threats to Validity

**Internal validity.**
- The test contains 97 matched pairs, drawn at random (seed 0) from the 111 test phishing e-mails of 160 to 900 characters; three explicit adult e-mails were skipped. Conclusions concern short e-mails. The Wilson 95% interval of recall is wide, [92.8%, 99.4%].
- TF-IDF + LR is deterministic, so the five seeds give identical numbers and the interval over seeds has zero width. Uncertainty is carried by the exact test, the Clopper-Pearson bound and the interval of the score shift instead.
- Realised FPR was 0.49% (11 of 2,231 legitimate test e-mails), below the 1% target and within the guardrail of 2%. With 97 phishing and 2,231 legitimate e-mails, precision (0.896) and F1 (0.936) depend on this class ratio.
- Outliers (25 of 970 rows by the 1.5 x IQR rule) were flagged and kept.

**External validity.**
- One detector (TF-IDF + LR). DistilBERT and the zero-shot LLM were planned but not run, so nothing is claimed for neural detectors.
- One paraphraser. The planned Qwen2.5-1.5B-Instruct was unavailable; Claude produced the 97 paraphrases with the same instruction. Results are specific to this paraphraser and prompt.
- English only; one phishing source (Phishing Pot honeypot, which also contains advertising and scam e-mail) and one legitimate source (CEAS-08, 2008). The corpora differ in era and style from current traffic.

**Construct validity.**
- Recall at FPR = 1% is a single operating point; behaviour at other thresholds is not studied.
- The quality filter checks length ratio (0.5 to 1.5) and URL retention, not preservation of meaning or persuasive intent. **TODO:** state whether the manual check of 50 paraphrases planned for W5 was done; if not, list it as not done.
- Only the message text is scored; headers, sender authentication, attachments and rendered HTML are out of scope.

---

## 3. Conclusion and Future Work

**Paragraph 1: problem and approach.** We asked how much LLM paraphrasing of phishing e-mail lowers the recall of text-based detectors at a false-positive rate of 1%. We used a paired design on 97 test e-mails, a threshold fixed on validation ham, and five seeds, with TF-IDF + logistic regression trained on Phishing Pot and CEAS-08.

**Paragraph 2: RQ1.** Recall was 0.979 for original and for paraphrased e-mails, with no e-mail lost (upper 95% bound 3.0%). Paraphrasing lowered the detector score by 0.53 on average, but the median margin of 4.27 was large enough to absorb it. Within the tested envelope, the evidence does not show a recall loss for this detector.

**Paragraph 3: RQ2, RQ3 and model.** RQ2 and RQ3 were not run in this study, so we make no claim about the logistic-regression effect size or about augmentation. The location-shift model predicted paraphrased recall within 1.2 p.p. of the measured value and suggests that a score shift of roughly twice the observed size would be needed to cost 2 p.p.

**Paragraph 4: scope.** These findings are limited by a small sample of short e-mails, one paraphraser and a corpus pair that a detector can separate easily. They should not be read as evidence that paraphrasing is harmless for detectors trained on harder, same-source data.

**Future work.**
1. Run DistilBERT and the zero-shot LLM (B2 to B5) on a GPU, with five seeds, and report recall at FPR = 1% for each.
2. Rebuild the corpus with modern, same-source legitimate e-mail so that style leakage is reduced, and repeat RQ1 to test whether the ceiling hides an effect.
3. Paraphrase the training set with the planned open-weights model and a second, different paraphraser for testing, then run RQ3 (augmentation) and RQ2 with `statsmodels`.

---

## 4. Cross-references (LaTeX)

Mapping from repository files to labels:

| File | Label | First mention |
|---|---|---|
| `reports/figures/fig1_recall_original_vs_paraphrased` | `fig:recall` | Results, para 1 |
| `reports/figures/fig2_score_margin_distribution` | `fig:margin` | Results, para 2 |
| `reports/figures/fig3_model_check` | `fig:modelcheck` | Model validation |
| Metrics table (recall, FPR, F1) | `tab:metrics` | Results, para 1 |
| RQ1 table (McNemar, Holm) | `tab:rq1` | Results, para 1 |
| Model vs measurement table | `tab:model` | Model validation |
| Recall under normal scores | `eq:recall` | Method |
| Paraphrase as location shift | `eq:shift` | Method |
| Break-even shift | `eq:breakeven` | Method |

Text that mentions each object before it appears:

```latex
Table~\ref{tab:metrics} and Figure~\ref{fig:recall} show that recall at
$\mathrm{FPR}=1\%$ is 0.979 for original and for paraphrased e-mails.
Table~\ref{tab:rq1} reports the paired exact test: no e-mail was lost.
Figure~\ref{fig:margin} shows the score margin of every phishing e-mail,
and Equation~\eqref{eq:recall} gives the modelled recall. Paraphrasing is
modelled as the location shift in Equation~\eqref{eq:shift}; the
break-even shift is Equation~\eqref{eq:breakeven}. Table~\ref{tab:model} and
Figure~\ref{fig:modelcheck} compare the model with the measurement.
```

Figure and equation skeletons:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/figures/fig1_recall_original_vs_paraphrased.pdf}
  \caption{Recall at FPR = 1\%, original vs paraphrased e-mails.}
  \label{fig:recall}
\end{figure}

\begin{equation}
  R(\mu,\sigma,t) = 1-\Phi\!\left(\frac{t-\mu}{\sigma}\right)
  \label{eq:recall}
\end{equation}
\begin{equation}
  R_{\mathrm{para}} = 1-\Phi\!\left(\frac{t-\mu-\delta}{\sigma}\right)
  \label{eq:shift}
\end{equation}
\begin{equation}
  \delta^{*}(d) = (t-\mu) - \sigma\,\Phi^{-1}\!\bigl(1-(R_{\mathrm{orig}}-d)\bigr)
  \label{eq:breakeven}
\end{equation}
```

---

## 5. Citation audit (.bib)

1. Every background claim in the Introduction and Discussion needs `\cite{key}`. Claims to check: LLMs rewrite phishing; detectors lose recall under paraphrase; datasets (Phishing Pot, CEAS-08); TF-IDF + LR, DistilBERT, Qwen2.5; bootstrap, McNemar, Holm, Clopper-Pearson, Wilson.
2. Find unused entries: list keys in the `.bib` and compare with keys used in the `.tex`:
   ```bash
   grep -o '@[a-z]*{[^,]*' refs.bib | sed 's/.*{//' | sort -u > bib_keys.txt
   grep -oh '\\cite[a-z]*{[^}]*}' *.tex | sed 's/.*{//; s/}//' | tr ',' '\n' | sort -u > used_keys.txt
   comm -23 bib_keys.txt used_keys.txt     # entries never cited: remove
   ```
3. Do not invent keys: add only sources the authors actually read. **TODO:** attach the `.bib` file and the `.tex` so this audit can be run on the real files.

---

## 6. Final integrity audit

- [ ] Every figure and table is referenced with `\ref{...}` before it appears.
- [ ] Limitations are explicit, constructive and non-defensive (section 2 above).
- [ ] Abstract and Conclusion give the same final metrics: recall 0.979 / 0.979, 0 of 97 lost, upper bound 3.0%, score shift -0.53 [-0.67, -0.39].
- [ ] Abstract does not mention RQ2, RQ3, DistilBERT or the zero-shot LLM as results.
- [ ] "Group: [fill in]" replaced in `reports/TASK3_report.pdf`.
- [ ] Overleaf log has no "Undefined reference" or "Label(s) may have changed" warnings (compile twice).
