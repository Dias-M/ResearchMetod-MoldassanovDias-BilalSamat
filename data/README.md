# Data

`data/raw/` and the large `data/processed/emails.csv` are not committed; rebuild them with the scripts (order in `docs/task3_runbook.md`). The small result tables of the reported run are committed: `metrics_summary.csv`, `paired_delta.csv`, `model_check.csv`, `model_check_cells.csv`, `predictions_flagged.csv`, `paraphrases.csv`.

## Schema

`data/processed/emails.csv`: `id, text, label, source, split`
`label`: 1 = phishing, 0 = legitimate (ham). `split`: train / val / test (stratified by label x source).
Paraphrases: `data/processed/paraphrases.csv`: `id, text_para, ok, backend`.

## Sources (used in the reported run)

| Source | Class | How to get it | Notes |
|--------|-------|---------------|-------|
| Phishing Pot (github.com/rf-peixoto/phishing_pot, CC BY-NC 4.0, commit 80a4867) | phishing | `git clone` into a folder, then `python scripts/00_build_corpus.py --phish-dir <folder>/email --ceas <CEAS_08.csv>` | 12,865 real honeypot e-mails; English only, near-duplicates collapsed (1,639 kept). |
| CEAS-08 (github.com/PHP4Life/phishing-email-dataset, `CEAS_08.csv`) | ham | Download the CSV | Only rows with label 0 are used; spam rows are dropped. |

Nazario, Enron-Spam and SpamAssassin from the first plan were not used in this run. `scripts/01_download_data.py` is the old downloader and is untested.

## Sample data

`data/sample/emails.csv` has 10 synthetic rows (5 phishing, 5 ham; `.invalid` domains) and a fixed `split` column (6 train, 2 val, 2 test). It exists only to check that the pipeline runs: `python src/benchmark.py --sample`.
