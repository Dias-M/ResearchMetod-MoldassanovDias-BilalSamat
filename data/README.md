# Data

Nothing in `data/raw/` or `data/processed/` is committed. Rebuild it with the scripts.

## Schema

`data/processed/emails.csv`: `id, text, label, source, split`
`label`: 1 = phishing, 0 = legitimate (ham). `split`: train / val / test (stratified by label x source).
Paraphrases: `data/processed/paraphrases.csv`: `id, text_para, ok, backend`.

## Sources

| Source | Class | How to get it | Notes |
|--------|-------|---------------|-------|
| Nazario phishing corpus | phishing | Manual download of the `.mbox` files from the author's page (monkey.org/~jose/phishing). Put them into `data/raw/`. | Availability and licence must be checked before use. |
| Additional phishing corpus (modern) | phishing | To be chosen in W4 (candidates: public Kaggle/HF phishing e-mail sets). Put a CSV with `text,label,source` into `data/raw/`. | Needed to reduce the source/label confound (see plan, section 8). |
| Enron-Spam (ham only) | ham | `python scripts/01_download_data.py --only enron` (Hugging Face `SetFit/enron_spam`) | Spam rows are dropped: spam is not phishing. |
| SpamAssassin public corpus (ham only) | ham | `python scripts/01_download_data.py --only spamassassin` | Easy-ham archive from 2003. |
| CSDMC2010 | ham | Optional, manual. | Used in related work [10]. |

URLs in the download script were not checked against the live servers. Check each one before the first run.

## Sample data

`data/sample/emails.csv` has 10 synthetic rows (5 phishing, 5 ham; `.invalid` domains) and a fixed `split` column (6 train, 2 val, 2 test). It exists only to check that the pipeline runs: `python src/benchmark.py --sample`.
