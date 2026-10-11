# Robustness of Text-Based Phishing Detectors to LLM Paraphrasing

## Authors

Dias Moldassanov, Bilal Samat. Narxoz University, Research Methods course (Task 2: Methodology & Repo Setup).
Supervisor: Aisultan Shoiynbek.

## Overview

LLMs can rewrite phishing e-mails in fresh wording. This project measures how much such paraphrasing lowers
the recall of text-based phishing detectors at a fixed false-positive rate, and whether training on paraphrased
phishing recovers the loss. The full methodology is in [`docs/methodology_card.md`](docs/methodology_card.md) and
[`docs/experiment_plan.md`](docs/experiment_plan.md).

**Research question.** How much does LLM paraphrasing of phishing e-mails reduce the recall of TF-IDF + logistic
regression, DistilBERT and a zero-shot LLM at FPR = 1%, and does paraphrase-augmented training restore it?

## Repository structure

```
.
├── config/config.yaml        experiment parameters, seeds, paths, timeouts
├── data/
│   ├── README.md             data sources and schema
│   └── sample/emails.csv     10 synthetic rows for the smoke run
├── docs/
│   ├── methodology_card.md   Methodology Passport Card
│   └── experiment_plan.md    detailed experiment plan
├── notebooks/                smoke-test notebook
├── scripts/                  pipeline steps 01..05
├── src/
│   ├── benchmark.py          benchmark runner (writes results/benchmark.log)
│   └── phishrobust/          data, models, metrics, paraphrase, stats
├── tests/                    unit tests
├── Dockerfile  docker-compose.yml  requirements.txt  Makefile
└── LICENSE
```

## Planned benchmarks

| ID | Detector | Trained on | Evaluated on | Purpose |
|----|----------|------------|--------------|---------|
| B0 | TF-IDF + LR | original | original phishing | baseline |
| B1 | TF-IDF + LR | original | paraphrased phishing | RQ1, RQ2 |
| B2 | DistilBERT | original | original phishing | baseline |
| B3 | DistilBERT | original | paraphrased phishing | RQ1, RQ2 |
| B4 | Zero-shot LLM | none | original phishing | baseline |
| B5 | Zero-shot LLM | none | paraphrased phishing | RQ1, RQ2 |
| B6 | TF-IDF + LR | original + paraphrased | paraphrased phishing | RQ3 |
| B7 | DistilBERT | original + paraphrased | paraphrased phishing | RQ3 |

Primary metric for every row: recall at FPR = 1% (threshold fixed on validation ham).

## System requirements

- Docker 24 or newer (or Python 3.11.9 with `pip install -r requirements.txt`)
- 8 GB RAM and 10 GB free disk for the full experiment (the sample run needs far less)
- GPU optional; DistilBERT and the LLM are slow on CPU
- Internet access for the full experiment (datasets and model weights)

## Quickstart

Runs the whole pipeline on the sample data and writes `results/benchmark.log`:

```bash
git clone https://github.com/Dias-M/ResearchMetod-MoldassanovDias-BilalSamat.git && cd ResearchMetod-MoldassanovDias-BilalSamat
docker build -t phish-robust .
docker run --rm -v "$(pwd)/results:/workspace/results" phish-robust
```

A successful run ends with `BENCHMARK OK`. Numbers from the sample (2 test e-mails) are meaningless; the run only
proves that the pipeline works. For the real experiment, get the data first (see `data/README.md`), then run
`python src/benchmark.py --full`. JupyterLab: `docker compose up`, then open http://localhost:8888.

## Reproducibility

Direct dependencies are pinned to exact versions in `requirements.txt` and the base image is pinned
(`python:3.11.9-slim`). Seeds and all parameters live in `config/config.yaml`. Every benchmark run logs the Python,
OS and library versions.

## Citation

```bibtex
@misc{moldassanov_samat_2026_phishing_paraphrase,
  author = {Moldassanov, Dias and Samat, Bilal},
  title  = {Robustness of Text-Based Phishing Detectors to LLM Paraphrasing},
  year   = {2026},
  url    = {https://github.com/Dias-M/ResearchMetod-MoldassanovDias-BilalSamat}
}
```

## License

MIT, see [`LICENSE`](LICENSE). Datasets keep their own licences.
