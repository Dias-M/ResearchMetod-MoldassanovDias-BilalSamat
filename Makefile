PY ?= python
export PYTHONPATH := src

.PHONY: build up test smoke full

build:
	docker build -t phish-robust .
up:
	docker compose up
test:
	$(PY) -m unittest discover -s tests -v

# offline benchmark on data/sample (no downloads)
smoke:
	$(PY) src/benchmark.py --sample

# real experiment (needs data, see data/README.md)
full:
	$(PY) src/benchmark.py --full
