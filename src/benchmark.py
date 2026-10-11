"""Benchmark runner: executes the whole pipeline and writes results/benchmark.log.

    python src/benchmark.py --sample   # offline, uses data/sample (10 synthetic rows), mock paraphraser
    python src/benchmark.py --full     # real experiment (needs data in data/raw, see data/README.md)

The log records the environment (Python, OS, library versions) next to every step so that
the controlled variables of each run are documented.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIBS = ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "torch", "transformers"]

SAMPLE_STEPS = [
    ["scripts/02_prepare_data.py", "--sample"],
    ["scripts/03_paraphrase.py", "--backend", "mock", "--splits", "val", "test", "train"],
    ["scripts/04_train_eval.py", "--models", "tfidf_lr", "--seeds", "42"],
    ["scripts/04_train_eval.py", "--models", "tfidf_lr", "--seeds", "42", "--augment"],
    ["scripts/05_stats.py"],
]
FULL_STEPS = [
    ["scripts/02_prepare_data.py"],
    ["scripts/03_paraphrase.py", "--backend", "hf", "--splits", "val", "test"],
    ["scripts/04_train_eval.py", "--models", "tfidf_lr", "distilbert", "zero_shot_llm"],
    ["scripts/03_paraphrase.py", "--backend", "hf", "--splits", "train"],
    ["scripts/04_train_eval.py", "--models", "tfidf_lr", "distilbert", "--augment"],
    ["scripts/05_stats.py"],
]


def version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--sample", action="store_true", help="offline run on data/sample (default)")
    mode.add_argument("--full", action="store_true", help="real experiment")
    args = ap.parse_args()
    steps = FULL_STEPS if args.full else SAMPLE_STEPS

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))["benchmark"]
    log_path = ROOT / cfg["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log:
        def emit(msg: str = "") -> None:
            print(msg)
            log.write(msg + "\n")
            log.flush()

        emit(f"benchmark mode: {'full' if args.full else 'sample'}")
        emit(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        emit(f"python: {platform.python_version()}  os: {platform.platform()}  cpu: {platform.processor() or platform.machine()}")
        emit("libraries: " + ", ".join(f"{p}=={version(p)}" for p in LIBS))
        t_all = time.time()
        for step in steps:
            emit(f"\n$ python {' '.join(step)}")
            t0 = time.time()
            proc = subprocess.run([sys.executable, *step], cwd=ROOT, capture_output=True, text=True, timeout=cfg["step_timeout_s"])
            emit(proc.stdout.rstrip())
            if proc.returncode != 0:
                emit(proc.stderr.rstrip())
                emit(f"\nBENCHMARK FAILED at step: {' '.join(step)} (exit code {proc.returncode})")
                return proc.returncode
            emit(f"[step finished in {time.time() - t0:.1f}s]")
        emit(f"\ntotal time: {time.time() - t_all:.1f}s")
        emit("results: results/summary.md, results/summary.csv, results/metrics*.csv")
        emit("BENCHMARK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
