import csv
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestSampleAndBenchmark(unittest.TestCase):
    def test_sample_is_micro_file(self):
        rows = list(csv.DictReader(open(ROOT / "data" / "sample" / "emails.csv", encoding="utf-8")))
        self.assertTrue(5 <= len(rows) <= 10)
        self.assertEqual({r["split"] for r in rows}, {"train", "val", "test"})
        for split in ("train", "val", "test"):
            self.assertEqual({r["label"] for r in rows if r["split"] == split}, {"0", "1"})

    def test_benchmark_sample_runs(self):
        proc = subprocess.run([sys.executable, "src/benchmark.py", "--sample"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-500:] + proc.stderr[-500:])
        self.assertIn("BENCHMARK OK", (ROOT / "results" / "benchmark.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
