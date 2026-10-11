import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishrobust.data import assign_splits, prepare  # noqa: E402
from phishrobust.metrics import evaluate, threshold_at_fpr  # noqa: E402
from phishrobust.paraphrase import quality_ok  # noqa: E402
from phishrobust.stats import holm, mcnemar_exact, paired_bootstrap_delta_recall  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_threshold_respects_fpr(self):
        ham = np.arange(100, dtype=float)               # scores 0..99
        thr = threshold_at_fpr(ham, 0.01)
        self.assertLessEqual((ham > thr).mean(), 0.01)
        self.assertEqual(thr, 98.0)

    def test_evaluate_recall(self):
        y = [0, 0, 1, 1]
        m = evaluate(y, [0.1, 0.2, 0.9, 0.3], threshold=0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.0)


class TestData(unittest.TestCase):
    def setUp(self):
        n = 200
        self.df = pd.DataFrame({
            "text": [f"unique e-mail text number {i} with enough characters" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "source": ["a" if i % 4 < 2 else "b" for i in range(n)],
        })

    def test_dedupe_and_conflicts(self):
        d = pd.concat([self.df, self.df.head(5), pd.DataFrame({"text": [self.df.text[0]], "label": [1], "source": ["a"]})])
        out = prepare(d)
        self.assertEqual(out.hash.is_unique, True)
        self.assertNotIn(self.df.text[0], out.text.values)      # label conflict removed

    def test_splits_disjoint_and_complete(self):
        out = assign_splits(prepare(self.df), 0.7, 0.15, seed=1)
        self.assertEqual(set(out.split), {"train", "val", "test"})
        self.assertEqual(len(out), out.id.nunique())
        self.assertAlmostEqual((out.split == "train").mean(), 0.7, delta=0.03)


class TestParaphraseQuality(unittest.TestCase):
    def test_rules(self):
        o = "Please verify at http://x.invalid/a now to keep your account active"
        self.assertTrue(quality_ok(o, "Kindly confirm at http://x.invalid/a today to keep your profile active"))
        self.assertFalse(quality_ok(o, o))
        self.assertFalse(quality_ok(o, "Kindly confirm today to keep your profile active"))   # URL lost
        self.assertFalse(quality_ok(o, "ok"))                                                  # too short


class TestStats(unittest.TestCase):
    def test_mcnemar_exact(self):
        r = mcnemar_exact([1] * 10, [0] * 10)
        self.assertEqual((r["b"], r["c"]), (10, 0))
        self.assertAlmostEqual(r["p_two_sided"], 2 * 0.5**10)
        self.assertAlmostEqual(r["p_one_sided"], 0.5**10)

    def test_bootstrap_ci_contains_delta(self):
        rng = np.random.default_rng(0)
        o = rng.random(300) < 0.9
        p = rng.random(300) < 0.7
        r = paired_bootstrap_delta_recall(o, p, n_boot=2000, seed=1)
        self.assertLess(r["ci_low"], r["delta"])
        self.assertGreater(r["ci_high"], r["delta"])

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
