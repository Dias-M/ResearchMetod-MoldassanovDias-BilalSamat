import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import signal_model as sm  # noqa: E402


class TestSignalModel(unittest.TestCase):
    def test_recall_formula(self):
        self.assertAlmostEqual(sm.recall_normal(0.0, 1.0, 0.0), 0.5)
        self.assertAlmostEqual(sm.recall_normal(1.6449, 1.0, 0.0), 0.95, places=3)

    def test_shift_lowers_recall(self):
        self.assertLess(sm.predicted_recall_after_shift(1.0, 1.0, 0.0, -0.5), sm.recall_normal(1.0, 1.0, 0.0))

    def test_breakeven_costs_exactly_d_points(self):
        mu, sd, t = 1.0, 1.0, 0.2
        d = sm.breakeven_shift(mu, sd, t, 2.0)
        self.assertLess(d, 0)
        self.assertAlmostEqual(sm.recall_normal(mu, sd, t) - sm.predicted_recall_after_shift(mu, sd, t, d), 0.02, places=6)

    def test_check_cells_recovers_pure_shift(self):
        rng = np.random.default_rng(1)
        o = rng.normal(1.0, 1.0, 4000)
        p = o - 0.6
        rows = []
        for ver, s in (("original", o), ("paraphrased", p)):
            rows += [dict(seed=1, model="m", condition="baseline", version=ver, email_id=i, score=x, threshold=0.0) for i, x in enumerate(s)]
        cells = sm.check_cells(pd.DataFrame(rows))
        self.assertEqual(len(cells), 1)
        self.assertLess(float(cells.abs_error_pp.iloc[0]), 1.0)   # model matches data that obey its assumptions
        self.assertAlmostEqual(float(cells.sigma_ratio_para_orig.iloc[0]), 1.0, places=6)


class TestCorpusBuilder(unittest.TestCase):
    def setUp(self):
        import importlib.util
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("build_corpus", ROOT / "scripts" / "00_build_corpus.py")
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_neutralise_removes_label_giveaways(self):
        out = self.m.neutralise("Dear phishing@pot-fp@hotmail.com, since 2023 and 2008 at Phishing Pot")
        self.assertNotIn("pot", out.lower().replace("example", ""))
        self.assertNotIn("2023", out)
        self.assertNotIn("2008", out)

    def test_english_share(self):
        en = "Please click the link to verify your account and you will be able to use all of our services for the rest of the year."
        self.assertGreater(self.m.english_share(en), 0.12)
        self.assertLess(self.m.english_share("Bitte klicken Sie auf den Link um Ihr Konto zu bestätigen und alle unsere Dienste zu nutzen heute"), 0.12)

    def test_signature_ignores_digits_and_case(self):
        self.assertEqual(self.m.signature("Order 123 Now"), self.m.signature("order 999 now"))


if __name__ == "__main__":
    unittest.main()
