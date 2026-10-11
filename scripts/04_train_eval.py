"""Train detectors on original data, evaluate on original vs paraphrased test phishing.

Protocol: threshold is fixed on VALIDATION ham (original) at target_fpr and reused everywhere.
Only phishing texts differ between the 'original' and 'paraphrased' conditions (matched pairs).
--augment adds paraphrased TRAIN phishing to training (RQ3).
"""
import argparse

import numpy as np
import pandas as pd

from _common import load_config, p
from phishrobust.metrics import evaluate, threshold_at_fpr
from phishrobust.models import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tfidf_lr"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--para-file", default="paraphrases.csv")
    ap.add_argument("--max-test", type=int, default=None, help="subsample test set (slow LLM models)")
    args = ap.parse_args()
    cfg = load_config()
    proc, res = p(cfg, "processed"), p(cfg, "results")

    df = pd.read_csv(proc / "emails.csv")
    para = pd.read_csv(proc / args.para_file)
    para = para[para.ok.astype(bool)].set_index("id")["text_para"]

    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    if args.max_test:
        test = test.sample(min(args.max_test, len(test)), random_state=0)
    test_ham = test[test.label == 0]
    test_ph = test[(test.label == 1) & test.id.isin(para.index)]       # matched pairs only
    test_ph_para = test_ph.assign(text=test_ph.id.map(para))
    if args.augment:
        aug = train[(train.label == 1) & train.id.isin(para.index)]
        train = pd.concat([train, aug.assign(text=aug.id.map(para))], ignore_index=True)
    print(f"train={len(train)} val={len(val)} test_ham={len(test_ham)} matched_test_phishing={len(test_ph)}")

    metrics, preds = [], []
    for seed in args.seeds or cfg["seeds"]:
        for name in args.models:
            det = build(name, cfg, seed).fit(train.text, train.label)
            vs = det.score(val.text)
            thr = threshold_at_fpr(vs[val.label.values == 0], cfg["target_fpr"])
            ham_s = det.score(test_ham.text)
            for version, ph in (("original", test_ph), ("paraphrased", test_ph_para)):
                ph_s = det.score(ph.text)
                y = np.r_[np.zeros(len(ham_s)), np.ones(len(ph_s))]
                m = evaluate(y, np.r_[ham_s, ph_s], thr)
                metrics.append({"seed": seed, "model": name, "augment": int(args.augment), "version": version,
                                "threshold": thr, "n_ham": len(ham_s), "n_phish": len(ph_s), **m})
                preds += [{"seed": seed, "model": name, "augment": int(args.augment), "version": version,
                           "email_id": i, "source": s, "length": len(t), "score": sc, "detected": int(sc > thr)}
                          for i, s, t, sc in zip(ph.id, ph.source, ph.text, ph_s)]
            print(f"seed={seed} {name}: " + " | ".join(
                f"{r['version']} recall={r['recall']:.3f} fpr={r['fpr']:.3f}" for r in metrics[-2:]))

    tag = "_aug" if args.augment else ""
    pd.DataFrame(metrics).to_csv(res / f"metrics{tag}.csv", index=False)
    pd.DataFrame(preds).to_csv(res / f"predictions{tag}.csv", index=False)
