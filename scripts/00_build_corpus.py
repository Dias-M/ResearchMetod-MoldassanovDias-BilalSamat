"""Build data/raw/*.csv for the Task 3 run from two public GitHub corpora (read-only inputs).

  phishing : Phishing Pot, real phishing .eml files (https://github.com/rf-peixoto/phishing_pot, CC BY-NC 4.0)
  ham      : CEAS-08, rows with label 0 (https://github.com/PHP4Life/phishing-email-dataset, file CEAS_08.csv)

Choices (all documented in the report):
  * <style>/<script>/comment blocks are removed before tag stripping (they are not e-mail text).
  * English only (share of common English words >= 12%), because phishing_pot is multilingual and the ham is English;
    otherwise a detector could separate the classes by language alone.
  * Leakage markers are neutralised in BOTH classes: the corpus anonymiser string "phishing@pot" / "phishing pot" (a label giveaway)
    and four-digit years (phishing 2019-2026 vs ham 2007-2008 would let a detector classify by date).
  * Near-duplicate campaigns are collapsed: one e-mail per signature (first 120 letters, digits and punctuation removed),
    so that copies of one template cannot sit in both train and test.
  python scripts/00_build_corpus.py --phish-dir <phishing_pot/email> --ceas <CEAS_08.csv>
"""
import argparse
import email
import re
from email import policy
from pathlib import Path

import pandas as pd

from _common import ROOT
from phishrobust.data import _body

BLOCKS = re.compile(r"<(style|script)\b.*?</\1>|<!--.*?-->", re.I | re.S)
EN = set("the and to of you your a in is for on we our this that with be are it as at by from or have please if not will can".split())


def strip_blocks(t: str) -> str:
    return BLOCKS.sub(" ", str(t))


def english_share(t: str) -> float:
    w = re.findall(r"[a-z']+", t.lower())
    return sum(x in EN for x in w) / len(w) if len(w) >= 15 else 0.0


def neutralise(t: str) -> str:
    t = re.sub(r"phishing@pot[\w.\-+@]*", "user@example.com", str(t), flags=re.I)
    t = re.sub(r"phishing[ _-]?pot", "example", t, flags=re.I)
    return re.sub(r"\b(19|20)\d{2}\b", "YEAR", t)


def signature(t: str) -> str:
    return re.sub(r"[^a-z]", "", t.lower())[:120]


def read_eml_dir(d: Path) -> list[str]:
    out = []
    for p in sorted(d.iterdir()):
        try:
            with open(p, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            out.append(f"{msg.get('subject', '') or ''}\n{strip_blocks(_body(msg))}")
        except Exception:
            continue
    return out


def finish(texts: list[str], label: int, source: str) -> pd.DataFrame:
    df = pd.DataFrame({"text": [strip_blocks(t) for t in texts]})
    n0 = len(df)
    df = df[df.text.map(english_share) >= 0.12]
    n1 = len(df)
    df = df.assign(sig=df.text.map(signature)).drop_duplicates("sig").drop(columns="sig")
    df["text"] = df["text"].map(neutralise)
    print(f"{source}: {n0} read -> {n1} English -> {len(df)} after near-duplicate collapse")
    return df.assign(label=label, source=source)[["text", "label", "source"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phish-dir", required=True)
    ap.add_argument("--ceas", required=True)
    a = ap.parse_args()
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    finish(read_eml_dir(Path(a.phish_dir)), 1, "phishing_pot").to_csv(raw / "phishing_pot.csv", index=False)
    c = pd.read_csv(a.ceas)
    c = c[c.label == 0]
    finish((c.subject.fillna("").astype(str) + "\n" + c.body.astype(str)).tolist(), 0, "ceas08_ham").to_csv(raw / "ceas08_ham.csv", index=False)
