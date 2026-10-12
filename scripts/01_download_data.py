"""Fetch public corpora into data/raw/. UNTESTED against live servers: verify each URL first.

Design: positive class (phishing) comes from phishing corpora only. Spam is NOT used as phishing
(spam != phishing); only legitimate mail (ham) is used as the negative class.
Nazario's phishing corpus must be downloaded manually (see data/README.md).
"""
import argparse
import tarfile
import urllib.request

import pandas as pd

from _common import load_config, p

SPAMASSASSIN_HAM = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"


def enron_ham(raw):
    from datasets import load_dataset

    ds = load_dataset("SetFit/enron_spam")
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()])
    df = df[df["label"] == 0][["text"]].assign(label=0, source="enron")
    df.to_csv(raw / "enron_ham.csv", index=False)
    print(f"enron ham: {len(df)} rows -> {raw / 'enron_ham.csv'}")


def spamassassin_ham(raw):
    tgz = raw / "spamassassin_easy_ham.tar.bz2"
    urllib.request.urlretrieve(SPAMASSASSIN_HAM, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(raw / "spamassassin_ham", filter="data")
    print(f"spamassassin ham extracted -> {raw / 'spamassassin_ham'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["enron", "spamassassin"], nargs="*", default=["enron", "spamassassin"])
    args = ap.parse_args()
    raw = p(load_config(), "raw")
    if "enron" in args.only:
        enron_ham(raw)
    if "spamassassin" in args.only:
        spamassassin_ham(raw)
    print("Now place phishing data in data/raw/ (Nazario *.mbox and/or CSVs with text,label,source). See data/README.md.")
