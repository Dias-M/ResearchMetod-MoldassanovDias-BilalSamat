"""Raw corpora -> data/processed/emails.csv (clean, de-duplicated, split)."""
import argparse

import pandas as pd

from _common import ROOT, load_config, p
from phishrobust.data import assign_splits, prepare, read_email_dir, read_mbox, text_hash

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use data/sample/emails.csv (synthetic)")
    args = ap.parse_args()
    cfg = load_config()
    raw, out = p(cfg, "raw"), p(cfg, "processed")

    parts = []
    if args.sample:
        parts.append(pd.read_csv(ROOT / "data" / "sample" / "emails.csv"))
    else:
        for f in sorted(raw.glob("*.csv")):                      # schema: text,label,source (label 1=phishing)
            parts.append(pd.read_csv(f))
        for f in sorted(raw.rglob("*.mbox")) + sorted(raw.rglob("phishing*.mbox")):
            parts.append(read_mbox(f, label=1, source="nazario"))
        if (raw / "spamassassin_ham").exists():
            parts.append(read_email_dir(raw / "spamassassin_ham", label=0, source="spamassassin"))
    if not parts:
        raise SystemExit("No raw data found. Run scripts/01_download_data.py and see data/README.md.")

    raw_df = pd.concat(parts, ignore_index=True)
    df = prepare(raw_df, **cfg["cleaning"])
    if "split" in raw_df.columns and raw_df["split"].notna().all():   # sample file ships with fixed splits
        df = df.merge(raw_df[["text", "split"]].assign(hash=lambda d: d["text"].map(text_hash))[["hash", "split"]], on="hash")
    else:
        df = assign_splits(df, cfg["split"]["train"], cfg["split"]["val"], seed=cfg["seeds"][0])
    df.drop(columns=["hash"]).to_csv(out / "emails.csv", index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
