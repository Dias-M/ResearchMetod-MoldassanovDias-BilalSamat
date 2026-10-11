"""Paraphrase PHISHING e-mails of the chosen splits. Ham is never changed. Resumable."""
import argparse

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(x, **kw):
        return x

from _common import load_config, p
from phishrobust.paraphrase import build, quality_ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="paraphrases.csv")
    ap.add_argument("--limit", type=int, default=None, help="max e-mails (debug)")
    args = ap.parse_args()
    cfg = load_config()
    proc = p(cfg, "processed")
    df = pd.read_csv(proc / "emails.csv")
    todo = df[(df.label == 1) & (df.split.isin(args.splits))]
    if args.limit:
        todo = todo.head(args.limit)

    path = proc / args.out
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "text_para", "ok", "backend"])
    todo = todo[~todo.id.isin(done.id)]
    print(f"{len(todo)} e-mails to paraphrase ({len(done)} already done)")

    para = build(args.backend, cfg)
    bs, q = cfg["paraphrase"]["batch_size"], cfg["quality_filter"]
    rows = done.to_dict("records")
    for i in tqdm(range(0, len(todo), bs)):
        chunk = todo.iloc[i : i + bs]
        for (_, r), new in zip(chunk.iterrows(), para.rewrite(chunk.text.tolist())):
            rows.append({"id": r.id, "text_para": new, "backend": para.name,
                         "ok": quality_ok(r.text, new, q["min_len_ratio"], q["max_len_ratio"], q["min_url_retention"])})
        pd.DataFrame(rows).to_csv(path, index=False)   # checkpoint after every batch
    res = pd.DataFrame(rows)
    print(f"accepted by quality filter: {int(res.ok.sum())}/{len(res)}")
