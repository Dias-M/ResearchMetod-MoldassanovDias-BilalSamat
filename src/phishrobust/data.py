"""Loading, cleaning, de-duplication and splitting of e-mail corpora.

Unified schema: id, text, label (1 = phishing, 0 = legitimate/ham), source, split.
"""
from __future__ import annotations

import email
import hashlib
import html
import mailbox
import re
from email import policy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)


def clean_text(text: str) -> str:
    text = html.unescape(_TAG.sub(" ", str(text)))
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(clean_text(text).lower().encode("utf-8")).hexdigest()


def _body(msg) -> str:
    part = msg.get_body(preferencelist=("plain", "html")) if msg.is_multipart() else msg
    if part is None:
        return ""
    try:
        return part.get_content()
    except Exception:  # undecodable charset etc.
        return ""


def _msg_to_text(msg) -> str:
    return f"{msg.get('subject', '') or ''}\n{_body(msg)}"


def read_mbox(path: str | Path, label: int, source: str) -> pd.DataFrame:
    box = mailbox.mbox(str(path), factory=lambda f: email.message_from_binary_file(f, policy=policy.default))
    rows = [{"text": _msg_to_text(m), "label": label, "source": source} for m in box]
    return pd.DataFrame(rows)


def read_email_dir(path: str | Path, label: int, source: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(path).rglob("*")):
        if p.is_file() and not p.name.startswith(("cmds", ".")):
            with open(p, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            rows.append({"text": _msg_to_text(msg), "label": label, "source": source})
    return pd.DataFrame(rows)


def prepare(df: pd.DataFrame, min_chars: int = 30, max_chars: int = 4000) -> pd.DataFrame:
    """Clean, truncate, drop duplicates and texts that appear with both labels."""
    df = df.copy()
    df["text"] = df["text"].map(clean_text).str.slice(0, max_chars)
    df = df[df["text"].str.len() >= min_chars].copy()
    df["hash"] = df["text"].map(text_hash)
    conflicting = df.groupby("hash")["label"].nunique()
    df = df[~df["hash"].isin(conflicting[conflicting > 1].index)]
    df = df.drop_duplicates("hash").reset_index(drop=True)
    df["id"] = df["hash"].str.slice(0, 12)
    df["label"] = df["label"].astype(int)
    return df[["id", "text", "label", "source", "hash"]]


def assign_splits(df: pd.DataFrame, train: float, val: float, seed: int) -> pd.DataFrame:
    """Stratified split by label x source. Each unique text lands in exactly one split."""
    df = df.copy()
    key = df["label"].astype(str) + "_" + df["source"].astype(str)
    counts = key.map(key.value_counts())
    key = key.where(counts >= 4, df["label"].astype(str))  # rare groups: stratify by label only
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, train_size=train, stratify=key, random_state=seed)
    va, te = train_test_split(rest, train_size=val / (1 - train), stratify=key.iloc[rest], random_state=seed)
    df["split"] = ""
    df.iloc[tr, df.columns.get_loc("split")] = "train"
    df.iloc[va, df.columns.get_loc("split")] = "val"
    df.iloc[te, df.columns.get_loc("split")] = "test"
    return df
