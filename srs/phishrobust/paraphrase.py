"""LLM paraphrasing of EXISTING phishing corpus e-mails (for detector-robustness evaluation).

Only text from public research corpora is rewritten; nothing is generated from scratch,
sent to anyone, or personalised.
"""
from __future__ import annotations

import random
import re

from .data import URL_RE

SYSTEM_PROMPT = (
    "You rewrite e-mails for a text-classification robustness study. Rewrite the e-mail you "
    "receive in different words. Keep the same meaning, tone and structure. Keep every URL, "
    "e-mail address and number exactly as written. Output only the rewritten e-mail."
)

_SYN = {
    "please": "kindly", "immediately": "right away", "verify": "confirm", "account": "profile",
    "click": "follow", "urgent": "time-sensitive", "update": "refresh", "information": "details",
    "security": "safety", "password": "passcode", "important": "essential", "confirm": "validate",
}


class MockParaphraser:
    """Deterministic word-substitution stub. Used ONLY to smoke-test the pipeline offline."""

    name = "mock"

    def rewrite(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            parts = re.split(r"(\s+)", t)
            out.append("".join(_SYN.get(w.lower().strip(".,:;!?"), w) if not URL_RE.match(w) else w for w in parts))
        return out


class HFParaphraser:
    def __init__(self, model: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, seed: int = 42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(self.device).eval()
        self.kw = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
        torch.manual_seed(seed)

    def rewrite(self, texts: list[str]) -> list[str]:
        prompts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in texts
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**enc, pad_token_id=self.tok.pad_token_id, **self.kw)
        n = enc["input_ids"].shape[1]
        return [self.tok.decode(g[n:], skip_special_tokens=True).strip() for g in gen]


def quality_ok(orig: str, new: str, min_len_ratio=0.5, max_len_ratio=1.5, min_url_retention=1.0) -> bool:
    """Accept a paraphrase only if it is non-trivial, similar in length and keeps all URLs."""
    if not new or new.strip() == orig.strip():
        return False
    ratio = len(new) / max(len(orig), 1)
    if not (min_len_ratio <= ratio <= max_len_ratio):
        return False
    urls = set(URL_RE.findall(orig))
    if urls:
        kept = sum(u in new for u in urls) / len(urls)
        if kept < min_url_retention:
            return False
    return True


def build(backend: str, cfg: dict):
    if backend == "mock":
        return MockParaphraser()
    if backend == "hf":
        p = cfg["paraphrase"]
        return HFParaphraser(p["hf_model"], p["max_new_tokens"], p["temperature"], p["top_p"], p["seed"])
    raise ValueError(f"unknown backend {backend!r}")
