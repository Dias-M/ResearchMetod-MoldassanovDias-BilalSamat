"""Three detector families behind one interface: fit(texts, y) / score(texts) -> higher = more phishing."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class TfidfLR:
    name = "tfidf_lr"

    def __init__(self, max_features=50000, ngram_range=(1, 2), C=4.0, seed=42):
        self.pipe = make_pipeline(
            TfidfVectorizer(max_features=max_features, ngram_range=tuple(ngram_range), sublinear_tf=True),
            LogisticRegression(C=C, max_iter=1000, random_state=seed),
        )

    def fit(self, texts, y):
        self.pipe.fit(list(texts), np.asarray(y))
        return self

    def score(self, texts):
        return self.pipe.decision_function(list(texts))


class DistilBert:
    name = "distilbert"

    def __init__(self, name="distilbert-base-uncased", epochs=2, lr=5e-5, batch_size=16, max_len=256, seed=42):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(self.device)
        self.epochs, self.lr, self.bs, self.max_len, self.seed = epochs, lr, batch_size, max_len, seed

    def _enc(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)

    def fit(self, texts, y):
        torch = self.torch
        texts, y = list(texts), np.asarray(y)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(texts))
            for i in range(0, len(order), self.bs):
                b = order[i : i + self.bs]
                enc = self._enc([texts[j] for j in b])
                out = self.model(**enc, labels=torch.tensor(y[b], device=self.device))
                out.loss.backward()
                opt.step()
                opt.zero_grad()
        return self

    def score(self, texts):
        torch = self.torch
        texts, res = list(texts), []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.bs * 2):
                logits = self.model(**self._enc(texts[i : i + self.bs * 2])).logits
                res.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
        return np.concatenate(res)


class ZeroShotLLM:
    """No training: score = logit('Yes') - logit('No') for 'Is this e-mail phishing?'."""

    name = "zero_shot_llm"
    PROMPT = (
        "You are an e-mail security filter. Decide whether the e-mail below is a phishing attempt. "
        "Answer with one word: Yes or No.\n\nE-mail:\n{email}"
    )

    def __init__(self, name="Qwen/Qwen2.5-1.5B-Instruct", max_chars=2000, batch_size=4, seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto").to(self.device).eval()
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no = self.tok.encode("No", add_special_tokens=False)[0]
        self.max_chars, self.bs = max_chars, batch_size

    def fit(self, texts, y):  # zero-shot: nothing to learn
        return self

    def score(self, texts):
        torch = self.torch
        texts, res = list(texts), []
        for i in range(0, len(texts), self.bs):
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": self.PROMPT.format(email=t[: self.max_chars])}],
                    tokenize=False, add_generation_prompt=True,
                )
                for t in texts[i : i + self.bs]
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]
            res.append((logits[:, self.yes] - logits[:, self.no]).float().cpu().numpy())
        return np.concatenate(res)


def build(name: str, cfg: dict, seed: int):
    m = cfg["models"][name]
    if name == "tfidf_lr":
        return TfidfLR(m["max_features"], m["ngram_range"], m["C"], seed)
    if name == "distilbert":
        return DistilBert(m["name"], m["epochs"], m["lr"], m["batch_size"], m["max_len"], seed)
    if name == "zero_shot_llm":
        return ZeroShotLLM(m["name"], m["max_chars"], m["batch_size"], seed)
    raise ValueError(name)
