"""Detector protocol and built-in baselines.

A detector is any object with  score(texts: list[str]) -> list[float]  (higher = more likely an
attack) and, optionally, fit(texts, labels). Load your own with  --detector package.module:factory
where factory() returns such an object."""
import importlib
import math
import random
import re
import zlib
from typing import List, Sequence


class LengthBaseline:
    """Scores by text length only. If this beats your detector, the dataset has a length shortcut."""
    name = "length-only"

    def score(self, texts: Sequence[str]) -> List[float]:
        return [math.log1p(len(t)) for t in texts]


class KeywordBaseline:
    """A small hand-written keyword counter, like a first-pass rule list."""
    name = "keywords"
    WORDS = ["jailbreak", "do anything now", "developer mode", "ignore previous", "ignore all previous",
             "no restrictions", "without restrictions", "stay in character", "dan ", "unfiltered", "uncensored",
             "bypass", "pretend you", "act as", "you are now", "no ethical", "never refuse"]

    def score(self, texts: Sequence[str]) -> List[float]:
        low = [t.casefold() for t in texts]
        return [float(sum(w in t for w in self.WORDS)) for t in low]


class HashedLR:
    """Trainable pure-Python baseline: hashed word uni+bigrams, logistic regression via SGD."""
    name = "hashed-lr"
    DIM = 1 << 16

    def __init__(self, epochs: int = 12, lr: float = 0.5, seed: int = 0, neg_ratio: int = 4):
        self.epochs, self.lr, self.seed, self.neg_ratio = epochs, lr, seed, neg_ratio
        self.w, self.b = {}, 0.0

    def _feat(self, text):
        toks = re.findall(r"[^\W_]+", text[:4000].casefold())
        f = {}
        for g in toks + [a + " " + b for a, b in zip(toks, toks[1:])]:
            i = zlib.crc32(g.encode()) % self.DIM
            f[i] = f.get(i, 0.0) + 1.0
        n = math.sqrt(sum(v * v for v in f.values())) or 1.0
        return {i: v / n for i, v in f.items()}

    def fit(self, texts, labels):
        r = random.Random(self.seed)
        pos = [t for t, y in zip(texts, labels) if y]
        neg = [t for t, y in zip(texts, labels) if not y]
        r.shuffle(neg)
        data = [(self._feat(t), 1) for t in pos] + [(self._feat(t), 0) for t in neg[: self.neg_ratio * len(pos)]]
        lr = self.lr
        for _ in range(self.epochs):
            r.shuffle(data)
            for x, y in data:
                z = self.b + sum(self.w.get(i, 0.0) * v for i, v in x.items())
                g = 1 / (1 + math.exp(-max(min(z, 30), -30))) - y
                for i, v in x.items():
                    self.w[i] = self.w.get(i, 0.0) * (1 - lr * 1e-5) - lr * g * v
                self.b -= lr * g
            lr *= 0.85
        return self

    def score(self, texts):
        return [self.b + sum(self.w.get(i, 0.0) * v for i, v in self._feat(t).items()) for t in texts]


class _GuardkitAdapter:
    def __init__(self, mode, model="jailbreak"):
        from guardkit import InjectionGuard    # pip install guardkit
        self.g = InjectionGuard(mode=mode, model=model)
        self.mode = mode
        self.name = f"guardkit:{mode}"

    def score(self, texts):
        if self.mode == "model":                      # continuous probability, not the thresholded finding
            return [self.g._model.score(t) for t in texts]
        out = []
        for t in texts:
            fs = self.g.scan(t)
            out.append(max((f.score for f in fs), default=0.0))
        return out


BUILTIN = {
    "length": LengthBaseline, "keywords": KeywordBaseline, "hashed-lr": HashedLR,
    "guardkit:rules": lambda: _GuardkitAdapter("rules"),
    "guardkit:model": lambda: _GuardkitAdapter("model"),
}


def load_detector(spec: str):
    if spec in BUILTIN:
        return BUILTIN[spec]()
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"unknown detector {spec!r}; builtins: {sorted(BUILTIN)} or 'module:factory'")
    obj = getattr(importlib.import_module(mod), attr)
    return obj() if callable(obj) and not hasattr(obj, "score") else obj
