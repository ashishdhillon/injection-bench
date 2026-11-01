"""Datasets, leakage-controlled temporal splits and loaders."""
import csv
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Tuple

JAILBREAKHUB_BASE = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/"
JAILBREAKHUB_FILES = {1: "jailbreak_prompts_2023_12_25.csv", 0: "regular_prompts_2023_12_25.csv"}


@dataclass(frozen=True)
class Row:
    text: str
    label: int          # 1 = attack / jailbreak, 0 = benign
    date: str = ""      # ISO date or timestamp; "" if unknown


def download_jailbreakhub(dest: str) -> None:
    """Fetch the in-the-wild jailbreak dataset (Shen et al., CCS 2024; MIT licence)."""
    os.makedirs(dest, exist_ok=True)
    for name in JAILBREAKHUB_FILES.values():
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(JAILBREAKHUB_BASE + name, path)


def load_jailbreakhub(directory: str) -> List[Row]:
    csv.field_size_limit(10 ** 8)
    rows = []
    for label, name in JAILBREAKHUB_FILES.items():
        with open(os.path.join(directory, name), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["prompt"].strip():
                    rows.append(Row(r["prompt"], label, r.get("created_at", "")))
    return rows


def load_jsonl(path: str) -> List[Row]:
    """JSONL with keys text, label (and optional date)."""
    with open(path, encoding="utf-8") as fh:
        return [Row(d["text"], int(d["label"]), d.get("date", "")) for d in map(json.loads, fh) if d["text"].strip()]


def dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()[:300]


def temporal_split(rows: Iterable[Row], train_frac: float = 0.65) -> Tuple[List[Row], List[Row]]:
    """Train on earlier items, test on later ones. The cut is the date at which `train_frac` of the
    positives have occurred. Near-duplicates (same normalised 300-char prefix) are removed within
    each side and any test item duplicating a train item is dropped, so no prompt leaks across.
    Rows without a date are excluded."""
    rows = sorted((r for r in rows if r.date), key=lambda r: r.date)     # undated rows cannot be time-split
    dates = sorted(r.date for r in rows if r.label == 1)
    if not dates:
        raise ValueError("temporal_split needs dated positive rows")
    cut = dates[int(train_frac * len(dates))]
    train, test, seen_tr, seen_te = [], [], set(), set()
    for r in rows:
        k = dedupe_key(r.text)
        if r.date < cut and k not in seen_tr:
            seen_tr.add(k); train.append(r)
    for r in rows:
        k = dedupe_key(r.text)
        if r.date >= cut and k not in seen_tr and k not in seen_te:
            seen_te.add(k); test.append(r)
    return train, test
