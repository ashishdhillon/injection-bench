"""Metrics for detector evaluation. Pure Python."""
import math
import random
from bisect import bisect_left, bisect_right
from typing import Callable, Dict, List, Sequence, Tuple


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    ns = sorted(neg)
    tot = 0.0
    for p in pos:
        tot += (bisect_left(ns, p) + 0.5 * (bisect_right(ns, p) - bisect_left(ns, p))) / len(ns)
    return tot / len(pos)


def threshold_at_fpr(neg: Sequence[float], fpr: float) -> float:
    """Smallest threshold t such that the share of negatives scoring > t is at most fpr."""
    s = sorted(neg)
    k = math.ceil(len(s) * (1 - fpr))
    return s[min(max(k, 1), len(s)) - 1]


def recall_at_fpr(pos: Sequence[float], neg: Sequence[float], fpr: float) -> float:
    t = threshold_at_fpr(neg, fpr)
    return sum(1 for p in pos if p > t) / len(pos)


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0


def bootstrap_ci(metric: Callable[[List[float], List[float]], float], pos: Sequence[float], neg: Sequence[float],
                 n: int = 200, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    r = random.Random(seed)
    vals = []
    for _ in range(n):
        p = [r.choice(pos) for _ in pos]
        q = [r.choice(neg) for _ in neg]
        vals.append(metric(p, q))
    vals.sort()
    return vals[int(alpha / 2 * n)], vals[min(n - 1, int((1 - alpha / 2) * n))]


def summarize(pos: Sequence[float], neg: Sequence[float], fprs=(0.01, 0.05), prevalence=0.05, n_boot=200) -> Dict:
    out = {"n_pos": len(pos), "n_neg": len(neg), "auc": round(auc(pos, neg), 4)}
    lo, hi = bootstrap_ci(auc, pos, neg, n_boot)
    out["auc_ci95"] = [round(lo, 4), round(hi, 4)]
    for f in fprs:
        r = recall_at_fpr(pos, neg, f)
        lo, hi = bootstrap_ci(lambda p, q, f=f: recall_at_fpr(p, q, f), pos, neg, n_boot)
        out[f"recall@{int(f * 100)}%fpr"] = round(r, 4)
        out[f"recall@{int(f * 100)}%fpr_ci95"] = [round(lo, 4), round(hi, 4)]
        out[f"precision@{int(f * 100)}%fpr_prev{int(prevalence * 100)}%"] = round(precision_at_prevalence(r, f, prevalence), 4)
    return out
