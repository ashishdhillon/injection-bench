"""Paired comparison of two detectors scored on the SAME test items (needs --save-scores output)."""
import random
from typing import Dict, List

from .metrics import auc, recall_at_fpr


def paired_compare(labels: List[int], a: List[float], b: List[float], n: int = 500, seed: int = 0,
                   fprs=(0.01, 0.05)) -> Dict:
    """Stratified bootstrap over test items. The same resampled items score both detectors, so the
    difference reflects the detectors, not sampling luck. Returns point estimates and 95% CIs of (a - b)."""
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]

    def metrics(scores, ip, ineg):
        p, q = [scores[i] for i in ip], [scores[i] for i in ineg]
        out = {"auc": auc(p, q)}
        for f in fprs:
            out[f"recall@{int(f * 100)}%fpr"] = recall_at_fpr(p, q, f)
        return out

    base_a, base_b = metrics(a, pos, neg), metrics(b, pos, neg)
    r = random.Random(seed)
    diffs = {k: [] for k in base_a}
    for _ in range(n):
        ip = [r.choice(pos) for _ in pos]
        ineg = [r.choice(neg) for _ in neg]
        ma, mb = metrics(a, ip, ineg), metrics(b, ip, ineg)
        for k in diffs:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k, v in diffs.items():
        v.sort()
        lo, hi = v[int(0.025 * n)], v[min(n - 1, int(0.975 * n))]
        out[k] = {"a": round(base_a[k], 4), "b": round(base_b[k], 4), "diff": round(base_a[k] - base_b[k], 4),
                  "ci95": [round(lo, 4), round(hi, 4)], "significant": bool(lo > 0 or hi < 0)}
    return out


def to_markdown(name_a: str, name_b: str, res: Dict) -> str:
    lines = [f"Paired comparison, {name_a} minus {name_b} (same test items, stratified bootstrap).", "",
             "| metric | " + name_a + " | " + name_b + " | difference (95% CI) | interval excludes 0 |", "|---|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['diff']:+.3f} ({v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f}) | {'yes' if v['significant'] else 'no'} |")
    return "\n".join(lines) + "\n"
