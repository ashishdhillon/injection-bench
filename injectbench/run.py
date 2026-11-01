"""Run detectors on a train/test split and render a report."""
import time
from typing import Dict, List

from .data import Row
from .detectors import load_detector
from .metrics import summarize


def evaluate(detector, train: List[Row], test: List[Row], prevalence: float = 0.05, n_boot: int = 200,
             scores_out: Dict = None) -> Dict:
    """If `scores_out` is a dict, the detector's raw test scores are stored in it (for paired comparisons)."""
    if hasattr(detector, "fit"):
        detector.fit([r.text for r in train], [r.label for r in train])
    t0 = time.time()
    scores = detector.score([r.text for r in test])
    dt = time.time() - t0
    if scores_out is not None:
        scores_out["labels"] = [r.label for r in test]
        scores_out.setdefault("scores", {})[getattr(detector, "name", type(detector).__name__)] = [float(x) for x in scores]
    pos = [s for s, r in zip(scores, test) if r.label == 1]
    neg = [s for s, r in zip(scores, test) if r.label == 0]
    res = summarize(pos, neg, prevalence=prevalence, n_boot=n_boot)
    res["ms_per_item"] = round(1000 * dt / max(len(test), 1), 3)
    res["name"] = getattr(detector, "name", type(detector).__name__)
    return res


def to_markdown(results: List[Dict], split_info: str, prevalence: float = 0.05) -> str:
    p = f"prev{int(prevalence * 100)}%"
    lines = [split_info, "",
             f"| detector | AUC (95% CI) | recall @1% FPR (95% CI) | recall @5% FPR | precision @1% FPR, {int(prevalence*100)}% prevalence | ms/item |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['auc']:.3f} ({r['auc_ci95'][0]:.2f}-{r['auc_ci95'][1]:.2f}) "
                     f"| {r['recall@1%fpr']:.3f} ({r['recall@1%fpr_ci95'][0]:.2f}-{r['recall@1%fpr_ci95'][1]:.2f}) "
                     f"| {r['recall@5%fpr']:.3f} | {r['precision@1%fpr_' + p]:.3f} | {r['ms_per_item']} |")
    return "\n".join(lines) + "\n"
