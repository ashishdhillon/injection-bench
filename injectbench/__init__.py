"""injectbench: leakage-controlled, time-split evaluation of prompt-injection / jailbreak detectors."""
from .data import Row, temporal_split
from .metrics import auc, recall_at_fpr, summarize
from .run import evaluate

__version__ = "0.2.0"
__all__ = ["Row", "temporal_split", "auc", "recall_at_fpr", "summarize", "evaluate"]
