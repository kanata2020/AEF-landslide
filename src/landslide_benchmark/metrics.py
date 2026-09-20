from collections import defaultdict
from typing import Dict

import torch


def empty_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def confusion_counts(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, int]:
    prediction, target = prediction.bool(), target.bool()
    return {
        "tp": int((prediction & target).sum()),
        "fp": int((prediction & ~target).sum()),
        "fn": int((~prediction & target).sum()),
        "tn": int((~prediction & ~target).sum()),
    }


def add_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key in total:
        total[key] += counts[key]


def _divide(a, b):
    return float(a / b) if b else 0.0


def class_metrics(tp, fp, fn):
    return {
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "iou": _divide(tp, tp + fp + fn),
    }


def detailed_metrics(counts: Dict[str, int]) -> Dict:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    positive = class_metrics(tp, fp, fn)
    negative = class_metrics(tn, fn, fp)
    macro = {key: (positive[key] + negative[key]) / 2 for key in positive}
    return {
        "landslide": positive,
        "non_landslide": negative,
        "macro": macro,
        "accuracy": _divide(tp + tn, tp + fp + fn + tn),
        "confusion_matrix": counts,
    }


def region_accumulators():
    return defaultdict(empty_counts), defaultdict(int)
