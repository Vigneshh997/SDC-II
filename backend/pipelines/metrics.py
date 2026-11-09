"""Held-out multi-label metrics with undefined classes reported as null."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from backend.config import DISEASE_LABELS


def binary_reliability(targets, probabilities, bins=10):
    """Positive-event calibration, not thresholded prediction confidence.

    Equal-width bins are [left, right), with probability 1 in the last bin.
    Empty bins have null means. This measures calibration; it does not fit it.
    """
    y, p = np.asarray(targets), np.asarray(probabilities)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not y.size
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
        or type(bins) is not int
        or bins < 1
    ):
        raise ValueError("Expected aligned nonempty binary targets, probabilities and positive integer bins")
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        mean = float(p[selected].mean()) if count else None
        frequency = float(y[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean - frequency)
        reliability.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_probability": mean,
                "positive_fraction": frequency,
            }
        )
    return {
        "brier_score": float(np.mean((p.astype(float) - y) ** 2)),
        "positive_event_ece": float(ece),
        "reliability_bins": reliability,
    }


def compute_metrics(targets, probabilities, thresholds):
    targets, probabilities, thresholds = (
        np.asarray(targets),
        np.asarray(probabilities),
        np.asarray(thresholds),
    )
    if (
        targets.ndim != 2
        or targets.shape != probabilities.shape
        or targets.shape[1] != 14
        or not len(targets)
    ):
        raise ValueError("Expected nonempty N x 14 aligned targets and probabilities")
    if (
        not np.isin(targets, [0, 1]).all()
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Invalid binary targets or probabilities")
    if (
        thresholds.shape != (14,)
        or not np.isfinite(thresholds).all()
        or not ((thresholds > 0) & (thresholds < 1)).all()
    ):
        raise ValueError("Expected 14 valid decision thresholds")
    prediction = probabilities >= thresholds
    result = {
        "samples": len(targets),
        "micro_f1": float(f1_score(targets, prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(targets, prediction, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(targets, prediction, average="micro", zero_division=0)),
        "per_class": {},
    }
    for index, label in enumerate(DISEASE_LABELS):
        y, p, pred = targets[:, index], probabilities[:, index], prediction[:, index]
        both = len(np.unique(y)) == 2
        result["per_class"][label] = {
            **binary_reliability(y, p),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "pr_auc_average_precision": float(average_precision_score(y, p)) if both else None,
            "tp": int(((y == 1) & pred).sum()),
            "tn": int(((y == 0) & ~pred).sum()),
            "fp": int(((y == 0) & pred).sum()),
            "fn": int(((y == 1) & ~pred).sum()),
        }
    for metric in ("roc_auc", "pr_auc_average_precision"):
        values = [v[metric] for v in result["per_class"].values() if v[metric] is not None]
        result[f"macro_{metric}"] = float(np.mean(values)) if values else None
    for metric in ("brier_score", "positive_event_ece"):
        result[f"macro_{metric}"] = float(np.mean([v[metric] for v in result["per_class"].values()]))
    result["calibration_definition"] = (
        "Per-disease positive-event ECE: 10 equal-width probability bins, weighted by sample count. "
        "Macro values average diseases equally. Brier is mean squared probability error. "
        "These measurements do not establish calibrated or clinically reliable predictions."
    )
    return result


def select_thresholds(targets, probabilities):
    """Validation data only. Undefined classes retain a documented default 0.5."""
    thresholds = []
    for index in range(14):
        y, p = targets[:, index], probabilities[:, index]
        if len(np.unique(y)) < 2:
            thresholds.append(0.5)
        else:
            grid = np.linspace(0.05, 0.95, 19)
            scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
            thresholds.append(float(grid[np.argmax(scores)]))
    return thresholds
