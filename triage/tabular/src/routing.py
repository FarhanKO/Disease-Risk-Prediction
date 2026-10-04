"""
routing.py — How the three organ heads become one routing decision, and how the
router is scored. Shared by train.py, evaluate.py and predict.py, and identical to
the notebook's cost-threshold and routing cells.

Routing rule: a patient goes to every organ whose calibrated probability is at or
above that head's cost-optimal threshold. The first route is the organ furthest
above its own threshold (probability / threshold), because the thresholds differ
between heads. No organ reached means "other".
"""

import numpy as np
import pandas as pd

from .data import ORGANS

OTHER = "other"
FN_COST, FP_COST = 5, 1        # a missed organ costs 5x an unnecessary routing (one extra organ model run)


def threshold_costs(y_true: np.ndarray, probas: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """5·FN + 1·FP at every candidate threshold (positive when p >= t)."""
    positives = np.sort(probas[y_true == 1])
    negatives = np.sort(probas[y_true == 0])
    missed = np.searchsorted(positives, candidates, side="left")
    false_alarms = len(negatives) - np.searchsorted(negatives, candidates, side="left")
    return FN_COST * missed + FP_COST * false_alarms


def cost_optimal_threshold(y_true, probas) -> float:
    """The candidate (an observed probability) with the lowest 5·FN + FP; the lowest one on ties."""
    y_true, probas = np.asarray(y_true), np.asarray(probas)
    candidates = np.unique(probas)
    return float(candidates[int(np.argmin(threshold_costs(y_true, probas, candidates)))])


def route_visits(probabilities, thresholds: dict) -> tuple[pd.DataFrame, pd.Series]:
    """(one True/False column per organ, the first route of each visit or "other")."""
    probs = pd.DataFrame(probabilities)[ORGANS]
    cutoffs = pd.Series(thresholds)[ORGANS]
    flags = probs.ge(cutoffs)
    strength = probs.div(cutoffs).where(flags)
    primary = pd.Series(OTHER, index=probs.index)
    routed = flags.any(axis=1)
    primary[routed] = strength[routed].idxmax(axis=1)
    return flags, primary


def routing_metrics(Y_true: pd.DataFrame, flags: pd.DataFrame) -> dict:
    """The router as a whole: per-organ recall and precision, and visit-level outcomes (the notebook's names)."""
    Y_true = Y_true.astype(bool)
    has_organ = Y_true.any(axis=1)
    metrics = {f"{o.capitalize()} recall": float((flags[o] & Y_true[o]).sum() / Y_true[o].sum()) for o in ORGANS}
    metrics.update({f"{o.capitalize()} precision": float((flags[o] & Y_true[o]).sum() / max(flags[o].sum(), 1))
                    for o in ORGANS})
    metrics["All diagnosed organs routed"] = float(((flags | ~Y_true).all(axis=1) & has_organ).sum() / has_organ.sum())
    metrics["At least one diagnosed organ routed"] = float((flags & Y_true).any(axis=1).sum() / has_organ.sum())
    metrics["Other visits left alone"] = float((~flags.any(axis=1) & ~has_organ).sum() / (~has_organ).sum())
    metrics["Visits routed to any organ"] = float(flags.any(axis=1).mean())
    metrics["Organ models run per visit"] = float(flags.sum(axis=1).mean())
    return metrics
