"""
image.py — Re-scoring an image module's test split with the src/ cascade and
comparing it with the test metrics its notebook recorded in models/metadata.json.
Used by the ECG and CT modules' `python -m src.evaluate`, whose models are trained in
the notebooks: this is the check that src/ serves exactly what the notebook evaluated.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from ..image import ImageModel


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    classes = list(range(probabilities.shape[1]))
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "macro_roc_auc": roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", labels=classes),
        "log_loss": log_loss(labels, probabilities, labels=classes),
    }


def confidence_coverage(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Share of images that clear the Stage 3 threshold, and how accurate those and the rest are."""
    accepted = probabilities.max(axis=1) >= threshold
    correct = probabilities.argmax(axis=1) == labels
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accuracy_accepted": float(correct[accepted].mean()) if accepted.any() else None,
        "accuracy_flagged": float(correct[~accepted].mean()) if (~accepted).any() else None,
    }


def compare_with_notebook(model: ImageModel, images: np.ndarray, labels: np.ndarray, test_set_name: str,
                          ensemble: bool = False, tolerance: float = 0.005) -> bool:
    """
    Print notebook vs src/ for the test metrics, the confidence coverage and the OOD gate's
    rejection rate on real test images. Returns True when every difference is within `tolerance`.
    """
    model.load()
    rejected, probabilities = model.score(images, ensemble=ensemble)
    notebook = model.metadata["ensemble_test_metrics" if ensemble else "test_metrics"]
    rows = [(name, notebook[name], value) for name, value in classification_metrics(labels, probabilities).items()]
    if not ensemble:
        coverage = confidence_coverage(labels, probabilities, model.threshold)
        rows += [(f"confidence {name}", model.metadata["confidence_coverage"][name], coverage[name])
                 for name in ("coverage", "accuracy_accepted", "accuracy_flagged")]
    rows.append((f"OOD gate: {test_set_name} rejected", model.metadata["ood_gate"]["rejected"][test_set_name],
                 float(rejected.mean())))

    ok = True
    print(f"{'':42s} {'notebook':>10s} {'src/':>10s} {'diff':>9s}")
    for name, expected, actual in rows:
        diff = actual - expected
        ok &= abs(diff) <= tolerance
        print(f"{name:42s} {expected:10.4f} {actual:10.4f} {diff:+9.4f}{'' if abs(diff) <= tolerance else '  <-- differs'}")
    print(f"\n{'OK' if ok else 'MISMATCH'}: {len(labels)} test images, {model.best_model if not ensemble else 'ensemble'}, "
          f"tolerance {tolerance}")
    return ok
