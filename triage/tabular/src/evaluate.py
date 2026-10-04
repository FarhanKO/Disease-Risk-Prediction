"""
evaluate.py — Test-set evaluation of the three calibrated heads and of the router
they make together, as recorded and with every vital sign removed:

    per head:    ROC-AUC, PR-AUC, Brier score, recall / precision / specificity at the
                 saved cost-optimal threshold, bootstrapped 95 % confidence intervals
    the router:  per-organ recall and precision, "all diagnosed organs routed",
                 "other visits left alone", organ models run per visit
    importance:  permutation importance of the raw inputs per head
                 (results/permutational_feature_importance.csv)

The thresholds come from models/metadata.json (chosen by train.py / the notebook on
out-of-fold TRAINING predictions); they are never re-tuned on the test set here. The
test metrics and intervals are written back into metadata.json.

CLI (from triage/tabular/):
    python -m src.evaluate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.utils import resample

from common.utils.artifacts import load_pipeline

from . import data
from .data import (DEFAULT_DATA_PATH, MODULE_DIR, ORGANS, RANDOM_STATE, VITAL_SIGNS, get_X_y, keep_workers_light,
                   load_raw_data, split_data)
from .routing import route_visits, routing_metrics
from .train import DEFAULT_N_JOBS, MODEL_DIR, artifact_paths


def metrics_at(y_true, probas, threshold: float) -> dict:
    preds = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probas),
        "pr_auc": average_precision_score(y_true, probas),
        "brier_score": brier_score_loss(y_true, probas),
        "recall_at_threshold": recall_score(y_true, preds, zero_division=0),
        "precision_at_threshold": precision_score(y_true, preds, zero_division=0),
        "specificity_at_threshold": tn / (tn + fp),
        "missed_fn": int(fn), "routed_unnecessarily_fp": int(fp),
    }


def bootstrap_confidence_intervals(y_true, probas, threshold: float, n_iterations: int = 1000) -> dict:
    """95 % intervals from resampling the test set with replacement (same keys as the notebook)."""
    y_true = np.asarray(y_true)
    preds = (probas >= threshold).astype(int)
    samples = {"Recall": [], "Precision": [], "ROC-AUC": [], "PR-AUC": []}
    for i in range(n_iterations):
        idx = resample(np.arange(len(y_true)), replace=True, n_samples=len(y_true), random_state=i)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples["Recall"].append(recall_score(y_true[idx], preds[idx], zero_division=0))
        samples["Precision"].append(precision_score(y_true[idx], preds[idx], zero_division=0))
        samples["ROC-AUC"].append(roc_auc_score(y_true[idx], probas[idx]))
        samples["PR-AUC"].append(average_precision_score(y_true[idx], probas[idx]))
    return {metric: {"mean": float(np.mean(v)), "ci_lower": float(np.percentile(v, 2.5)),
                     "ci_upper": float(np.percentile(v, 97.5))} for metric, v in samples.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the symptom-triage heads and router on the test set.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="artifacts to evaluate (default: models/)")
    parser.add_argument("--importance-out", default=str(MODULE_DIR / "results" / "permutational_feature_importance.csv"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args()

    keep_workers_light()
    X, Y = get_X_y(load_raw_data(args.data_path))
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    X_test_no_vitals = X_test.copy()
    X_test_no_vitals[VITAL_SIGNS] = np.nan
    test_sets = {"with vital signs": X_test, "without vital signs": X_test_no_vitals}

    paths = artifact_paths(args.model_dir)
    metadata = json.loads(paths["metadata"].read_text())
    thresholds = {organ: metadata["heads"][organ]["optimal_threshold"] for organ in ORGANS}
    models = {organ: load_pipeline(paths["heads"][organ], data) for organ in ORGANS}
    probas = {inputs: pd.DataFrame({organ: models[organ].predict_proba(X_in)[:, 1] for organ in ORGANS}, index=X_test.index)
              for inputs, X_in in test_sets.items()}

    print(f"=== Symptom triage — test set: {len(X_test):,} visits ===")
    for organ in ORGANS:
        head = metadata["heads"][organ]
        print(f"\n--- {organ} head: {head['best_model']} (calibrated), threshold {thresholds[organ]:.4f}, "
              f"prevalence {Y_test[organ].mean():.2%} ---")
        for inputs in test_sets:
            m = metrics_at(Y_test[organ], probas[inputs][organ].values, thresholds[organ])
            key = "test_metrics" if inputs == "with vital signs" else "test_metrics_no_vital_signs"
            head[key] = {k: v for k, v in m.items() if k not in ("missed_fn", "routed_unnecessarily_fp")}
            print(f"{inputs:20s}: " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
        head["bootstrap_confidence_intervals"] = {
            inputs: bootstrap_confidence_intervals(Y_test[organ], probas[inputs][organ].values, thresholds[organ],
                                                   args.bootstrap_iterations) for inputs in test_sets}
        for inputs, intervals in head["bootstrap_confidence_intervals"].items():
            print(f"95% CI, {inputs}: " + " | ".join(f"{metric} {ci['mean']:.3f} ({ci['ci_lower']:.3f}-{ci['ci_upper']:.3f})"
                                                   for metric, ci in intervals.items()))

    metadata["routing_test_metrics"] = {}
    print("\n=== Router (the three heads together) ===")
    for inputs in test_sets:
        flags, _ = route_visits(probas[inputs], thresholds)
        metadata["routing_test_metrics"][inputs] = routing_metrics(Y_test, flags)
        print(f"{inputs}:")
        for name, value in metadata["routing_test_metrics"][inputs].items():
            print(f"  {name:36s} {value:.4f}")

    rows = []
    for organ in ORGANS:
        result = permutation_importance(models[organ], X_test, Y_test[organ], scoring="average_precision", n_repeats=10,
                                        random_state=RANDOM_STATE, n_jobs=args.n_jobs)
        rows += [{"Head": organ, "Feature": f, "Importance_Mean": m, "Importance_Std": s}
                 for f, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
    importance = pd.DataFrame(rows).sort_values(["Head", "Importance_Mean"], ascending=[True, False])
    Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_out, index=False)
    print("\n=== Top 5 permutation importances per head (drop in test PR-AUC) ===")
    print(importance.groupby("Head").head(5).round(4).to_string(index=False))

    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=float))
    print(f"\n[SAVED] {args.importance_out}\n[SAVED] test metrics -> {paths['metadata']}")


if __name__ == "__main__":
    main()
