"""
evaluate.py — Test-set evaluation of the calibrated heart disease model:
metrics at 0.5 and at the saved cost-optimal threshold, bootstrapped 95 % confidence
intervals, Brier score, SHAP (grouped by clinical input) and permutation importance.

The threshold comes from models/metadata.json (chosen by train.py / the notebook on
out-of-fold TRAINING predictions); it is never re-tuned on the test set here.
Test metrics and intervals are written back into metadata.json, and permutation
importance to results/permutational_feature_importance.csv.

CLI (from heart/tabular/):
    python -m src.evaluate
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.utils import resample

from .data import (CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, RANDOM_STATE, get_X_y, keep_workers_light,
                   load_raw_data, register_notebook_functions, split_data)

MODULE_DIR = Path(__file__).resolve().parent.parent

# Pulse pressure and non-HDL cholesterol are exact combinations of raw columns, so their SHAP
# values are summed with their sources (SHAP values are additive), as in the notebook
FEATURE_GROUPS = {"blood_pressure": ["systolic_bp", "diastolic_bp", "pulse_pressure"],
                  "cholesterol": ["total_cholesterol", "hdl_cholesterol", "non_hdl_cholesterol", "tc_hdl_ratio"]}


def metrics_at(y_true, probas, threshold: float) -> dict:
    preds = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1_score": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probas),
        "pr_auc": average_precision_score(y_true, probas),
        "missed_patients_fn": int(fn),
        "false_alarms_fp": int(fp),
    }


def bootstrap_confidence_intervals(y_true, probas, threshold: float, n_iterations: int = 1000) -> dict:
    """95 % intervals from resampling the test set with replacement."""
    y_true = np.asarray(y_true)
    preds = (probas >= threshold).astype(int)
    samples = {"Recall": [], "Precision": [], "ROC-AUC": [], "PR-AUC": []}     # same keys as the notebook's metadata
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


def source_column(feature: str) -> str:
    """Encoded feature -> clinical input it belongs to (blood pressure, cholesterol or a categorical answer)."""
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    for column in CATEGORICAL_FEATURES:
        if feature.startswith(column + "_"):
            return column
    return feature


def grouped_shap_importance(calibrated_model, X_train, X_test, sample_size: int = 300):
    """
    Mean |SHAP| per clinical input for one of the calibrated model's base pipelines.
    Exact explainers only (logistic regression, tree ensembles); returns None otherwise.
    """
    import shap
    from sklearn.linear_model import LogisticRegression

    pipeline = calibrated_model.calibrated_classifiers_[0].estimator
    classifier = pipeline.named_steps["classifier"]
    preprocess = pipeline[:-1]
    feature_names = list(pipeline.named_steps["encoding"].get_feature_names_out())
    X_sample = preprocess.transform(X_test)[:sample_size]

    if isinstance(classifier, LogisticRegression):
        background = shap.sample(preprocess.transform(X_train), 100, random_state=RANDOM_STATE)
        values = shap.LinearExplainer(classifier, background)(X_sample).values
    elif hasattr(classifier, "get_booster") or hasattr(classifier, "booster_") or type(classifier).__name__ == "CatBoostClassifier":
        values = shap.TreeExplainer(classifier)(X_sample).values
    else:
        return None
    if values.ndim == 3:
        values = values[..., 1]

    grouped = pd.DataFrame(values, columns=feature_names).T.groupby(source_column).sum().T
    return grouped.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the calibrated heart disease model on the test set.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=str(MODULE_DIR / "models" / "heart_disease_calibrated_model.joblib"))
    parser.add_argument("--metadata-path", default=str(MODULE_DIR / "models" / "metadata.json"))
    parser.add_argument("--importance-out", default=str(MODULE_DIR / "results" / "permutational_feature_importance.csv"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    keep_workers_light()
    register_notebook_functions()          # models saved by the notebook reference __main__.add_custom_features
    X, y = get_X_y(load_raw_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(X, y)

    with open(args.metadata_path) as f:
        metadata = json.load(f)
    threshold = metadata["optimal_threshold"]

    model = joblib.load(args.model_path)
    probas = model.predict_proba(X_test)[:, 1]

    print(f"=== {metadata.get('best_model', 'model')} (calibrated) — test set: {len(X_test):,} patients ===")
    for label, t in [("default 0.5", 0.5), (f"cost-optimal {threshold:.4f}", threshold)]:
        m = metrics_at(y_test, probas, t)
        print(f"\n--- Threshold {label} ---")
        for k, v in m.items():
            print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")

    brier = brier_score_loss(y_test, probas)
    brier_reference = brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))
    print(f"\nBrier score: {brier:.4f} (always predicting the prevalence: {brier_reference:.4f})")

    intervals = bootstrap_confidence_intervals(y_test, probas, threshold, args.bootstrap_iterations)
    print(f"\n=== 95% bootstrapped confidence intervals ({args.bootstrap_iterations} iterations) ===")
    for metric, ci in intervals.items():
        print(f"{metric:10s}: {ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f} - {ci['ci_upper']:.3f})")

    shap_table = grouped_shap_importance(model, X_train, X_test)
    if shap_table is not None:
        print("\n=== Top 10 clinical inputs by mean |SHAP| (grouped) ===")
        print(shap_table.head(10).round(4).to_string())

    result = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    importance = pd.DataFrame({"Feature": X_test.columns, "Importance_Mean": result.importances_mean,
                               "Importance_Std": result.importances_std}).sort_values("Importance_Mean", ascending=False)
    Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_out, index=False)
    print("\n=== Top 10 permutation importance (drop in test PR-AUC) ===")
    print(importance.head(10).round(4).to_string(index=False))

    optimal = metrics_at(y_test, probas, threshold)
    metadata["test_metrics"] = {"roc_auc": optimal["roc_auc"], "pr_auc": optimal["pr_auc"], "brier_score": brier,
                                "recall_at_threshold": optimal["recall"],
                                "precision_at_threshold": optimal["precision"],
                                "specificity_at_threshold": optimal["specificity"]}
    metadata["bootstrap_confidence_intervals"] = intervals
    with open(args.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    print(f"\n[SAVED] {args.importance_out}\n[SAVED] test metrics -> {args.metadata_path}")


if __name__ == "__main__":
    main()
