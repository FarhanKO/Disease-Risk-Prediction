"""
evaluate.py — Post-training evaluation for the calibrated kidney disease model:
core metrics, cost-based threshold optimization, calibration check (Brier score),
bootstrapped confidence intervals, SHAP and permutation feature importance.

Note: unlike the heart module (always CatBoost), the kidney winner can be
any of 9 model types, so SHAP falls back from TreeExplainer to a generic
Explainer when the winning model isn't tree-based.

CLI:
    python -m src.evaluate --data-path data/kidney_disease.csv --model-path models/kidney_tabular_calibrated.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              brier_score_loss, confusion_matrix, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score, roc_auc_score, roc_curve)
from sklearn.utils import resample

from src.data import get_X_y, load_raw_data, split_data


def core_metrics(y_test, preds, probas) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": auc(*roc_curve(y_test, probas)[:2]),
        "pr_auc": average_precision_score(y_test, probas),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def find_optimal_threshold(y_test, probas, fn_cost: float = 5, fp_cost: float = 1) -> float:
    """
    Clinical cost function: false negatives (missed CKD) are weighted
    `fn_cost`x more than false positives (false alarms). Returns the
    probability cutoff that minimizes total cost.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, probas)
    cost = fn_cost * (1 - recalls[:-1]) + fp_cost * (1 - precisions[:-1])
    best_idx = np.argmin(cost)
    return float(thresholds[best_idx])


def bootstrap_confidence_intervals(y_test, preds, probas, n_iterations: int = 1000) -> dict:
    """95% CI for recall, ROC-AUC, and PR-AUC via resampling with replacement."""
    y_test_array = np.asarray(y_test)
    n_size = len(y_test_array)

    recalls, roc_aucs, pr_aucs = [], [], []
    for i in range(n_iterations):
        idx = resample(np.arange(n_size), replace=True, n_samples=n_size, random_state=i)
        y_true_b, y_pred_b, y_proba_b = y_test_array[idx], preds[idx], probas[idx]
        if len(np.unique(y_true_b)) < 2:
            continue
        recalls.append(recall_score(y_true_b, y_pred_b, zero_division=0))
        roc_aucs.append(roc_auc_score(y_true_b, y_proba_b))
        pr_aucs.append(average_precision_score(y_true_b, y_proba_b))

    def ci(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }

    return {"recall": ci(recalls), "roc_auc": ci(roc_aucs), "pr_auc": ci(pr_aucs)}


def calibration_brier_score(y_test, probas) -> float:
    return float(brier_score_loss(y_test, probas))


def _get_base_pipeline(calibrated_model):
    """Pull a fitted base pipeline out of a CalibratedClassifierCV for SHAP/permutation use."""
    return calibrated_model.calibrated_classifiers_[0].estimator


def _transform_features(base_pipeline, X):
    X_eng = base_pipeline.named_steps["engineering"].transform(X)
    X_processed = base_pipeline.named_steps["transformations"].transform(X_eng)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    try:
        feature_names = [n.split("__")[-1] for n in base_pipeline.named_steps["transformations"].get_feature_names_out()]
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    return X_processed, feature_names


def shap_feature_importance(calibrated_model, X_test, sample_size: int = 200) -> pd.DataFrame:
    import shap

    base_pipeline = _get_base_pipeline(calibrated_model)
    X_processed, feature_names = _transform_features(base_pipeline, X_test)
    X_sample = X_processed[:sample_size]
    clf = base_pipeline.named_steps["classifier"]

    try:
        # Fast path for tree-based winners (RF, XGBoost, LightGBM, DecisionTree)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(X_sample).values
    except Exception:
        # Generic fallback for LR / KNN / NB / SVM / MLP — slower but always works
        background = X_sample[:50]
        explainer = shap.Explainer(clf.predict_proba, background)
        raw_values = explainer(X_sample).values
        shap_values = raw_values[..., 1] if raw_values.ndim == 3 else raw_values

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_feature_importance(calibrated_model, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    base_pipeline = _get_base_pipeline(calibrated_model)
    X_processed, feature_names = _transform_features(base_pipeline, X_test)

    result = permutation_importance(
        base_pipeline.named_steps["classifier"], X_processed, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )
    return pd.DataFrame({"feature": feature_names, "importance_mean": result.importances_mean}) \
        .sort_values("importance_mean", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the calibrated kidney disease model.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", default="models/kidney_tabular_calibrated.joblib")
    parser.add_argument("--metadata-out", default="models/metadata.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    df = load_raw_data(args.data_path)
    X, y = get_X_y(df)
    _, X_test, _, y_test = split_data(X, y)

    model = joblib.load(args.model_path)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = core_metrics(y_test, preds, probas)
    print("=== Core Metrics (default 0.5 threshold) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    optimal_threshold = find_optimal_threshold(y_test, probas)
    optimal_preds = (probas >= optimal_threshold).astype(int)
    optimal_metrics = core_metrics(y_test, optimal_preds, probas)
    print(f"\n=== Metrics at Cost-Optimal Threshold ({optimal_threshold:.4f}) ===")
    for k, v in optimal_metrics.items():
        print(f"{k}: {v}")

    brier = calibration_brier_score(y_test, probas)
    print(f"\nBrier Score: {brier:.4f}")

    print("\n=== 95% Bootstrapped Confidence Intervals ===")
    ci_results = bootstrap_confidence_intervals(y_test, preds, probas, args.bootstrap_iterations)
    print(ci_results)

    print("\n=== Top 10 SHAP Features ===")
    print(shap_feature_importance(model, X_test).head(10))

    print("\n=== Top 10 Permutation Importance Features ===")
    print(permutation_feature_importance(model, X_test, y_test).head(10))

    metadata = {
        "optimal_threshold": optimal_threshold,
        "brier_score": brier,
        "metrics_at_default_threshold": metrics,
        "metrics_at_optimal_threshold": optimal_metrics,
        "bootstrap_confidence_intervals": ci_results,
    }
    out_path = Path(args.metadata_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[SAVED] Evaluation metadata -> {out_path}")


if __name__ == "__main__":
    main()
