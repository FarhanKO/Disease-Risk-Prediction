"""
evaluate.py — Evaluates the trained kidney cascade on the held-out test split:
metrics at the saved threshold, bootstrapped 95 % confidence intervals, Brier
score, the Stage 1 gate's rejection rates, SHAP and permutation importance.

The threshold is NOT re-tuned here: train.py chose it on out-of-fold training
predictions, and tuning it on the test set would make these numbers optimistic.

CLI (run from models/kidney/tabular/):
    python -m src.evaluate
    python -m src.evaluate --importance-out results/permutational_feature_importance.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.utils import resample

from .data import DEFAULT_DATA_PATH, RANDOM_STATE, get_X_y, load_raw_data, split_data
from .predict import DEFAULT_ANOMALY_MODEL_PATH, DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, load_threshold
from .train import N_JOBS


def core_metrics(y_test, preds, probas) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probas),
        "pr_auc": average_precision_score(y_test, probas),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def bootstrap_confidence_intervals(y_test, preds, probas, n_iterations: int = 1000) -> dict:
    """95 % CI for recall, precision, ROC-AUC and PR-AUC via resampling the test set."""
    y_true, preds, probas = np.asarray(y_test), np.asarray(preds), np.asarray(probas)
    samples = {"recall": [], "precision": [], "roc_auc": [], "pr_auc": []}
    for i in range(n_iterations):
        idx = resample(np.arange(len(y_true)), replace=True, random_state=i)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples["recall"].append(recall_score(y_true[idx], preds[idx], zero_division=0))
        samples["precision"].append(precision_score(y_true[idx], preds[idx], zero_division=0))
        samples["roc_auc"].append(roc_auc_score(y_true[idx], probas[idx]))
        samples["pr_auc"].append(average_precision_score(y_true[idx], probas[idx]))
    return {k: {"mean": float(np.mean(v)), "ci_lower": float(np.percentile(v, 2.5)),
                "ci_upper": float(np.percentile(v, 97.5))} for k, v in samples.items()}


def calibration_brier_score(y_test, probas) -> float:
    return brier_score_loss(y_test, probas)


def shap_feature_importance(calibrated_model, X_train, X_test, sample_size: int = 300) -> pd.DataFrame | None:
    """Mean |SHAP| per encoded feature, with the explainer matched to the model type."""
    import shap
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    pipeline = calibrated_model.estimator            # the fitted best pipeline train.py calibrated
    if not hasattr(pipeline, "named_steps"):
        print("[SKIP] SHAP: the selected model is an ensemble without a single classifier step.")
        return None

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline[:-1]                     # samplers are skipped at transform time
    names = list(pipeline.named_steps["encoding"].get_feature_names_out())
    background = shap.sample(preprocessor.transform(X_train), 200, random_state=RANDOM_STATE)
    X_sample = preprocessor.transform(X_test)[:sample_size]

    if isinstance(model, (XGBClassifier, LGBMClassifier, CatBoostClassifier, RandomForestClassifier, DecisionTreeClassifier)):
        values = shap.TreeExplainer(model)(X_sample)
    elif isinstance(model, LogisticRegression):
        values = shap.LinearExplainer(model, background)(X_sample)
    else:
        scorer = model.predict_proba if hasattr(model, "predict_proba") else model.decision_function
        explainer = shap.explainers.Permutation(
            lambda d: scorer(d)[:, 1] if hasattr(model, "predict_proba") else scorer(d),
            shap.maskers.Independent(background, max_samples=50))
        values = explainer(X_sample[:100], max_evals=2 * X_sample.shape[1] + 1, silent=True)

    shap_values = values.values[..., 1] if values.values.ndim == 3 else values.values
    return (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def permutation_feature_importance(calibrated_model, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """Shuffle each raw input column through the full calibrated pipeline; drop in PR-AUC."""
    result = permutation_importance(calibrated_model, X_test, y_test, scoring="average_precision",
                                    n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    return (pd.DataFrame({"Feature": X_test.columns, "Importance_Mean": result.importances_mean,
                          "Importance_Std": result.importances_std})
            .sort_values("Importance_Mean", ascending=False).reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the kidney (CKD) cascade on the test split.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--anomaly-model-path", default=str(DEFAULT_ANOMALY_MODEL_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--importance-out", default=None, help="Optional CSV for permutation importance.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    X, y = get_X_y(load_raw_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = joblib.load(args.model_path)
    gate = joblib.load(args.anomaly_model_path)
    threshold = load_threshold(args.metadata_path)

    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)

    metrics = core_metrics(y_test, preds, probas)
    print(f"=== Test metrics at the saved threshold ({threshold:.4f}) ===")
    for k, v in metrics.items():
        print(f"{k:16s}: {v:.4f}" if isinstance(v, float) else f"{k:16s}: {v}")

    brier = calibration_brier_score(y_test, probas)
    brier_reference = calibration_brier_score(y_test, np.full(len(y_test), y_train.mean()))
    print(f"\nBrier score: {brier:.4f} (always-predict-prevalence reference: {brier_reference:.4f})")

    flagged = gate.predict(X_test) == -1
    gate_rates = {"overall": float(flagged.mean()),
                  "ckd": float(flagged[np.asarray(y_test) == 1].mean()),
                  "no_ckd": float(flagged[np.asarray(y_test) == 0].mean())}
    print(f"Stage 1 gate rejection rate: {gate_rates}")

    ci = bootstrap_confidence_intervals(y_test, preds, probas, args.bootstrap_iterations)
    print("\n=== 95% bootstrapped confidence intervals ===")
    for k, v in ci.items():
        print(f"{k:10s}: {v['mean']:.3f} ({v['ci_lower']:.3f} - {v['ci_upper']:.3f})")

    shap_df = shap_feature_importance(model, X_train, X_test)
    if shap_df is not None:
        print("\n=== Top 10 SHAP features ===")
        print(shap_df.head(10).to_string(index=False))

    perm_df = permutation_feature_importance(model, X_test, y_test)
    print("\n=== Top 10 permutation importance (raw inputs) ===")
    print(perm_df.head(10).to_string(index=False))
    if args.importance_out:
        Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
        perm_df.to_csv(args.importance_out, index=False)
        print(f"[SAVED] {args.importance_out}")

    metadata_path = Path(args.metadata_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"test_metrics_at_threshold": metrics, "brier_score": brier,
                     "bootstrap_confidence_intervals": ci, "anomaly_gate_rejection_rate": gate_rates})
    metadata_path.write_text(json.dumps(metadata, indent=2, default=float))
    print(f"\n[SAVED] Evaluation results -> {metadata_path}")


if __name__ == "__main__":
    main()
