"""
train.py — Trains the kidney (CKD) cascade artifacts, mirroring
notebooks/Kidney Disease.ipynb:

    Stage 1 (Anomaly Gate):   models/kidney_anomaly_gate.joblib
    Stage 2 (Classification): models/kidney_disease_calibrated_model.joblib
    Threshold + schema:       models/metadata.json
    Typical patient:          models/reference_profile.json   (what predict.py's explanations compare with)

Default: re-train the model the notebook selected (src.models.SELECTED_MODEL)
with its grid. With --compare, re-run the whole comparison and pick the model
with the one-standard-error rule on 5-fold CV PR-AUC.

CLI (run from models/kidney/tabular/):
    python -m src.train
    python -m src.train --compare                          # 11 CPU models + stacking
    python -m src.train --compare --include-deep           # + TabNet, FT-Transformer, RealMLP, TabM, TabPFN (GPU)
    python -m src.train --compare --results-out results/model_comparison.csv
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, StackingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from common.tabular import REFERENCE_FILE, save_reference_profile

from .data import (CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, LABEL_DEFINING_FEATURES, MODULE_DIR,
                   NUMERICAL_FEATURES, RANDOM_STATE, RAW_INPUT_COLUMNS, add_custom_features,
                   build_pipeline, get_X_y, load_raw_data, split_data)
from .models import COMPLEXITY_ORDER, DEEP_MODELS, SELECTED_MODEL, cpu_model_configs, deep_model_configs

CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Each parallel worker holds ~0.7 GB of RAM; keep ~2.5 GB for the main process and the OS
N_JOBS = int(max(1, min(8, (os.cpu_count() or 2) - 2, (psutil.virtual_memory().available / 1024**3 - 2.5) // 0.7)))
FN_COST, FP_COST = 5, 1                   # a missed CKD case costs 5x a false alarm
ANOMALY_CONTAMINATION = 0.01

MODEL_DIR = MODULE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "kidney_disease_calibrated_model.joblib"
DEFAULT_GATE_PATH = MODEL_DIR / "kidney_anomaly_gate.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"


def run_grid_search(name, estimator, param_grid, X_train, y_train, n_jobs=None, oversample=True):
    """Tune on 5-fold CV PR-AUC. Returns (best estimator, CV summary)."""
    start = time.time()
    pipeline = estimator if isinstance(estimator, StackingClassifier) else build_pipeline(estimator, oversample)
    grid = GridSearchCV(pipeline, param_grid, cv=CV_STRATEGY, scoring="average_precision", n_jobs=n_jobs or N_JOBS)
    grid.fit(X_train, y_train)

    folds = np.array([grid.cv_results_[f"split{k}_test_score"][grid.best_index_]
                      for k in range(CV_STRATEGY.get_n_splits())])
    steps = getattr(grid.best_estimator_, "named_steps", {})
    smote = "per base model" if "smote" not in steps else ("on" if steps["smote"] != "passthrough" else "off")
    summary = {"CV PR-AUC": folds.mean(), "CV SE": folds.std(ddof=1) / np.sqrt(len(folds)),
               "SMOTE": smote, "Tuning Time (min)": (time.time() - start) / 60}
    print(f"[{name}] CV PR-AUC {summary['CV PR-AUC']:.4f} ± {summary['CV SE']:.4f} | SMOTE {smote} | "
          f"{summary['Tuning Time (min)']:.1f} min")
    return grid.best_estimator_, summary


def positive_scores(model, X):
    """Probability of CKD, or the decision function for models without predict_proba (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compare_models(fitted: dict, cv_summary: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in fitted.items():
        scores, preds = positive_scores(model, X_test), model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        rows.append({"Model": name, **{k: cv_summary[name][k] for k in ("CV PR-AUC", "CV SE")},
                     "Test PR-AUC": average_precision_score(y_test, scores),
                     "Test ROC-AUC": roc_auc_score(y_test, scores),
                     "Accuracy": accuracy_score(y_test, preds),
                     "Precision": precision_score(y_test, preds, zero_division=0),
                     "Recall": recall_score(y_test, preds, zero_division=0),
                     "F1-Score": f1_score(y_test, preds, zero_division=0),
                     "Missed Patients (FN)": int(fn), "False Alarms (FP)": int(fp),
                     "SMOTE": cv_summary[name]["SMOTE"], "Tuning Time (min)": cv_summary[name]["Tuning Time (min)"]})
    return pd.DataFrame(rows).set_index("Model").sort_values("CV PR-AUC", ascending=False)


def select_one_se(comparison: pd.DataFrame) -> tuple[str, list]:
    """Simplest model whose CV PR-AUC is within one standard error of the best."""
    top = comparison["CV PR-AUC"].idxmax()
    cutoff = comparison.loc[top, "CV PR-AUC"] - comparison.loc[top, "CV SE"]
    within = [m for m in COMPLEXITY_ORDER if m in comparison.index and comparison.loc[m, "CV PR-AUC"] >= cutoff]
    return within[0], within


def calibrate(best_pipeline, X_train, y_train):
    """Isotonic calibration on held-out folds of the original (un-oversampled) data."""
    return CalibratedClassifierCV(estimator=best_pipeline, method="isotonic", cv=3).fit(X_train, y_train)


def oof_threshold(best_pipeline, X_train, y_train, n_jobs=5) -> float:
    """Cost-optimal threshold (5·FN + FP) on out-of-fold calibrated training predictions."""
    oof = cross_val_predict(CalibratedClassifierCV(estimator=clone(best_pipeline), method="isotonic", cv=3),
                            X_train, y_train, cv=CV_STRATEGY, method="predict_proba", n_jobs=n_jobs)[:, 1]
    positives = np.asarray(y_train) == 1
    candidates = np.unique(oof)
    cost = [FN_COST * np.sum(positives & (oof < t)) + FP_COST * np.sum(~positives & (oof >= t)) for t in candidates]
    return float(candidates[int(np.argmin(cost))])


def build_anomaly_gate() -> Pipeline:
    """Stage 1: IsolationForest on the scaled, imputed numeric features."""
    return Pipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("select", ColumnTransformer([("numeric", "passthrough", NUMERICAL_FEATURES)])),
        ("scaler", RobustScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
        ("detector", IsolationForest(n_estimators=200, contamination=ANOMALY_CONTAMINATION,
                                     random_state=RANDOM_STATE)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train the kidney (CKD) cascade artifacts.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Raw CKD_NHANES.csv export.")
    parser.add_argument("--compare", action="store_true", help="Re-run the model comparison + one-SE selection.")
    parser.add_argument("--include-deep", action="store_true", help="Add the GPU deep / foundation models.")
    parser.add_argument("--model", default=SELECTED_MODEL, help="Model to train when not comparing.")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--anomaly-model-out", default=str(DEFAULT_GATE_PATH))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--results-out", default=None, help="Optional CSV for the comparison table.")
    args = parser.parse_args()

    df = load_raw_data(args.data_path)
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"{len(df):,} adults | train {len(X_train):,} | test {len(X_test):,} | CKD prevalence {y.mean():.1%}")

    configs = cpu_model_configs()
    if args.include_deep or args.model in DEEP_MODELS:
        configs.update(deep_model_configs())
    names = list(configs) if args.compare else [args.model]

    fitted, cv_summary = {}, {}
    for name in names:
        estimator, grid, n_jobs, oversample = configs[name]
        fitted[name], cv_summary[name] = run_grid_search(name, estimator, grid, X_train, y_train, n_jobs, oversample)

    if args.compare:
        members = sorted((m for m in fitted if m not in DEEP_MODELS),
                         key=lambda m: cv_summary[m]["CV PR-AUC"], reverse=True)[:4]
        stacking = StackingClassifier(
            estimators=[(m.lower().replace(" ", "_"), fitted[m]) for m in members],
            final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1)
        fitted["Stacking Ensemble"], cv_summary["Stacking Ensemble"] = run_grid_search(
            "Stacking Ensemble", stacking, {}, X_train, y_train, n_jobs=5)

    comparison = compare_models(fitted, cv_summary, X_test, y_test)
    print("\n=== Model Comparison (test set; selection uses CV PR-AUC) ===")
    print(comparison[["CV PR-AUC", "CV SE", "Test PR-AUC", "Test ROC-AUC", "SMOTE"]].round(4).to_string())
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.results_out)
        print(f"[SAVED] Model comparison -> {args.results_out}")

    best_name, within = select_one_se(comparison) if args.compare else (args.model, [args.model])
    print(f"\nSelected model: {best_name} (within one SE: {within})")

    best_pipeline = fitted[best_name]
    calibrated = calibrate(best_pipeline, X_train, y_train)
    threshold = oof_threshold(best_pipeline, X_train, y_train, n_jobs=1 if best_name in DEEP_MODELS else 5)
    gate = build_anomaly_gate().fit(X_train)

    calib_probas = calibrated.predict_proba(X_test)[:, 1]
    preds = (calib_probas >= threshold).astype(int)
    test_metrics = {"roc_auc": roc_auc_score(y_test, calib_probas),
                    "pr_auc": average_precision_score(y_test, calib_probas),
                    "recall_at_threshold": recall_score(y_test, preds),
                    "precision_at_threshold": precision_score(y_test, preds, zero_division=0)}
    print(f"Out-of-fold cost-optimal threshold: {threshold:.4f}")
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    for path, obj in ((args.model_out, calibrated), (args.anomaly_model_out, gate)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        print(f"[SAVED] {path}")

    # The typical patient that predict.py's explanations compare against
    reference_out = Path(args.model_out).parent / REFERENCE_FILE
    save_reference_profile(X_train, CATEGORICAL_FEATURES, reference_out,
                           f"training split (80 %) of data/{Path(args.data_path).name} (adults 20+ with a KDIGO label)")
    print(f"[SAVED] {reference_out}")

    metadata = {
        "best_model": best_name,
        "selection_rule": "one-standard-error rule on 5-fold CV PR-AUC",
        "target": "KDIGO: eGFR < 60 or urine ACR >= 30 (adults 20+)",
        "removed_label_defining_features": LABEL_DEFINING_FEATURES,
        "optimal_threshold": threshold,
        "cost_weights": {"false_negative": FN_COST, "false_positive": FP_COST},
        "risk_bands": {"low": f"< {threshold:.4f}", "medium": f"{threshold:.4f} - 0.50", "high": ">= 0.50"},
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_metrics": test_metrics,
        "training_rows": len(X_train), "test_rows": len(X_test), "prevalence": float(y.mean()),
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2, default=float))
    print(f"[SAVED] {args.metadata_out}")


if __name__ == "__main__":
    main()
