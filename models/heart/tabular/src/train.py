"""
train.py — Reproduces the notebook's deployed model and saves both cascade artifacts:

    Stage 1 (Anomaly Gate):   models/heart_disease_anomaly_gate.joblib
    Stage 2 (Classification): models/heart_disease_calibrated_model.joblib
    Threshold + features:     models/metadata.json   (test metrics are added by evaluate.py)
    Typical patient:          models/reference_profile.json   (what predict.py's explanations compare with)

Steps, as in notebook/Heart_Diseases.ipynb:
    1. Tune each candidate on 5-fold CV PR-AUC; SMOTENC on/off is part of every grid.
    2. One-standard-error rule: the simplest candidate within one SE of the best CV PR-AUC.
    3. Isotonic calibration (CalibratedClassifierCV, cv=3).
    4. Cost-based threshold (missed case = 5 x false alarm) on out-of-fold TRAINING
       predictions — the test set is never used here.

The notebook benchmarks 17 models; its winner is Logistic Regression, the default here.
--compare adds the strongest CPU challengers from the notebook (GPU models are
benchmark-only and are not trained by this script).

CLI (from heart/tabular/):
    python -m src.train
    python -m src.train --compare --n-jobs 8
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

from common.tabular import REFERENCE_FILE, save_reference_profile

from .data import (CATEGORICAL_FEATURES, CV_STRATEGY, DEFAULT_DATA_PATH, NUMERICAL_FEATURES,
                   RANDOM_STATE, RAW_INPUT_COLUMNS, build_anomaly_gate, build_pipeline, get_X_y,
                   keep_workers_light, load_raw_data, split_data)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FN_COST, FP_COST = 5, 1
SMOTE_OPTIONS = [SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=RANDOM_STATE), "passthrough"]

# Simplest -> most complex; the one-SE rule picks the first candidate that clears the cutoff
COMPLEXITY_ORDER = ["Logistic Regression", "Explainable Boosting", "LightGBM", "XGBoost", "CatBoost"]


def candidate_configs(compare: bool) -> dict:
    """(estimator, param grid) per candidate — grids copied from the notebook."""
    configs = {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0], "classifier__l1_ratio": [0.0, 1.0]},
        ),
    }
    if compare:
        from catboost import CatBoostClassifier
        from interpret.glassbox import ExplainableBoostingClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        configs.update({
            "Explainable Boosting": (ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=3), {}),
            "LightGBM": (
                LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
                {"classifier__n_estimators": [200, 500], "classifier__learning_rate": [0.03, 0.1],
                 "classifier__num_leaves": [15, 31], "classifier__min_child_samples": [20, 100]},
            ),
            "XGBoost": (
                XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
                {"classifier__n_estimators": [100, 300, 500], "classifier__learning_rate": [0.01, 0.05, 0.1],
                 "classifier__max_depth": [3, 5, 7], "classifier__subsample": [0.8, 1.0]},
            ),
            "CatBoost": (
                CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=1, allow_writing_files=False),
                {"classifier__iterations": [500], "classifier__depth": [4, 6, 8],
                 "classifier__learning_rate": [0.03, 0.1]},
            ),
        })
    return configs


def tune(name, estimator, param_grid, X_train, y_train, n_jobs):
    """Grid search on 5-fold CV PR-AUC; returns the refit pipeline and its fold mean / standard error."""
    start = time.time()
    grid = GridSearchCV(build_pipeline(estimator), {"smote": SMOTE_OPTIONS, **param_grid},
                        cv=CV_STRATEGY, scoring="average_precision", n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    folds = np.array([grid.cv_results_[f"split{k}_test_score"][grid.best_index_]
                      for k in range(CV_STRATEGY.get_n_splits())])
    se = folds.std(ddof=1) / np.sqrt(len(folds))
    smote = "off" if grid.best_estimator_.named_steps["smote"] == "passthrough" else "on"
    print(f"[{name}] CV PR-AUC {folds.mean():.4f} (± {se:.4f} SE) | SMOTE {smote} | "
          f"{(time.time() - start) / 60:.1f} min | {grid.best_params_}")
    return grid.best_estimator_, folds.mean(), se


def select_one_se(cv_scores: dict) -> str:
    top = max(cv_scores, key=lambda m: cv_scores[m][0])
    cutoff = cv_scores[top][0] - cv_scores[top][1]
    return next(m for m in COMPLEXITY_ORDER if m in cv_scores and cv_scores[m][0] >= cutoff)


def cost_optimal_threshold(winner, X_train, y_train, n_jobs) -> float:
    """Threshold minimising 5·FN + 1·FP on out-of-fold calibrated training predictions."""
    oof = cross_val_predict(CalibratedClassifierCV(estimator=clone(winner), method="isotonic", cv=3),
                            X_train, y_train, cv=CV_STRATEGY, method="predict_proba", n_jobs=n_jobs)[:, 1]
    positives = y_train.values == 1
    candidates = np.unique(oof)
    cost = [FN_COST * np.sum(positives & (oof < t)) + FP_COST * np.sum(~positives & (oof >= t)) for t in candidates]
    return float(candidates[int(np.argmin(cost))])


def main():
    parser = argparse.ArgumentParser(description="Train the heart disease cascade (calibrated model + anomaly gate).")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--compare", action="store_true", help="also tune the CPU challengers from the notebook")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--model-out", default=str(MODEL_DIR / "heart_disease_calibrated_model.joblib"))
    parser.add_argument("--gate-out", default=str(MODEL_DIR / "heart_disease_anomaly_gate.joblib"))
    parser.add_argument("--metadata-out", default=str(MODEL_DIR / "metadata.json"))
    args = parser.parse_args()

    keep_workers_light()
    X, y = get_X_y(load_raw_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training rows: {len(X_train):,} ({y_train.mean():.1%} heart disease) | test rows held out: {len(X_test):,}")

    # --- Stage 2: tune, select, calibrate, threshold ---
    fitted, cv_scores = {}, {}
    for name, (estimator, grid) in candidate_configs(args.compare).items():
        fitted[name], mean, se = tune(name, estimator, grid, X_train, y_train, args.n_jobs)
        cv_scores[name] = (mean, se)

    best_name = select_one_se(cv_scores)
    print(f"\nSelected (one-SE rule): {best_name}")

    calibrated = CalibratedClassifierCV(estimator=fitted[best_name], method="isotonic", cv=3).fit(X_train, y_train)
    threshold = cost_optimal_threshold(fitted[best_name], X_train, y_train, n_jobs=min(5, args.n_jobs))
    print(f"Cost-optimal threshold (out-of-fold, training set): {threshold:.4f}")

    # --- Stage 1: anomaly gate on the same training split ---
    gate = build_anomaly_gate().fit(X_train)

    for path, artifact in [(args.model_out, calibrated), (args.gate_out, gate)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        print(f"[SAVED] {path}")

    # The typical patient that predict.py's explanations compare against
    reference_out = Path(args.model_out).parent / REFERENCE_FILE
    save_reference_profile(X_train, CATEGORICAL_FEATURES, reference_out,
                           f"training split (80 %) of data/{Path(args.data_path).name}")
    print(f"[SAVED] {reference_out}")

    metadata = {
        "dataset": "HEART_NHANES.csv (CDC NHANES 2007-March 2020, examined adults aged 40+)",
        "target": "heart_disease: doctor-diagnosed coronary heart disease, angina, heart attack or congestive heart failure",
        "best_model": best_name,
        "selection_rule": "one-standard-error rule on 5-fold CV PR-AUC among CPU-deployable models",
        "cv_pr_auc": {m: {"mean": s[0], "se": s[1]} for m, s in cv_scores.items()},
        "optimal_threshold": threshold,
        "cost_weights": {"false_negative": FN_COST, "false_positive": FP_COST},
        "risk_bands": {"low": f"< {threshold:.4f}", "medium": f"{threshold:.4f} - 0.50", "high": ">= 0.50"},
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_rows": len(X_train), "test_rows": len(X_test), "prevalence": float(y.mean()),
    }
    with open(args.metadata_out, "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    print(f"[SAVED] {args.metadata_out}  (run `python -m src.evaluate` to add test-set metrics)")


if __name__ == "__main__":
    main()
