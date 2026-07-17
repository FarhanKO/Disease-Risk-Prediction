"""
train.py — Trains and compares 8 classifiers, calibrates the winner, and
saves the deployable artifact.

CLI:
    python -m src.train --data-path data/heart_disease.csv --model-out models/heart_tabular_calibrated.joblib
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data import RANDOM_STATE, build_pipeline, get_X_y, load_raw_data, split_data

CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# (estimator, param_grid) — grids ported verbatim from the notebook
MODEL_CONFIGS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 5, 7, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": [None, "sqrt", "log2"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [6, 8, 12, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__bootstrap": [True, False],
        },
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"classifier__var_smoothing": np.logspace(0, -9, num=20)},
    ),
    "Neural Network": (
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        {
            "classifier__hidden_layer_sizes": [(32, 16), (64, 32), (100,)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate": ["constant", "adaptive"],
        },
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__subsample": [0.8, 1.0],
        },
    ),
    "CatBoost": (
        CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
        {
            "classifier__iterations": [100, 200],
            "classifier__depth": [4, 6],
            "classifier__learning_rate": [0.01, 0.1],
        },
    ),
}


def run_grid_search(name, estimator, param_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=build_pipeline(estimator),
        param_grid=param_grid,
        cv=CV_STRATEGY,
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"[{name}] best CV recall: {grid.best_score_:.4f}")
    return grid


def compare_models(fitted_grids: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, grid in fitted_grids.items():
        model = grid.best_estimator_
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": auc(*roc_curve(y_test, probas)[:2]),
            "PR-AUC": average_precision_score(y_test, probas),
        })
    return pd.DataFrame(rows).set_index("Model")


def select_best(comparison_df: pd.DataFrame) -> str:
    """Recall first (clinical priority), precision as tiebreaker — matches notebook."""
    ranked = comparison_df.sort_values(by=["Recall", "Precision"], ascending=[False, False])
    return ranked.index[0]


def calibrate(best_pipeline, X_train, y_train):
    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate the heart disease tabular model.")
    parser.add_argument("--data-path", required=True, help="Path to the heart disease CSV.")
    parser.add_argument("--model-out", default="models/heart_tabular_calibrated.joblib")
    args = parser.parse_args()

    df = load_raw_data(args.data_path)
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    fitted_grids = {}
    for name, (estimator, param_grid) in MODEL_CONFIGS.items():
        fitted_grids[name] = run_grid_search(name, estimator, param_grid, X_train, y_train)

    comparison_df = compare_models(fitted_grids, X_test, y_test)
    print("\n=== Model Comparison (Test Set) ===")
    print(comparison_df.round(4))

    best_name = select_best(comparison_df)
    print(f"\nBest model: {best_name}")

    best_pipeline = fitted_grids[best_name].best_estimator_
    calibrated_model = calibrate(best_pipeline, X_train, y_train)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, out_path)
    print(f"\n[SAVED] Calibrated {best_name} model -> {out_path}")


if __name__ == "__main__":
    main()
