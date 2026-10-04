"""
train.py — Trains the symptom-triage cascade artifacts, mirroring notebooks/Symptom_Triage.ipynb:

    Stage 1 (Anomaly Gate):   models/triage_anomaly_gate.joblib
    Stage 2 (Routing heads):  models/triage_{heart,lung,kidney}_calibrated_model.joblib
    Thresholds + schema:      models/metadata.json   (test metrics are added by evaluate.py)
    Typical patient:          models/reference_profile.json   (what predict.py's explanations compare with)

Steps for each organ head, as in the notebook:
    1. Tune on 5-fold CV PR-AUC of the training split with vital-sign dropout
       (every vital sign removed from a random half of the training visits).
    2. With --compare: the one-standard-error rule picks the simplest of the nine
       candidates within one SE of the best CV PR-AUC.
    3. Isotonic calibration (CalibratedClassifierCV, cv=3).
    4. Cost-based threshold (a missed organ = 5 x an unnecessary routing) on out-of-fold
       TRAINING predictions — the test set is never used here.

Default: re-train the model the notebook selected for each head (src.models.SELECTED_MODELS).

CLI (from triage/tabular/):
    python -m src.train
    python -m src.train --compare --n-jobs 4
    python -m src.train --model-dir /tmp/triage_models     # write somewhere else, e.g. to compare with the notebook's
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_predict

from .data import (CATEGORICAL_FEATURES, CV_STRATEGY, DEFAULT_DATA_PATH, GATE_FEATURES, MODULE_DIR,
                   NUMERICAL_FEATURES, ONE_HOT_FEATURES, ORGANS, RAW_INPUT_COLUMNS, RAW_NUMERICAL_FEATURES,
                   SYMPTOM_COLUMNS, SYMPTOMS, VITAL_DROPOUT, build_anomaly_gate, build_pipeline, drop_vital_signs,
                   get_X_y, keep_workers_light, load_raw_data, split_data)
from .models import COMPLEXITY_ORDER, SELECTED_MODELS, model_configs
from .routing import FN_COST, FP_COST, cost_optimal_threshold

MODEL_DIR = MODULE_DIR / "models"


def artifact_paths(model_dir=MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    return {"heads": {organ: model_dir / f"triage_{organ}_calibrated_model.joblib" for organ in ORGANS},
            "gate": model_dir / "triage_anomaly_gate.joblib",
            "metadata": model_dir / "metadata.json",
            "reference": model_dir / "reference_profile.json"}


PATHS = artifact_paths()
MODEL_PATHS, METADATA_PATH = PATHS["heads"], PATHS["metadata"]

# Each parallel worker holds ~0.6 GB of RAM (147,000 training visits); keep ~2 GB for the main process and the OS
DEFAULT_N_JOBS = int(max(1, min(8, (os.cpu_count() or 2) - 2, (psutil.virtual_memory().available / 1024**3 - 2.0) // 0.6)))

TARGETS = {
    "heart": "ischemic heart disease, arrhythmia, heart failure, other heart disease (first three ED diagnoses)",
    "lung": "influenza, pneumonia, lower respiratory infection, COPD, asthma, other lung disease, lung cancer, TB, COVID-19",
    "kidney": "kidney failure / CKD, kidney infection, stones and renal colic, other kidney disorders, kidney cancer, dialysis",
}


def tune(organ, name, estimator, param_grid, X_train, y_train, n_jobs):
    """Grid search on 5-fold CV PR-AUC; returns the refit pipeline and its fold mean / standard error."""
    start = time.time()
    grid = GridSearchCV(build_pipeline(estimator), param_grid, cv=CV_STRATEGY, scoring="average_precision", n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    folds = np.array([grid.cv_results_[f"split{k}_test_score"][grid.best_index_]
                      for k in range(CV_STRATEGY.get_n_splits())])
    se = folds.std(ddof=1) / np.sqrt(len(folds))
    params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
    print(f"[{organ}] {name}: CV PR-AUC {folds.mean():.4f} (± {se:.4f} SE) | {(time.time() - start) / 60:.1f} min | {params}")
    return grid.best_estimator_, folds.mean(), se


def select_one_se(cv_scores: dict) -> tuple[str, list]:
    """Simplest model whose CV PR-AUC is within one standard error of the best."""
    top = max(cv_scores, key=lambda m: cv_scores[m][0])
    cutoff = cv_scores[top][0] - cv_scores[top][1]
    within = [m for m in COMPLEXITY_ORDER if m in cv_scores and cv_scores[m][0] >= cutoff]
    return within[0], within


def oof_threshold(winner, X_train, y_train, n_jobs) -> float:
    """Cost-optimal threshold on out-of-fold calibrated training predictions."""
    oof = cross_val_predict(CalibratedClassifierCV(estimator=clone(winner), method="isotonic", cv=3),
                            X_train, y_train, cv=CV_STRATEGY, method="predict_proba", n_jobs=n_jobs)[:, 1]
    return cost_optimal_threshold(y_train, oof)


def reference_profile(X_train) -> dict:
    """The typical patient the explanations compare with: median measurements, most common answers, no symptom."""
    values = {c: float(X_train[c].median()) for c in RAW_NUMERICAL_FEATURES}
    values.update({c: X_train[c].mode().iloc[0] for c in ONE_HOT_FEATURES})
    values.update({c: None for c in SYMPTOM_COLUMNS})
    return values


def main():
    parser = argparse.ArgumentParser(description="Train the symptom-triage cascade (three calibrated heads + anomaly gate).")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--compare", action="store_true", help="tune all nine candidates per head and apply the one-SE rule")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="where to write the artifacts (default: models/)")
    args = parser.parse_args()
    paths = artifact_paths(args.model_dir)

    keep_workers_light()
    X, Y = get_X_y(load_raw_data(args.data_path))
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    X_train_aug = drop_vital_signs(X_train)
    print(f"Training visits: {len(X_train):,} (vital signs removed from {VITAL_DROPOUT:.0%}) | test visits held out: {len(X_test):,}")
    print("Prevalence: " + ", ".join(f"{o} {Y_train[o].mean():.2%}" for o in ORGANS))

    configs = model_configs()
    heads = {}
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    for organ in ORGANS:
        names = COMPLEXITY_ORDER if args.compare else [SELECTED_MODELS[organ]]
        fitted, cv_scores = {}, {}
        for name in names:
            estimator, grid, grid_jobs = configs[name]
            fitted[name], mean, se = tune(organ, name, estimator, grid, X_train_aug, Y_train[organ], grid_jobs or args.n_jobs)
            cv_scores[name] = (mean, se)
        best, within = select_one_se(cv_scores) if args.compare else (names[0], names)

        calib_jobs = 1 if best == "Explainable Boosting" else min(3, args.n_jobs)
        calibrated = CalibratedClassifierCV(estimator=fitted[best], method="isotonic", cv=3,
                                            n_jobs=calib_jobs).fit(X_train_aug, Y_train[organ])
        threshold = oof_threshold(fitted[best], X_train_aug, Y_train[organ],
                                  n_jobs=1 if best == "Explainable Boosting" else min(5, args.n_jobs))
        joblib.dump(calibrated, paths["heads"][organ])
        print(f"[{organ}] selected {best} (within one SE: {within}) | cost-optimal threshold {threshold:.4f}")
        print(f"[SAVED] {paths['heads'][organ]}")

        heads[organ] = {"best_model": best, "optimal_threshold": threshold,
                        "cv_pr_auc": {m: {"mean": s[0], "se": s[1]} for m, s in cv_scores.items()},
                        "within_one_se": within, "prevalence": float(Y[organ].mean())}

    # --- Stage 1: anomaly gate on the training visits as recorded (no dropout) ---
    gate = build_anomaly_gate().fit(X_train)
    joblib.dump(gate, paths["gate"])
    print(f"[SAVED] {paths['gate']}")

    paths["reference"].write_text(json.dumps({
        "source": f"training split (80 %) of data/{Path(args.data_path).name}, as recorded; symptoms: none",
        "values": reference_profile(X_train)}, indent=2))
    print(f"[SAVED] {paths['reference']}")

    metadata = {
        "dataset": "TRIAGE_NHAMCS.csv (CDC NHAMCS emergency-department visits 2010-2022, adults 18+ with a symptom and a diagnosis)",
        "task": "layer 1 router: which organ modules (heart, lung, kidney) should see a patient, from symptoms",
        "targets": TARGETS,
        "selection_rule": "one-standard-error rule on 5-fold CV PR-AUC, per head",
        "routing_rule": "route to every organ with probability >= its threshold; first route = highest "
                        "probability / threshold; none = other",
        "vital_dropout": VITAL_DROPOUT,
        "cost_weights": {"false_negative": FN_COST, "false_positive": FP_COST},
        "heads": heads,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "gate_features": GATE_FEATURES,
        "symptoms": SYMPTOMS,
        "training_rows": len(X_train), "test_rows": len(X_test),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=float))
    print(f"[SAVED] {paths['metadata']}  (run `python -m src.evaluate` to add test-set metrics)")


if __name__ == "__main__":
    main()
