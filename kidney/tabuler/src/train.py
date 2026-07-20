"""
train.py — Trains and compares 9 classifiers, calibrates the winner, and
saves the deployable artifact.

Adaptation note: the notebook used RAPIDS cuML / GPU XGBoost / GPU LightGBM
(Colab GPU runtime only). Those are swapped here for their CPU sklearn
equivalents so this runs anywhere. The notebook's "Decision Tree" was
actually a single-tree GPU XGBoost workaround (no cuML decision tree
exists) — replaced here with a real sklearn DecisionTreeClassifier and an
equivalent grid.

CLI:
    python -m src.train --data-path data/kidney_disease.csv --model-out models/kidney_tabular_calibrated.joblib
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data import RANDOM_STATE, build_pipeline, get_X_y, load_raw_data, split_data

CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

MODEL_CONFIGS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
        },
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"classifier__var_smoothing": np.logspace(0, -11, num=50)},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"classifier__n_neighbors": [3, 5, 7, 9, 11, 15]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [5, 10, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30, None],
            "classifier__max_features": ["sqrt", "log2"],
        },
    ),
    "XGBoost": (
        XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, eval_metric="logloss"),
        {
            "classifier__n_estimators": [100, 300, 500],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
        },
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__num_leaves": [31],
        },
    ),
    "SVM": (
        SVC(probability=True),
        {
            "classifier__C": [0.1, 1.0, 10.0, 50.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf"],
        },
    ),
    "Neural Network": (
        MLPClassifier(max_iter=1000, early_stopping=True, random_state=RANDOM_STATE),
        {
            "classifier__hidden_layer_sizes": [(100,), (100, 50), (128, 64, 32)],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate_init": [0.001, 0.01],
        },
    ),
}


