"""
models.py — The candidate classifiers and their grids, matching
notebooks/Symptom_Triage.ipynb. The same candidates are tuned for each organ head.

Nine CPU models. The other modules' KNN and SVM are left out (both compare every
visit with every training visit: hours on 180,000 visits), and so are their GPU
models and the stacking ensemble, whose cost here would be multiplied by three
heads. SMOTE is not used: each head has thousands of positive visits, and the
heart, kidney and lung grids switched it off for every model anyway.
"""

import numpy as np

from .data import RANDOM_STATE

# Simplest -> most complex (interpretability and deployment cost), for the one-SE rule
COMPLEXITY_ORDER = [
    "Logistic Regression", "Naive Bayes", "Decision Tree", "Explainable Boosting", "Random Forest",
    "LightGBM", "XGBoost", "CatBoost", "Neural Network (MLP)",
]

# The notebook's one-SE pick per head; `python -m src.train` re-trains these by default
SELECTED_MODELS = {"heart": "CatBoost", "lung": "Explainable Boosting", "kidney": "LightGBM"}


def model_configs() -> dict:
    """name -> (estimator, param_grid, grid n_jobs or None for the default)."""
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE),
            {"classifier__C": [0.01, 0.1, 1.0], "classifier__l1_ratio": [0.0, 1.0]},
            None),
        "Naive Bayes": (
            GaussianNB(),
            {"classifier__var_smoothing": np.logspace(0, -9, num=10)},
            None),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"classifier__max_depth": [6, 10, 16], "classifier__min_samples_leaf": [20, 100]},
            None),
        # Four outer bags instead of 14: each bag already sees ~118,000 visits (same test PR-AUC, 5x faster).
        # Its bags run in 4 processes, so the folds run one at a time
        "Explainable Boosting": (
            ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=4, outer_bags=4),
            {},
            1),
        # Leaves of at least 50 visits keep each forest under ~100 MB
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
            {"classifier__min_samples_leaf": [50, 100]},
            None),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
            {"classifier__n_estimators": [300, 600], "classifier__learning_rate": [0.03, 0.1],
             "classifier__num_leaves": [15, 31], "classifier__min_child_samples": [50]},
            None),
        "XGBoost": (
            XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
            {"classifier__n_estimators": [300, 600], "classifier__learning_rate": [0.05, 0.1],
             "classifier__max_depth": [4, 6], "classifier__subsample": [0.8]},
            None),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=1, allow_writing_files=False),
            {"classifier__iterations": [800], "classifier__depth": [6], "classifier__learning_rate": [0.05, 0.1]},
            None),
        "Neural Network (MLP)": (
            MLPClassifier(max_iter=200, early_stopping=True, random_state=RANDOM_STATE),
            {"classifier__hidden_layer_sizes": [(64,), (128, 64)], "classifier__alpha": [0.0001, 0.001]},
            None),
    }
