"""
models.py — The 17 candidate classifiers and their grids, matching
notebooks/Kidney Disease.ipynb.

    CPU (default)  : 9 baselines + CatBoost + Explainable Boosting Machine
    Deep (opt-in)  : TabNet, FT-Transformer, RealMLP, TabM, TabPFN — need
                     torch + pytorch-tabnet + pytabkit + skorch + tabpfn and
                     are imported only when requested (train.py --include-deep)
    Stacking       : built in train.py from the best fitted CPU models

Every grid includes `smote: [SMOTENC, 'passthrough']`, so cross-validation
decides per model whether oversampling helps.
"""

import contextlib
import io

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split

from .data import RANDOM_STATE, build_smote

# Simplest -> most complex (interpretability and deployment cost), for the one-SE rule
COMPLEXITY_ORDER = [
    "Logistic Regression", "Naive Bayes", "Decision Tree", "Explainable Boosting", "KNN", "SVM",
    "Random Forest", "LightGBM", "XGBoost", "CatBoost", "Neural Network (MLP)", "TabNet",
    "RealMLP", "TabM", "FT-Transformer", "TabPFN", "Stacking Ensemble",
]
DEEP_MODELS = ["TabNet", "FT-Transformer", "RealMLP", "TabM", "TabPFN"]

# The notebook's one-SE pick; `python -m src.train` re-trains this one by default
SELECTED_MODEL = "LightGBM"


def smote_options():
    return [build_smote(), "passthrough"]


def cpu_model_configs() -> dict:
    """name -> (estimator, param_grid, grid n_jobs or None for all cores, oversample)."""
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    s = smote_options
    return {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE),
            {"smote": s(), "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0], "classifier__l1_ratio": [0.0, 1.0]},
            None, True),
        "Naive Bayes": (
            GaussianNB(),
            {"smote": s(), "classifier__var_smoothing": np.logspace(0, -11, num=50)},
            None, True),
        "KNN": (
            KNeighborsClassifier(),
            {"smote": s(), "classifier__n_neighbors": [5, 11, 21, 51], "classifier__weights": ["uniform", "distance"]},
            None, True),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"smote": s(), "classifier__max_depth": [3, 5, 8, 12], "classifier__min_samples_leaf": [5, 20, 50],
             "classifier__criterion": ["gini", "entropy"]},
            None, True),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"smote": s(), "classifier__n_estimators": [100, 200, 300], "classifier__max_depth": [10, 20, 30, None],
             "classifier__max_features": ["sqrt", "log2"]},
            None, True),
        "XGBoost": (
            XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
            {"smote": s(), "classifier__n_estimators": [100, 300, 500], "classifier__learning_rate": [0.01, 0.05, 0.1],
             "classifier__max_depth": [3, 5, 7], "classifier__subsample": [0.8, 1.0]},
            None, True),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
            {"smote": s(), "classifier__n_estimators": [100, 200, 500], "classifier__learning_rate": [0.03, 0.05, 0.1],
             "classifier__num_leaves": [15, 31]},
            None, True),
        "SVM": (
            SVC(kernel="rbf", random_state=RANDOM_STATE),   # decision_function is enough for PR-AUC
            {"smote": s(), "classifier__C": [0.1, 1.0, 10.0, 50.0], "classifier__gamma": ["scale", "auto"]},
            None, True),
        "Neural Network (MLP)": (
            MLPClassifier(max_iter=1000, early_stopping=True, random_state=RANDOM_STATE),
            {"smote": s(), "classifier__hidden_layer_sizes": [(100,), (100, 50), (128, 64, 32)],
             "classifier__alpha": [0.0001, 0.001, 0.01], "classifier__learning_rate_init": [0.001, 0.01]},
            None, True),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=1, allow_writing_files=False),
            {"smote": s(), "classifier__iterations": [500], "classifier__depth": [4, 6, 8],
             "classifier__learning_rate": [0.03, 0.1]},
            None, True),
        "Explainable Boosting": (
            ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=3),
            {"smote": s()},
            5, True),
    }


# ---------- deep / foundation models (GPU) ----------

def silenced():
    """Swallow console output from libraries that print every epoch."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


class QuietClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a pytabkit deep model: float32 input, no per-epoch logging."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        with silenced():
            self.estimator_ = clone(self.estimator).fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        with silenced():
            return self.estimator_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabNetSK(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper for TabNet that holds out 10 % of the fold for early stopping."""

    def __init__(self, n_d=8, n_steps=3, gamma=1.3, max_epochs=100, patience=10,
                 batch_size=512, random_state=RANDOM_STATE, device="auto"):
        self.n_d = n_d
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier

        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.1, stratify=y,
                                                      random_state=self.random_state)
        self.model_ = TabNetClassifier(n_d=self.n_d, n_a=self.n_d, n_steps=self.n_steps, gamma=self.gamma,
                                       seed=self.random_state, device_name=self.device, verbose=0)
        with silenced():
            self.model_.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric=["auc"],
                            max_epochs=self.max_epochs, patience=self.patience,
                            batch_size=self.batch_size, virtual_batch_size=128)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def deep_model_configs() -> dict:
    """GPU models, one fit at a time. TabPFN gets no SMOTE (synthetic rows distort its in-context posterior)."""
    import logging

    import torch
    from pytabkit import FTT_D_Classifier, RealMLP_TD_Classifier, TabM_D_Classifier
    from tabpfn import TabPFNClassifier

    for noisy_logger in ("lightning", "lightning.pytorch", "pytorch_lightning", "lightning_fabric"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = smote_options

    return {
        "TabNet": (TabNetSK(device=device), {"smote": s(), "classifier__n_d": [8, 16]}, 1, True),
        "FT-Transformer": (QuietClassifier(FTT_D_Classifier(device=device, random_state=RANDOM_STATE, verbosity=0)),
                           {"smote": s()}, 1, True),
        "RealMLP": (QuietClassifier(RealMLP_TD_Classifier(device=device, random_state=RANDOM_STATE, verbosity=0)),
                    {"smote": s()}, 1, True),
        "TabM": (QuietClassifier(TabM_D_Classifier(device=device, random_state=RANDOM_STATE, verbosity=0)),
                 {"smote": s()}, 1, True),
        "TabPFN": (TabPFNClassifier(device=device, random_state=RANDOM_STATE, ignore_pretraining_limits=True),
                   {}, 1, False),
    }
