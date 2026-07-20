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

