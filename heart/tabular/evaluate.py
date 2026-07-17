"""
evaluate.py — Post-training evaluation for the calibrated heart disease model:
core metrics, cost-based threshold optimization, calibration check (Brier score),
bootstrapped confidence intervals, SHAP and permutation feature importance.

CLI:
    python -m src.evaluate --data-path data/heart_disease.csv --model-path models/heart_tabular_calibrated.joblib
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


