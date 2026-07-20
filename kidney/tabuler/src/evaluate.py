"""
evaluate.py — Post-training evaluation for the calibrated kidney disease model:
core metrics, cost-based threshold optimization, calibration check (Brier score),
bootstrapped confidence intervals, SHAP and permutation feature importance.
 
Note: unlike the heart module (always CatBoost), the kidney winner can be
any of 9 model types, so SHAP falls back from TreeExplainer to a generic
Explainer when the winning model isn't tree-based.
 
CLI:
    python -m src.evaluate --data-path data/kidney_disease.csv --model-path models/kidney_tabular_calibrated.joblib
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

