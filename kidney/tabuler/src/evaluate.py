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

def core_metrics(y_test, preds, probas) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": auc(*roc_curve(y_test, probas)[:2]),
        "pr_auc": average_precision_score(y_test, probas),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
 
 
def find_optimal_threshold(y_test, probas, fn_cost: float = 5, fp_cost: float = 1) -> float:
    """
    Clinical cost function: false negatives (missed CKD) are weighted
    `fn_cost`x more than false positives (false alarms). Returns the
    probability cutoff that minimizes total cost.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, probas)
    cost = fn_cost * (1 - recalls[:-1]) + fp_cost * (1 - precisions[:-1])
    best_idx = np.argmin(cost)
    return float(thresholds[best_idx])

def bootstrap_confidence_intervals(y_test, preds, probas, n_iterations: int = 1000) -> dict:
    """95% CI for recall, ROC-AUC, and PR-AUC via resampling with replacement."""
    y_test_array = np.asarray(y_test)
    n_size = len(y_test_array)
 
    recalls, roc_aucs, pr_aucs = [], [], []
    for i in range(n_iterations):
        idx = resample(np.arange(n_size), replace=True, n_samples=n_size, random_state=i)
        y_true_b, y_pred_b, y_proba_b = y_test_array[idx], preds[idx], probas[idx]
        if len(np.unique(y_true_b)) < 2:
            continue
        recalls.append(recall_score(y_true_b, y_pred_b, zero_division=0))
        roc_aucs.append(roc_auc_score(y_true_b, y_proba_b))
        pr_aucs.append(average_precision_score(y_true_b, y_proba_b))
 
    def ci(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
 
    return {"recall": ci(recalls), "roc_auc": ci(roc_aucs), "pr_auc": ci(pr_aucs)}
 
 
def calibration_brier_score(y_test, probas) -> float:
    return float(brier_score_loss(y_test, probas))
