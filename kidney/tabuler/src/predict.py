"""
predict.py — Stable inference interface for the kidney disease tabular model.
This is the module Streamlit (and later the cascade orchestrator) import
directly. Feature engineering and preprocessing happen inside the saved
pipeline, so callers only ever need to supply raw clinical columns.
"""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "kidney_tabular_calibrated.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

REQUIRED_COLUMNS = [
    "age", "gender", "ethnicity", "education_level", "poverty_income_ratio",
    "bmi", "height_cm", "bp_systolic", "bp_diastolic", "serum_creatinine",
    "blood_urea_nitrogen", "albumin_serum", "phosphorus", "bicarbonate",
    "calcium", "uric_acid", "urine_creatinine", "urine_albumin",
    "albumin_creatinine_ratio", "diabetes_diagnosed", "insulin_use",
    "diabetes_pills", "ever_smoked", "current_smoker", "egfr",
]

_model_cache = None


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load (and cache) the calibrated model artifact."""
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(model_path)
    return _model_cache


