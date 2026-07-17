"""
predict.py — Stable inference interface for the heart disease tabular model.
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
DEFAULT_MODEL_PATH = MODEL_DIR / "heart_tabular_calibrated.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

REQUIRED_COLUMNS = [
    "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]

_model_cache = None


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load (and cache) the calibrated model artifact."""
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(model_path)
    return _model_cache


