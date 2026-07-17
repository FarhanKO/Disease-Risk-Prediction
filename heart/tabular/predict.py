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


def load_threshold(metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH, default: float = 0.5) -> float:
    """Read the clinically optimized decision threshold saved by evaluate.py."""
    path = Path(metadata_path)
    if path.exists():
        with open(path) as f:
            return json.load(f).get("optimal_threshold", default)
    return default


def _risk_level(probability: float) -> str:
    if probability < 0.20:
        return "Low Risk"
    elif probability < 0.50:
        return "Medium Risk"
    return "High Risk"


def _validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def predict(patients: pd.DataFrame, model=None, threshold: Optional[float] = None) -> pd.DataFrame:
    """Run inference for one or more patients, returning probability/prediction/risk band."""
    _validate_columns(patients)
    model = model or load_model()
    threshold = threshold if threshold is not None else load_threshold()

    probabilities = model.predict_proba(patients)[:, 1]
    return pd.DataFrame({
        "heart_disease_probability": probabilities,
        "prediction": (probabilities >= threshold).astype(int),
        "risk_level": [_risk_level(p) for p in probabilities],
    })


