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
        "ckd_probability": probabilities,
        "prediction": (probabilities >= threshold).astype(int),
        "risk_level": [_risk_level(p) for p in probabilities],
    })


def predict_single(patient: dict, model=None, threshold: Optional[float] = None) -> dict:
    """Convenience wrapper for a single patient dict — what a Streamlit form will call."""
    df = pd.DataFrame([patient])
    return predict(df, model=model, threshold=threshold).iloc[0].to_dict()


