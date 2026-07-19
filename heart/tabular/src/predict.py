"""
predict.py — Cascade inference interface for the heart disease tabular model.
This is the module Streamlit (and any other caller) imports directly.

Every patient passes through the cascade — there is no non-cascade path:

    Stage 1 (Anomaly Gate):  IsolationForest pipeline flags extreme / clinically
                              implausible profiles (e.g. data entry errors,
                              severe/rare compounding conditions). Flagged
                              patients are withheld from Stage 2 and routed to
                              manual clinical review.
    Stage 2 (Classification): Survivors of Stage 1 are scored by the calibrated
                              CatBoost model. The cost-optimal threshold
                              (from evaluate.py's metadata.json) decides the
                              binary prediction; a separate fixed risk band
                              (<0.20 / <0.50 / >=0.50) drives display.

Feature engineering and preprocessing happen inside the saved Stage 2 pipeline,
so callers only ever need to supply raw clinical columns.
"""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

from src.data import NUMERICAL_FEATURES, add_custom_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# NOTE: heart_disease_calibrated_catboost.joblib is what's actually committed
# today. If you rename it (e.g. to match train.py's default output), update
# this path to match.
DEFAULT_MODEL_PATH = MODEL_DIR / "heart_disease_calibrated_catboost.joblib"

# Does not exist yet — train.py needs to fit + joblib.dump this (see note below).
DEFAULT_ANOMALY_MODEL_PATH = MODEL_DIR / "heart_anomaly_pipeline.joblib"

DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

REQUIRED_COLUMNS = [
    "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]

_model_cache = None
_anomaly_model_cache = None


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load (and cache) the calibrated Stage 2 CatBoost artifact."""
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(model_path)
    return _model_cache


def load_anomaly_model(model_path: Union[str, Path] = DEFAULT_ANOMALY_MODEL_PATH):
    """Load (and cache) the Stage 1 anomaly-gate pipeline."""
    global _anomaly_model_cache
    if _anomaly_model_cache is None:
        _anomaly_model_cache = joblib.load(model_path)
    return _anomaly_model_cache


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


def predict(
    patients: pd.DataFrame,
    model=None,
    anomaly_model=None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run the full cascade for one or more patients.

    Every row goes through Stage 1 first. Rows flagged as anomalies are
    withheld from Stage 2 entirely — their probability/prediction columns
    are left as None and risk_level reads "Anomaly - Manual Review", mirroring
    the notebook's cascade rejection behavior exactly.
    """
    _validate_columns(patients)
    model = model or load_model()
    anomaly_model = anomaly_model or load_anomaly_model()
    threshold = threshold if threshold is not None else load_threshold()

    # --- STAGE 1: Anomaly Gate ---
    # anomaly_model is a fitted Pipeline(imputer -> scaler -> IsolationForest)
    # expecting exactly NUMERICAL_FEATURES (6 raw + 3 engineered), in that
    # order — the same columns train.py's fit_anomaly_pipeline() used.
    # add_custom_features() is required here since `patients` only carries
    # raw clinical columns; the 3 engineered ones don't exist until computed.
    engineered = add_custom_features(patients)
    anomaly_labels = anomaly_model.predict(engineered[NUMERICAL_FEATURES])  # -1 = anomaly, 1 = normal
    is_anomaly = anomaly_labels == -1

    n = len(patients)
    probabilities = [None] * n
    predictions = [None] * n
    risk_levels = ["Anomaly - Manual Review"] * n

    # --- STAGE 2: Classification (survivors only) ---
    survivors = patients.loc[~is_anomaly]
    if len(survivors) > 0:
        survivor_probs = model.predict_proba(survivors)[:, 1]
        for pos, (idx, prob) in zip(
            [i for i, flagged in enumerate(is_anomaly) if not flagged],
            zip(survivors.index, survivor_probs),
        ):
            probabilities[pos] = float(prob)
            predictions[pos] = int(prob >= threshold)
            risk_levels[pos] = _risk_level(prob)

    return pd.DataFrame({
        "anomaly_flagged": is_anomaly,
        "heart_disease_probability": probabilities,
        "prediction": predictions,
        "risk_level": risk_levels,
    })


def predict_single(
    patient: dict,
    model=None,
    anomaly_model=None,
    threshold: Optional[float] = None,
) -> dict:
    """Convenience wrapper for a single patient dict — what a Streamlit form will call."""
    df = pd.DataFrame([patient])
    return predict(df, model=model, anomaly_model=anomaly_model, threshold=threshold).iloc[0].to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run heart disease risk prediction (cascade).")
    parser.add_argument("--input", required=True, help="JSON file: a single patient object or a list of them.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--anomaly-model-path", default=str(DEFAULT_ANOMALY_MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        payload = json.load(f)
    records = payload if isinstance(payload, list) else [payload]

    model = load_model(args.model_path)
    anomaly_model = load_anomaly_model(args.anomaly_model_path)
    output = predict(pd.DataFrame(records), model=model, anomaly_model=anomaly_model, threshold=args.threshold)
    print(output.to_string(index=False))
