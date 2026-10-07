"""
predict.py — Cascade inference for the kidney (CKD) tabular model, behind the interface
all six modules share (common.tabular.TabularModel). This is the module Streamlit and the
top-level cascade import.

Every patient passes through the cascade — there is no non-cascade path:

    Stage 1 (Anomaly Gate):   IsolationForest (1 % contamination, fitted on the
                              training set) withholds implausible profiles
                              (data-entry errors, rare compounding conditions)
                              and routes them to manual clinical review.
    Stage 2 (Classification): the calibrated best model scores the survivors.
                              The out-of-fold cost-optimal threshold from
                              models/metadata.json decides `positive` ("order
                              kidney tests"), and the risk bands are anchored to
                              it: Low < threshold <= Medium < 0.50 <= High.
    Explanation:              the inputs that moved this patient's probability most.

Inputs are raw clinical columns (see src.data.RAW_INPUT_COLUMNS) — no eGFR,
creatinine or urine albumin: those define CKD and are what a positive screen
should order next. Feature engineering and preprocessing happen inside the
saved pipelines.

    from src.predict import predict
    result = predict(patient)      # common.Prediction: status, label, probability, positive, explanation

CLI (from models/kidney/tabular/):
    python -m src.predict --input examples/patient.json
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd

from common import Prediction
from common.tabular import TabularModel, run_cli

from . import data

MODEL_DIR = data.MODULE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "kidney_disease_calibrated_model.joblib"
DEFAULT_ANOMALY_MODEL_PATH = MODEL_DIR / "kidney_anomaly_gate.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL = TabularModel(
    organ="kidney", disease="chronic kidney disease (KDIGO)", data_module=data,
    model_path=DEFAULT_MODEL_PATH, gate_path=DEFAULT_ANOMALY_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH,
    referral="order kidney tests (eGFR and urine albumin-to-creatinine ratio)",
)


def predict(patient: dict, explain: bool = True) -> Prediction:
    """Run the cascade for one patient: a dict of the raw input columns (NaN / None allowed)."""
    return MODEL.predict(patient, explain=explain)


def predict_batch(patients: pd.DataFrame) -> pd.DataFrame:
    """The cascade for many patients: status, label, probability, positive and message per row."""
    return MODEL.predict_batch(patients)


def load_threshold(metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH, default: float = 0.5) -> float:
    """The out-of-fold cost-optimal decision threshold written by train.py."""
    path = Path(metadata_path)
    if path.exists():
        return json.loads(path.read_text()).get("optimal_threshold", default)
    return default


if __name__ == "__main__":
    run_cli(MODEL, "Run kidney disease (CKD) risk prediction (cascade).")
