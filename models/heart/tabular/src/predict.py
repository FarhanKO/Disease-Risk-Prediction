"""
predict.py — Cascade inference for the heart disease tabular model, behind the interface
all six modules share (common.tabular.TabularModel). This is the module Streamlit and the
top-level cascade import.

Every patient passes through the cascade — there is no non-cascade path:

    Stage 1 (Anomaly Gate):   an Isolation Forest (1 % contamination, fitted on the
                              training set) flags implausible or extreme profiles, such
                              as data-entry errors or rare compounding conditions.
                              Flagged patients are withheld from Stage 2 and routed to
                              manual clinical review.
    Stage 2 (Classification): survivors are scored by the calibrated model (Logistic
                              Regression in the current release). The cost-optimal
                              threshold from models/metadata.json decides referral, and
                              the risk bands are anchored to it: Low (< threshold),
                              Medium (threshold to 0.50), High (>= 0.50).
    Explanation:              the inputs that moved this patient's probability most.

Feature engineering and preprocessing happen inside the saved pipelines, so callers
supply only the 31 raw columns of HEART_NHANES.csv (see data/README.md).

    from src.predict import predict
    result = predict(patient)      # common.Prediction: status, label, probability, positive, explanation

CLI (from heart/tabular/):
    python -m src.predict --input examples/patient.json
"""

from pathlib import Path

import pandas as pd

from common import Prediction
from common.tabular import TabularModel, run_cli

from . import data

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "heart_disease_calibrated_model.joblib"
DEFAULT_ANOMALY_MODEL_PATH = MODEL_DIR / "heart_disease_anomaly_gate.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

REQUIRED_COLUMNS = data.RAW_INPUT_COLUMNS

MODEL = TabularModel(
    organ="heart", disease="heart disease", data_module=data,
    model_path=DEFAULT_MODEL_PATH, gate_path=DEFAULT_ANOMALY_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH,
    referral="refer for cardiology work-up (ECG, stress test)",
)


def predict(patient: dict, explain: bool = True) -> Prediction:
    """Run the cascade for one patient: a dict of the raw input columns (NaN / None allowed)."""
    return MODEL.predict(patient, explain=explain)


def predict_batch(patients: pd.DataFrame) -> pd.DataFrame:
    """The cascade for many patients: status, label, probability, positive and message per row."""
    return MODEL.predict_batch(patients)


if __name__ == "__main__":
    run_cli(MODEL, "Run heart disease risk prediction (cascade).")
