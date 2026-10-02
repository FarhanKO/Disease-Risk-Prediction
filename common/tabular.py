"""
tabular.py — The screening cascade shared by the heart, kidney and lung tabular modules.

Every patient passes through the whole cascade:

    Stage 1 (Anomaly Gate):   an Isolation Forest fitted on the training set withholds
                              implausible or extreme profiles (data-entry errors, rare
                              compounding conditions) for manual clinical review.
    Stage 2 (Classification): the calibrated model gives the probability of disease. The
                              cost-optimal threshold from models/metadata.json decides
                              referral, and the risk bands are anchored to it:
                              Low < threshold <= Medium < 0.50 <= High.
    Explanation:              which of this patient's inputs moved the probability most,
                              compared with a typical patient (models/reference_profile.json).

Feature engineering and preprocessing live inside the saved pipelines, so callers pass
only the module's raw input columns (`RAW_INPUT_COLUMNS` in its src/data.py).

Why a typical patient and not "unknown": the pipelines impute missing values from similar
patients, and imputed values are rarely whole numbers. Tree models learn to recognise
them (in the kidney data a missing education level goes with a sicker population), so
"what if this value were missing" measures the model's reaction to missingness, not to
the patient's value.
"""

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Union

import numpy as np
import pandas as pd

from .utils.artifacts import load_pipeline
from .prediction import ACCEPTED, REVIEW, Prediction

HIGH_RISK = 0.50
TOP_FACTORS = 5
REFERENCE_FILE = "reference_profile.json"
EXPLANATION_METHOD = ("what-if: effect = probability for this patient - probability with this one value "
                      "replaced by the typical training patient's (median, or most common answer)")
ANOMALY_MESSAGE = "Unusual profile: withheld by the anomaly gate for manual clinical review"


def risk_level(probability: float, threshold: float) -> str:
    if probability < threshold:
        return "Low Risk"
    if probability < HIGH_RISK:
        return "Medium Risk"
    return "High Risk"


def reference_profile(X_train: pd.DataFrame, categorical_columns: list) -> dict:
    """The typical training patient: the median of each number, the most common answer to each question."""
    return {c: X_train[c].mode().iloc[0] if c in categorical_columns else float(X_train[c].median())
            for c in X_train.columns}


def save_reference_profile(X_train: pd.DataFrame, categorical_columns: list, path: Union[str, Path], source: str):
    """Written next to the model by train.py; the explanations compare each patient with it."""
    profile = {"source": source, "values": reference_profile(X_train, categorical_columns)}
    Path(path).write_text(json.dumps(profile, indent=2))


class TabularModel:
    """One module's cascade: anomaly gate -> calibrated classifier -> risk band + explanation."""

    def __init__(self, organ: str, disease: str, data_module: ModuleType,
                 model_path: Union[str, Path], gate_path: Union[str, Path], metadata_path: Union[str, Path],
                 referral: str, routine: str = "Routine care"):
        self.organ = organ
        self.disease = disease
        self.data_module = data_module
        self.model_path, self.gate_path, self.metadata_path = Path(model_path), Path(gate_path), Path(metadata_path)
        self.reference_path = self.model_path.parent / REFERENCE_FILE
        self.referral, self.routine = referral, routine
        self.input_columns = list(data_module.RAW_INPUT_COLUMNS)
        self.categorical_columns = [c for c in data_module.CATEGORICAL_FEATURES if c in self.input_columns]
        self.metadata = self.reference = None
        self._model = self._gate = None
        self._lock = threading.Lock()

    def load(self) -> "TabularModel":
        """Load the calibrated model, the anomaly gate, metadata.json and the reference profile (once)."""
        with self._lock:
            if self._model is None:
                self.metadata = json.loads(self.metadata_path.read_text())
                if self.reference_path.exists():
                    self.reference = json.loads(self.reference_path.read_text())["values"]
                self._gate = load_pipeline(self.gate_path, self.data_module)
                self._model = load_pipeline(self.model_path, self.data_module)
        return self

    @property
    def threshold(self) -> float:
        """Cost-optimal decision threshold, chosen on out-of-fold training predictions."""
        return float(self.load().metadata["optimal_threshold"])

    def frame(self, patients) -> pd.DataFrame:
        """
        One patient (dict / Series) or many (DataFrame) -> the input columns, in order.
        Numbers become floats and None becomes NaN; the pipelines impute missing values.
        """
        if isinstance(patients, Mapping):
            patients = pd.DataFrame([patients])
        elif isinstance(patients, pd.Series):
            patients = patients.to_frame().T
        missing = [c for c in self.input_columns if c not in patients.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = patients[self.input_columns].reset_index(drop=True)
        numeric = [c for c in self.input_columns if c not in self.categorical_columns]
        frame[numeric] = frame[numeric].apply(pd.to_numeric).astype(float)
        categorical = frame[self.categorical_columns].astype(object)
        frame[self.categorical_columns] = categorical.where(categorical.notna(), np.nan)
        return frame

    def _score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Stage 1 flags, then Stage 2 probabilities for the rows that passed (NaN where flagged)."""
        self.load()
        flagged = self._gate.predict(frame) == -1              # -1 = anomaly, 1 = normal
        probabilities = np.full(len(frame), np.nan)
        if (~flagged).any():
            probabilities[~flagged] = self._model.predict_proba(frame[~flagged])[:, 1]
        return flagged, probabilities

    def _outcome(self, flagged: bool, probability: float, threshold: float) -> tuple:
        """(status, label, positive, message) for one patient."""
        if flagged:
            return REVIEW, None, None, ANOMALY_MESSAGE
        label = risk_level(probability, threshold)
        positive = bool(probability >= threshold)
        return ACCEPTED, label, positive, f"{label} ({probability:.0%}): {self.referral if positive else self.routine}"

    def predict(self, patient, explain: bool = True) -> Prediction:
        """Run the cascade for one patient (a dict of raw input columns; NaN / None allowed)."""
        frame = self.frame(patient)
        if len(frame) != 1:
            raise ValueError("predict() scores one patient; use predict_batch() for several")
        flagged, probabilities = self._score(frame)
        threshold = self.threshold
        status, label, positive, message = self._outcome(flagged[0], probabilities[0], threshold)

        scored = not flagged[0]
        probability = float(probabilities[0]) if scored else None
        return Prediction(
            organ=self.organ, modality="tabular", status=status, label=label,
            probability=probability, positive=positive,
            explanation=self.explain(frame, probability) if scored and explain else {},
            message=message,
            details={"disease": self.disease, "model": self.metadata.get("best_model"),
                     "threshold": threshold, "risk_bands": self.metadata.get("risk_bands"),
                     "anomaly_flagged": bool(flagged[0])},
        )

    def predict_batch(self, patients: pd.DataFrame) -> pd.DataFrame:
        """The cascade for many patients at once: one row each, without explanations."""
        frame = self.frame(patients)
        if frame.empty:
            return pd.DataFrame(columns=["status", "label", "probability", "positive", "message"])
        flagged, probabilities = self._score(frame)
        threshold = self.threshold
        status, label, positive, message = zip(*(self._outcome(f, p, threshold)
                                                 for f, p in zip(flagged, probabilities)))
        return pd.DataFrame({
            "status": status,
            "label": label,
            "probability": probabilities,
            "positive": pd.array(positive, dtype="boolean"),
            "message": message,
        }, index=patients.index if isinstance(patients, pd.DataFrame) else None)

    def explain(self, frame: pd.DataFrame, probability: float, top: int = TOP_FACTORS) -> dict:
        """
        The inputs that moved this patient's probability most. Each value that differs from
        the typical training patient is swapped for the typical one, one at a time; a positive
        effect means the patient's actual value raised the risk.
        """
        if self.reference is None:
            return {"method": EXPLANATION_METHOD, "factors": [],
                    "note": f"No {self.reference_path.name}: run `python -m src.train` to write it"}
        differing = [c for c in self.input_columns
                     if pd.notna(frame.at[0, c]) and frame.at[0, c] != self.reference[c]]
        if not differing:
            return {"method": EXPLANATION_METHOD, "factors": []}

        variants = pd.concat([frame] * len(differing), ignore_index=True)
        for i, column in enumerate(differing):
            variants.at[i, column] = self.reference[column]
        effects = probability - self._model.predict_proba(variants)[:, 1]

        order = np.argsort(-np.abs(effects))[:top]
        factors = [{"feature": differing[i], "value": frame.at[0, differing[i]],
                    "typical": self.reference[differing[i]], "effect": float(effects[i])}
                   for i in order if effects[i] != 0]
        return {"method": EXPLANATION_METHOD, "factors": factors}


def run_cli(model: TabularModel, description: str, argv=None):
    """`python -m src.predict --input patient.json`: one patient object, or a list of them."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="JSON file: a single patient object or a list of them.")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text())
    if isinstance(payload, list):
        print(model.predict_batch(pd.DataFrame(payload)).to_string())
    else:
        print(json.dumps(model.predict(payload).to_dict(), indent=2, default=str))
