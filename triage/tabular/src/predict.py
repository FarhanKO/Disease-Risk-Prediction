"""
predict.py — Symptom triage, the first layer of the system: which organ modules
(heart, lung, kidney) should see a patient, from what the patient can tell. Behind
the interface all modules share (common.Prediction), with organ = "triage".

Every patient passes through the whole cascade:

    Stage 1 (Anomaly Gate):   an Isolation Forest on age and the vital signs (1 %
                              contamination, fitted on the training visits) withholds
                              impossible or extreme readings: typing errors, or a patient
                              who needs urgent care rather than an app.
    Stage 2 (Routing):        three calibrated heads give P(heart), P(lung) and P(kidney).
                              The patient is routed to every organ at or above its
                              cost-optimal threshold, first to the organ furthest above its
                              own threshold; no organ reached means "other".
    Explanation:              for each routed organ (or the closest one, for "other"), the
                              inputs that moved its probability most, compared with a
                              typical patient: no symptoms, typical measurements.

Only age, sex and one to three symptoms are needed. Diabetes, the injury question, the
vital signs and the pain score are optional: a blank is scored as typical. Symptom names
come from data/symptom_vocabulary.csv (`python -m src.predict --list-symptoms`).

    from src.predict import predict
    result = predict({"age": 64, "sex": "Male", "symptoms": ["Chest pain", "Shortness of breath"]})
    result.label                    # e.g. 'Heart', 'Heart + Lung' or 'Other'
    result.details["route"]         # e.g. ['heart', 'lung']: run these organ modules next
    result.details["probabilities"] # calibrated P(organ) per head

CLI (from triage/tabular/):
    python -m src.predict --input examples/patient.json
    python -m src.predict --list-symptoms
"""

import argparse
import difflib
import json
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from common import ACCEPTED, REVIEW, Prediction
from common.utils.artifacts import load_pipeline

from . import data
from .routing import OTHER, route_visits

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATHS = {organ: MODEL_DIR / f"triage_{organ}_calibrated_model.joblib" for organ in data.ORGANS}
DEFAULT_ANOMALY_MODEL_PATH = MODEL_DIR / "triage_anomaly_gate.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"
DEFAULT_REFERENCE_PATH = MODEL_DIR / "reference_profile.json"

REQUIRED_COLUMNS = ["age", "sex", "symptom_1"]
MAX_SYMPTOMS = len(data.SYMPTOM_COLUMNS)
TOP_FACTORS = 5
ORGAN_NAMES = {"heart": "Heart", "lung": "Lung", "kidney": "Kidney", OTHER: "Other"}
# What a routed patient meets next: that organ's tabular risk model, then its image model with a scan
NEXT_MODULES = {"heart": "heart disease risk (tabular) -> ECG (image)",
                "lung": "COPD risk (tabular) -> chest X-ray (image)",
                "kidney": "CKD risk (tabular) -> kidney CT (image)"}
EXPLANATION_METHOD = ("what-if: effect = this head's probability for the patient - its probability with this one input "
                      "replaced by a typical patient's (median measurement, most common answer, or no symptom)")
ANOMALY_MESSAGE = ("Vital signs outside the usual range: not routed, withheld for review. Check the values; "
                   "if they are correct, seek urgent medical care")


class TriageModel:
    """Anomaly gate -> three calibrated organ heads -> routing decision + explanation."""

    organ = "triage"

    def __init__(self, model_paths=None, gate_path=DEFAULT_ANOMALY_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH,
                 reference_path=DEFAULT_REFERENCE_PATH):
        self.model_paths = {organ: Path(p) for organ, p in (model_paths or DEFAULT_MODEL_PATHS).items()}
        self.gate_path, self.metadata_path, self.reference_path = Path(gate_path), Path(metadata_path), Path(reference_path)
        self.input_columns = list(data.RAW_INPUT_COLUMNS)
        self.categorical_columns = list(data.CATEGORICAL_FEATURES)
        self.metadata = self.reference = None
        self._heads = self._gate = None
        self._lock = threading.Lock()

    def load(self) -> "TriageModel":
        """Load the three heads, the gate, metadata.json and the reference profile (once)."""
        with self._lock:
            if self._heads is None:
                self.metadata = json.loads(self.metadata_path.read_text())
                if self.metadata.get("symptoms", data.SYMPTOMS) != data.SYMPTOMS:
                    raise RuntimeError("The saved heads were trained on another symptom vocabulary than "
                                       "data/symptom_vocabulary.csv; retrain them (python -m src.train).")
                if self.reference_path.exists():
                    self.reference = json.loads(self.reference_path.read_text())["values"]
                self._gate = load_pipeline(self.gate_path, data)
                self._heads = {organ: load_pipeline(path, data) for organ, path in self.model_paths.items()}
        return self

    @property
    def thresholds(self) -> dict:
        """Cost-optimal threshold of each head, chosen on out-of-fold training predictions."""
        heads = self.load().metadata["heads"]
        return {organ: float(heads[organ]["optimal_threshold"]) for organ in data.ORGANS}

    # ---------- input ----------

    def _one_row(self, patient: Mapping) -> dict:
        row = dict(patient)
        if "symptoms" in row:
            symptoms = [s for s in (row.pop("symptoms") or []) if s is not None]
            if len(symptoms) > MAX_SYMPTOMS:
                raise ValueError(f"At most {MAX_SYMPTOMS} symptoms, most important first (the router was trained on "
                                 f"the first three reasons for each visit); got {len(symptoms)}")
            row.update({column: symptoms[i] if i < len(symptoms) else np.nan
                        for i, column in enumerate(data.SYMPTOM_COLUMNS)})
        return row

    def frame(self, patients) -> pd.DataFrame:
        """
        One patient (dict / Series) or many (DataFrame or list of dicts) -> the input columns, in order.
        Symptoms may be given as symptom_1..3 or as a "symptoms" list. Optional columns may be left out;
        numbers become floats and None becomes NaN (scored as typical).
        """
        if isinstance(patients, pd.Series):
            patients = patients.to_dict()
        if isinstance(patients, Mapping):
            patients = [patients]
        if isinstance(patients, pd.DataFrame):
            patients = patients.to_dict(orient="records")
        frame = pd.DataFrame([self._one_row(p) for p in patients])
        frame = frame.reindex(columns=self.input_columns)

        missing = [c for c in REQUIRED_COLUMNS if frame[c].isna().any()]
        if missing:
            raise ValueError(f"Required inputs missing: {missing} (age, sex and at least one symptom)")
        numeric = [c for c in self.input_columns if c not in self.categorical_columns]
        frame[numeric] = frame[numeric].apply(pd.to_numeric).astype(float)
        categorical = frame[self.categorical_columns].astype(object)
        frame[self.categorical_columns] = categorical.where(categorical.notna(), np.nan)
        self._check_answers(frame)
        return frame

    @staticmethod
    def _check_answers(frame: pd.DataFrame):
        """Reject symptom names outside the vocabulary (with suggestions) and unknown answers."""
        given = pd.Series(frame[data.SYMPTOM_COLUMNS].to_numpy().ravel()).dropna()
        unknown = sorted(set(given) - set(data.SYMPTOMS))
        if unknown:
            hints = {s: difflib.get_close_matches(str(s), data.SYMPTOMS, n=3, cutoff=0.4) for s in unknown}
            raise ValueError("Unknown symptom(s): " + "; ".join(f"{s!r} (did you mean {h}?)" for s, h in hints.items())
                             + ". See data/symptom_vocabulary.csv or `python -m src.predict --list-symptoms`.")
        allowed = {"sex": {"Female", "Male"}, "injury_reason": {"Yes", "No"}, "diabetes": {"Yes", "No"}}
        for column, values in allowed.items():
            bad = set(frame[column].dropna()) - values
            if bad:
                blank = "" if column in REQUIRED_COLUMNS else " (or left out)"
                raise ValueError(f"{column} must be one of {sorted(values)}{blank}; got {sorted(bad)}")

    # ---------- scoring ----------

    def _score(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Stage 1 flags, then Stage 2 probabilities for the rows that passed (NaN where flagged)."""
        self.load()
        flagged = self._gate.predict(frame) == -1                  # -1 = anomaly, 1 = normal
        probabilities = pd.DataFrame(np.nan, index=frame.index, columns=data.ORGANS)
        if (~flagged).any():
            passed = frame[~flagged]
            for organ, head in self._heads.items():
                probabilities.loc[~flagged, organ] = head.predict_proba(passed)[:, 1]
        return flagged, probabilities

    def _decide(self, probabilities: pd.Series) -> tuple[list, str, str]:
        """(routes in order, label, message) for one patient who passed the gate."""
        thresholds = self.thresholds
        flags, _ = route_visits(probabilities.to_frame().T, thresholds)
        routes = sorted((o for o in data.ORGANS if flags.iloc[0][o]), key=lambda o: -probabilities[o] / thresholds[o])
        if routes:
            label = " + ".join(ORGAN_NAMES[o] for o in routes)
            message = ("Route to " + ", ".join(f"{ORGAN_NAMES[o]} ({probabilities[o]:.0%})" for o in routes)
                       + ": run " + ("its organ module" if len(routes) == 1 else "these organ modules") + " next")
        else:
            closest = max(data.ORGANS, key=lambda o: probabilities[o] / thresholds[o])
            label = ORGAN_NAMES[OTHER]
            message = (f"Other: no heart, lung or kidney signal (closest: {ORGAN_NAMES[closest]} "
                       f"{probabilities[closest]:.0%}, routed from {thresholds[closest]:.0%}). "
                       "General care; see a doctor if the symptoms persist or get worse")
        return routes, label, message

    def predict(self, patient, explain: bool = True) -> Prediction:
        """Route one patient: a dict with age, sex and symptoms (plus any optional inputs)."""
        frame = self.frame(patient)
        if len(frame) != 1:
            raise ValueError("predict() scores one patient; use predict_batch() for several")
        flagged, probabilities = self._score(frame)
        symptoms = [s for s in frame.loc[0, data.SYMPTOM_COLUMNS] if isinstance(s, str)]
        details = {"symptoms": symptoms, "thresholds": self.thresholds, "anomaly_flagged": bool(flagged[0]),
                   "models": {o: self.metadata["heads"][o]["best_model"] for o in data.ORGANS}}
        if flagged[0]:
            return Prediction(organ=self.organ, modality="tabular", status=REVIEW, message=ANOMALY_MESSAGE, details=details)

        p = probabilities.iloc[0]
        routes, label, message = self._decide(p)
        explained = routes or [max(data.ORGANS, key=lambda o: p[o] / self.thresholds[o])]
        details.update({"route": routes, "primary": routes[0] if routes else OTHER,
                        "probabilities": {o: float(p[o]) for o in data.ORGANS},
                        "next_modules": {o: NEXT_MODULES[o] for o in routes}})
        return Prediction(
            organ=self.organ, modality="tabular", status=ACCEPTED, label=label,
            probability=float(p.max()), positive=bool(routes),
            explanation=self.explain(frame, p, explained) if explain else {},
            message=message, details=details,
        )

    def predict_batch(self, patients) -> pd.DataFrame:
        """The cascade for many patients at once: one row each, without explanations."""
        frame = self.frame(patients)
        flagged, probabilities = self._score(frame)
        rows = []
        for i in range(len(frame)):
            if flagged[i]:
                rows.append({"status": REVIEW, "label": None, "route": None, "positive": None, "message": ANOMALY_MESSAGE})
                continue
            routes, label, message = self._decide(probabilities.iloc[i])
            rows.append({"status": ACCEPTED, "label": label, "route": "+".join(routes) or OTHER,
                         "positive": bool(routes), "message": message})
        out = pd.DataFrame(rows)
        for organ in data.ORGANS:
            out[f"p_{organ}"] = probabilities[organ].values
        out["positive"] = pd.array(out["positive"], dtype="boolean")
        if isinstance(patients, pd.DataFrame):
            out.index = patients.index
        return out[["status", "label", "route", "p_heart", "p_lung", "p_kidney", "positive", "message"]]

    # ---------- explanation ----------

    def explain(self, frame: pd.DataFrame, probabilities: pd.Series, organs: list, top: int = TOP_FACTORS) -> dict:
        """
        For each organ: the inputs that moved its head's probability most. Each input that differs from
        the typical patient is swapped for the typical value one at a time (a symptom is removed); a
        positive effect means the patient's actual input raised that organ's probability.
        """
        if self.reference is None:
            return {"method": EXPLANATION_METHOD, "factors": {},
                    "note": f"No {self.reference_path.name}: run `python -m src.train` to write it"}
        differing = [c for c in self.input_columns
                     if pd.notna(frame.at[0, c]) and frame.at[0, c] != self.reference.get(c)]
        factors = {}
        if differing:
            variants = pd.concat([frame] * len(differing), ignore_index=True)
            for i, column in enumerate(differing):
                variants.at[i, column] = np.nan if self.reference.get(column) is None else self.reference[column]
            for organ in organs:
                effects = probabilities[organ] - self._heads[organ].predict_proba(variants)[:, 1]
                order = np.argsort(-np.abs(effects))[:top]
                factors[organ] = [{"feature": differing[i], "value": frame.at[0, differing[i]],
                                   "typical": self.reference.get(differing[i]), "effect": float(effects[i])}
                                  for i in order if effects[i] != 0]
        return {"method": EXPLANATION_METHOD, "factors": factors}


MODEL = TriageModel()


def predict(patient: dict, explain: bool = True) -> Prediction:
    """Route one patient: age, sex and 1-3 symptoms (a "symptoms" list or symptom_1..3), plus optional inputs."""
    return MODEL.predict(patient, explain=explain)


def predict_batch(patients) -> pd.DataFrame:
    """The cascade for many patients: status, label, route, the three probabilities, positive and message."""
    return MODEL.predict_batch(patients)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Symptom triage: which organ modules should see a patient.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="JSON file: a single patient object or a list of them.")
    group.add_argument("--list-symptoms", action="store_true", help="Print the symptom names the router accepts.")
    args = parser.parse_args(argv)

    if args.list_symptoms:
        vocabulary = pd.read_csv(data.VOCABULARY_PATH)
        for system, names in vocabulary.groupby("body_system", sort=False)["symptom"]:
            print(f"{system}:\n  " + "\n  ".join(names))
        return
    payload = json.loads(Path(args.input).read_text())
    if isinstance(payload, list):
        print(predict_batch(payload).to_string())
    else:
        print(json.dumps(predict(payload).to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
