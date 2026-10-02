"""
registry.py — One entry point for all seven models: the symptom triage router
(layer 1) and the six organ models it routes to.

    from common.registry import predict, route
    route({"age": 64, "sex": "Male", "symptoms": ["Chest pain"]})   # which organs to check
    predict("heart", "tabular", {"age": 67, ...})
    predict("kidney", "image", "ct_slice.png")
    cascade({"age": 64, ...}, tabular={"heart": {...}}, images={"heart": "ecg.png"})   # all of it, chained

Each module is imported under its full package path (models.heart.tabular.src.predict, ...),
so the seven `src` packages never collide, and only on first use.
"""

import importlib
from types import ModuleType

from .config import MODALITIES, ORGANS, TRIAGE, package
from .prediction import Prediction

__all__ = ["MODALITIES", "ORGANS", "TRIAGE", "cascade", "module", "predict", "predict_batch", "route"]


def module(organ: str, modality: str) -> ModuleType:
    """The `src.predict` module of one organ / modality, e.g. module("lung", "image") or module("triage", "tabular")."""
    return importlib.import_module(f"{package(organ, modality)}.predict")


def route(patient, **kwargs) -> Prediction:
    """
    Layer 1: which organ modules should see this patient, from age, sex and 1-3 symptoms.
    `result.details["route"]` lists the organs to run next (empty for "other").
    """
    return module(TRIAGE, "tabular").predict(patient, **kwargs)


def predict(organ: str, modality: str, x, **kwargs) -> Prediction:
    """Run one module's cascade: a patient dict for "tabular", an image for "image"."""
    return module(organ, modality).predict(x, **kwargs)


def predict_batch(organ: str, modality: str, xs, **kwargs):
    """Many patients (a DataFrame) or images (a list) through one module's cascade."""
    return module(organ, modality).predict_batch(xs, **kwargs)


def cascade(patient, **kwargs):
    """The whole system for one patient: triage, then each routed organ's tabular and image model (common.cascade.run)."""
    from .cascade import run

    return run(patient, **kwargs)
