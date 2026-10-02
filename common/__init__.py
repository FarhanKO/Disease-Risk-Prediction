"""
common — Code shared by the six disease modules (heart / kidney / lung x tabular / image)
and the symptom triage router that decides which of them a patient needs.

Every module's `src/predict.py` exposes the same two functions:

    predict(x)              -> Prediction   one patient (dict of raw inputs) or one image
    predict_batch(xs)       -> DataFrame    many at once, without explanations

`common.registry` reaches all seven from one place:

    from common.registry import predict, route
    route({"age": 64, "sex": "Male", "symptoms": ["Chest pain"]}).details["route"]   # e.g. ['heart']
    result = predict("lung", "image", "xray.png")
    result.status, result.label, result.probability, result.explanation

The tabular and image cascades (common.tabular, common.image) are imported only by the
modules that need them, so loading a tabular model never imports Keras.
"""

from .prediction import ACCEPTED, REJECTED, REVIEW, Prediction

__all__ = ["ACCEPTED", "REJECTED", "REVIEW", "Prediction"]
