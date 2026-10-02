"""
prediction.py — The result every module's `predict()` returns, so the six models
(heart / kidney / lung x tabular / image) can be called and compared the same way.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import numpy as np

# Outcome of the module's safety gates
ACCEPTED = "accepted"   # passed every gate: the result can be reported as is
REVIEW = "review"       # a clinician must look: low image confidence, or an unusual patient profile the model did not score
REJECTED = "rejected"   # not an input this model can score (e.g. an image that is not a chest X-ray)


@dataclass
class Prediction:
    """
    organ        "heart", "kidney" or "lung"; "triage" for the layer-1 symptom router
    modality     "tabular" or "image"
    status       ACCEPTED, REVIEW or REJECTED
    label        tabular: risk band ("Low Risk" / "Medium Risk" / "High Risk");
                 image: predicted class ("Normal", "Tuberculosis", ...);
                 triage: the route ("Heart", "Heart + Lung", "Other"). None when not scored.
    probability  probability of disease on one scale for all six organ models:
                 tabular P(disease), image 1 - P(Normal); triage: the highest organ
                 probability (all three are in details). None when not scored.
    positive     the screen is positive: refer / follow up. Tabular: probability >= the
                 cost-optimal threshold; image: the predicted class is not "Normal";
                 triage: at least one organ module should run.
    explanation  tabular: the inputs that moved this patient's probability most;
                 image: Grad-CAM heatmap and overlay. Empty when not scored or not requested.
    message      one line for the user: the finding and the next step
    details      module-specific extras (threshold, class probabilities, top-3, model name)
    """

    organ: str
    modality: str
    status: str
    label: Optional[str] = None
    probability: Optional[float] = None
    positive: Optional[bool] = None
    explanation: dict = field(default_factory=dict)
    message: str = ""
    details: dict = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        """True when the model produced a probability, i.e. the input passed the gate."""
        return self.probability is not None

    def to_dict(self, arrays: bool = True) -> dict:
        """
        Plain Python types. arrays=False drops numpy arrays (heatmaps, overlays), so the
        result can go straight to json.dumps.
        """
        return {f.name: _plain(getattr(self, f.name), arrays) for f in fields(self)}


def _plain(value: Any, arrays: bool) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v, arrays) for k, v in value.items()
                if arrays or not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [_plain(v, arrays) for v in value]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value
