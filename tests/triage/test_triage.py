"""The layer-1 symptom router, and the cascade it starts."""

import json

from common.cascade import run
from common.config import REPO_ROOT
from common.registry import ORGANS, route

from conftest import STATUSES


def test_route_chest_pain_patient():
    result = route({"age": 64, "sex": "Male", "symptoms": ["Chest pain", "Shortness of breath"]})
    assert result.organ == "triage" and result.status in STATUSES
    assert set(result.details["route"]) <= set(ORGANS)


def test_cascade_on_frontend_example():
    """The example patient the front end loads runs through triage and the routed organs' tabular models."""
    payload = json.loads((REPO_ROOT / "frontend" / "assets" / "cascade_patient.json").read_text())
    result = run(payload["patient"], tabular=payload["tabular"], also_check=payload.get("also_check", ()))
    assert result.status in STATUSES
    assert set(result.organs) <= set(ORGANS)
