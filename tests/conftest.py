"""Shared test setup: the repo root on sys.path, and helpers for the per-organ tests."""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import ACCEPTED, REJECTED, REVIEW  # noqa: E402
from common.config import module_dir  # noqa: E402

STATUSES = {ACCEPTED, REVIEW, REJECTED}

# The image networks need about 1.5 GB each; run them only when asked
run_image_tests = pytest.mark.skipif(os.environ.get("RUN_IMAGE_TESTS") != "1",
                                     reason="set RUN_IMAGE_TESTS=1 to load the image networks")


def example_patient(organ: str) -> dict:
    return json.loads((module_dir(organ, "tabular") / "examples" / "patient.json").read_text())


def first_test_image(organ: str):
    """One image from the module's data/test split, or skip when the dataset is not downloaded."""
    test_dir = module_dir(organ, "image") / "data" / "test"
    images = sorted(test_dir.glob("*/*.png")) + sorted(test_dir.glob("*/*.jpg")) if test_dir.is_dir() else []
    if not images:
        pytest.skip(f"no test images under {test_dir} (dataset not in git)")
    return images[0]
