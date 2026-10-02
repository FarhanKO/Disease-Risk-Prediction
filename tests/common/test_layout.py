"""The repo layout common.config describes is the one on disk, and every module still runs from its own folder."""

import subprocess
import sys

import pytest

from common.config import MODALITIES, MODELS_ROOT, ORGANS, REPO_ROOT, TRIAGE, module_dir, package
from common.registry import module

ALL_MODULES = [(organ, modality) for organ in ORGANS for modality in MODALITIES] + [(TRIAGE, "tabular")]


@pytest.mark.parametrize("organ, modality", ALL_MODULES)
def test_module_folders_exist(organ, modality):
    folder = module_dir(organ, modality)
    assert (folder / "src" / "predict.py").is_file()
    for sub in ("notebooks", "models", "results"):
        assert (folder / sub).is_dir(), f"{folder} has no {sub}/"


def test_package_paths():
    assert package("heart", "tabular") == "models.heart.tabular.src"
    assert package("lung", "image") == "models.lung.image_based.src"
    assert package(TRIAGE, "tabular") == "triage.tabular.src"
    assert module_dir("kidney", "image") == MODELS_ROOT / "kidney" / "image_based"
    assert module_dir(TRIAGE, "tabular") == REPO_ROOT / "triage" / "tabular"
    with pytest.raises(ValueError):
        package("brain", "tabular")
    with pytest.raises(ValueError):
        package(TRIAGE, "image")


@pytest.mark.parametrize("organ, modality", ALL_MODULES)
def test_predict_modules_import(organ, modality):
    """Importing a module's src.predict loads no weights, so this is cheap even for the image models."""
    predict_module = module(organ, modality)
    assert callable(predict_module.predict) and callable(predict_module.predict_batch)


@pytest.mark.parametrize("organ", list(ORGANS) + [TRIAGE])
def test_tabular_cli_from_module_folder(organ):
    """`python -m src.predict` from the module folder: src/__init__.py must still find common/ at the repo root."""
    folder = module_dir(organ, "tabular")
    result = subprocess.run([sys.executable, "-m", "src.predict", "--input", "examples/patient.json"],
                            cwd=folder, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert '"status"' in result.stdout
