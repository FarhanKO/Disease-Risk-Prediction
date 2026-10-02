"""
config — Where the seven models live, and the names used to reach them.

    models/<organ>/<modality folder>/src/predict.py    the six organ models
    triage/tabular/src/predict.py                       the layer-1 symptom router
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"

ORGANS = ("heart", "kidney", "lung")
MODALITIES = ("tabular", "image")
TRIAGE = "triage"                  # layer 1: symptoms -> which organ modules to run (tabular only)
MODALITY_FOLDERS = {"tabular": "tabular", "image": "image_based"}


def package(organ: str, modality: str) -> str:
    """Import path of one module's `src` package, e.g. "models.lung.image_based.src" or "triage.tabular.src"."""
    if organ == TRIAGE and modality == "tabular":
        return f"{TRIAGE}.tabular.src"
    if organ not in ORGANS or modality not in MODALITIES:
        raise ValueError(f"Unknown model {organ!r} / {modality!r}: organs {ORGANS} + {TRIAGE!r} (tabular), "
                         f"modalities {MODALITIES}")
    return f"models.{organ}.{MODALITY_FOLDERS[modality]}.src"


def module_dir(organ: str, modality: str) -> Path:
    """Folder of one module (the one its CLI runs from), e.g. models/heart/tabular."""
    return REPO_ROOT.joinpath(*package(organ, modality).split(".")[:-1])
