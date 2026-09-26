"""Heart disease, image module (12-lead ECG). Run its commands from heart/image_based/, e.g. `python -m src.predict`."""

import sys
from pathlib import Path

# The shared prediction interface (common/) lives at the repo root
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
