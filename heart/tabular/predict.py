"""
predict.py — Stable inference interface for the heart disease tabular model.
This is the module Streamlit (and later the cascade orchestrator) import
directly. Feature engineering and preprocessing happen inside the saved
pipeline, so callers only ever need to supply raw clinical columns.
"""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

