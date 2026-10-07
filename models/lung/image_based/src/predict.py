"""
predict.py — Cascade inference for the lung chest X-ray module, behind the interface all
six modules share (common.image.ImageModel). This is the module Streamlit and the
top-level cascade import.

Every image passes through the cascade — there is no non-cascade path:

    Stage 1 (OOD Gate):        IsolationForest over DenseNet121 embeddings
                               rejects inputs that do not look like chest
                               X-rays (photos, CT slices, noise). Rejected
                               images never reach Stage 2.
    Stage 2 (Classification):  Fine-tuned DenseNet121 (or the soft-voting
                               ensemble) scores the six classes.
    Stage 3 (Confidence):      Top-class probability below the threshold
                               (0.75, from models/metadata.json) is flagged
                               for manual review.
    Stage 4 (Explainability):  Grad-CAM heatmap of the top class plus the
                               top-3 differential.

The checkpoints were trained with TensorFlow; common.image serves them on the PyTorch
backend when torch is installed, with the same test metrics.

    from src.predict import predict
    result = predict("xray.jpg")   # common.Prediction: status, label, probability, positive, explanation

CLI (from models/lung/image_based/):
    python -m src.predict --image xray1.jpg xray2.png --heatmap-dir out/
"""

import pandas as pd

from common import Prediction
from common.image import ImageModel, run_cli

from .data import CLASS_NAMES, load_image
from .models import MODEL_DIR, MODEL_FILES

CONFIDENCE_THRESHOLD = 0.75     # evaluate.py's default; predict() reads the saved value from metadata.json
DEFAULT_OOD_PATH = MODEL_DIR / "lung_ood_detector.joblib"

MODEL = ImageModel(
    organ="lung", model_dir=MODEL_DIR, gate_file=DEFAULT_OOD_PATH.name,
    preprocess=load_image, input_max=1.0, expected_input="a chest X-ray", class_names=CLASS_NAMES,
    model_files=MODEL_FILES,
)


def predict(image, explain: bool = True, ensemble: bool = False) -> Prediction:
    """Run the cascade for one X-ray: a path, bytes, a file-like upload, a PIL image or an array."""
    return MODEL.predict(image, explain=explain, ensemble=ensemble)


def predict_batch(images: list, ensemble: bool = False) -> pd.DataFrame:
    """The cascade for many images, without heatmaps: one row per image."""
    return MODEL.predict_batch(images, ensemble=ensemble)


if __name__ == "__main__":
    run_cli(MODEL, "Run the lung chest X-ray cascade.")
