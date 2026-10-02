"""
predict.py — Cascade inference for the kidney CT module, behind the interface all six
modules share (common.image.ImageModel). This is the module Streamlit and the top-level
cascade import. Extracted from the clinical pipeline in notebook/Kidney_CT.ipynb, which
trains the models and writes everything this file loads.

Every image passes through the cascade — there is no non-cascade path:

    Stage 1 (OOD Gate):        Isolation Forest on the best model's embeddings (chosen over
                               a kNN gate by the notebook's validation rule). Inputs that
                               are not kidney CT slices are rejected.
    Stage 2 (Classification):  fine-tuned DenseNet121 (models/metadata.json: best_model),
                               or the soft-voting ensemble, scores the 4 classes.
    Stage 3 (Confidence):      top-class probability below 0.75 -> manual review.
    Stage 4 (Explainability):  Grad-CAM heatmap of the top class, and its overlay.

    from src.predict import predict
    result = predict("slice.png")  # common.Prediction: status, label, probability, positive, explanation

CLI (from kidney/image_based/):
    python -m src.predict --image slice1.png slice2.jpg --heatmap-dir out/
"""

import pandas as pd

from common import Prediction
from common.image import ImageModel, run_cli

from .data import CLASS_NAMES, MODULE_DIR, load_image

MODEL = ImageModel(
    organ="kidney", model_dir=MODULE_DIR / "models", gate_file="kidney_ct_ood_detector.joblib",
    preprocess=load_image, input_max=255.0, expected_input="a kidney CT slice", class_names=CLASS_NAMES,
)


def predict(image, explain: bool = True, ensemble: bool = False) -> Prediction:
    """Run the cascade for one CT slice: a path, bytes, a file-like upload, a PIL image or an array."""
    return MODEL.predict(image, explain=explain, ensemble=ensemble)


def predict_batch(images: list, ensemble: bool = False) -> pd.DataFrame:
    """The cascade for many images, without heatmaps: one row per image."""
    return MODEL.predict_batch(images, ensemble=ensemble)


if __name__ == "__main__":
    run_cli(MODEL, "Run the kidney CT cascade.")
