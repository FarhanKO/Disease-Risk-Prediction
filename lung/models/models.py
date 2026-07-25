"""
Model Loader for Lung Disease Prediction System

This file loads trained deep learning models.
The actual .keras files are not included in GitHub because
they exceed GitHub's file size limits.

Download models and place them inside the models/ directory.
"""

import os
from tensorflow.keras.models import load_model

MODEL_DIR = "models"

MODEL_PATHS = {
    "Custom CNN": os.path.join(MODEL_DIR, "Custom_CNN_best.keras"),

    "DenseNet121": os.path.join(MODEL_DIR, "DenseNet121_best.keras"),
    "DenseNet121 Fine Tuned": os.path.join(
        MODEL_DIR, "DenseNet121_FT_best.keras"
    ),

    "EfficientNetB0": os.path.join(
        MODEL_DIR, "EfficientNetB0_best.keras"
    ),
    "EfficientNetB0 Fine Tuned": os.path.join(
        MODEL_DIR, "EfficientNetB0_FT_best.keras"
    ),

    "ResNet50": os.path.join(
        MODEL_DIR, "ResNet50_best.keras"
    ),
    "ResNet50 Fine Tuned": os.path.join(
        MODEL_DIR, "ResNet50_FT_best.keras"
    )
}
