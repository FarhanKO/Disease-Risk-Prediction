"""
explain.py — Grad-CAM heatmaps showing which regions of the X-ray drove a
prediction, and helpers to overlay them on the original image. Both come from
common.image, which all three image modules share and which runs on the
PyTorch or the TensorFlow Keras backend.
"""

import numpy as np

from common.image import gradcam, overlay


def make_gradcam_heatmap(model, img_array, class_index: int | None = None) -> np.ndarray:
    """
    Grad-CAM for one preprocessed image of shape (1, H, W, 3). Returns a
    heatmap in [0, 1] at the feature-map resolution (7x7 for DenseNet121).
    """
    heatmap, _ = gradcam(model, img_array, class_index=class_index)
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a jet-coloured heatmap onto an image. `image` is HxWx3, either
    [0, 1] floats or uint8. Returns uint8 RGB at the image's size.
    """
    base = np.asarray(image, dtype=np.float32)
    if base.max() <= 1.0:
        base = base * 255.0
    return overlay(np.clip(base, 0, 255).astype(np.uint8), heatmap, alpha=alpha)
