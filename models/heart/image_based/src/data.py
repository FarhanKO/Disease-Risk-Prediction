"""
data.py — Class list, split loading and single-image preprocessing for the 12-lead ECG
module. Mirrors load_image() in notebook/Heart_ECG.ipynb, so the saved checkpoints get
exactly the input they were trained on.

Dataset layout, built by data/build_ecg_dataset.py (one folder per class in each split):

    data/train/<class>/*.png    data/val/<class>/*.png    data/test/<class>/*.png
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common.image import open_image

MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"

CLASS_NAMES = ["Abnormal_Heartbeat", "History_of_MI", "Myocardial_Infarction", "Normal"]
IMG_SIZE = (256, 448)          # (height, width): keeps the ~1.7:1 printout aspect ratio
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Same constants as data/build_ecg_dataset.py: a full printout is cropped to the ECG grid,
# which removes the header (patient ID, date, heart rate) and the footer
PRINTOUT_SIZE = (2213, 1572)
GRID_BOX = (68, 283, 2177, 1518)


def load_image(source) -> np.ndarray:
    """
    One ECG image -> float32 (1, 256, 448, 3) in 0-255. Each model rescales the pixels
    itself, so nothing else is needed. Full uncropped printouts are cropped to the grid.
    `source`: a path, bytes, a file-like upload, a PIL image or a uint8 array.
    """
    img = open_image(source).convert("RGB")
    if img.size == PRINTOUT_SIZE:
        img = img.crop(GRID_BOX)
    # BOX (area) resampling keeps the thin ECG traces visible when shrinking
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BOX)
    return np.asarray(img, dtype=np.float32)[np.newaxis]


def load_split(split: str, data_dir=DEFAULT_DATA_DIR) -> tuple[np.ndarray, np.ndarray, list]:
    """All images of one split: (uint8 array (N, 256, 448, 3), integer labels, file paths)."""
    paths, labels = [], []
    for index, name in enumerate(CLASS_NAMES):
        files = sorted(p for p in (Path(data_dir) / split / name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        paths += files
        labels += [index] * len(files)
    images = np.concatenate([load_image(p).astype(np.uint8) for p in paths])
    return images, np.array(labels), paths
