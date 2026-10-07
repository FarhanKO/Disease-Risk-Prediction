"""
data.py — Class list, split loading and single-image preprocessing for the kidney CT
module. Mirrors load_image() in notebooks/Kidney_CT.ipynb, so the saved checkpoints get
exactly the input they were trained on.

Dataset layout, built by data/build_kidney_dataset.py (one folder per class in each split;
splits are by CT scan series, so no scan is in more than one split):

    data/train/<class>/*.png    data/val/<class>/*.png    data/test/<class>/*.png
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common.preprocessing.image import open_image

MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad with black (the CT background) so resizing never stretches anatomy."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_image(source) -> np.ndarray:
    """
    One CT slice -> float32 (1, 224, 224, 3) in 0-255: grayscale, padded to a square,
    resized with area (BOX) resampling, repeated to 3 channels. Each model rescales the
    pixels itself. `source`: a path, bytes, a file-like upload, a PIL image or a uint8 array.
    """
    img = pad_to_square(open_image(source).convert("L")).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BOX)
    gray = np.asarray(img, dtype=np.float32)[..., np.newaxis]
    return np.repeat(gray, 3, axis=-1)[np.newaxis]


def load_split(split: str, data_dir=DEFAULT_DATA_DIR) -> tuple[np.ndarray, np.ndarray, list]:
    """All slices of one split: (uint8 array (N, 224, 224, 3), integer labels, file paths)."""
    paths, labels = [], []
    for index, name in enumerate(CLASS_NAMES):
        files = sorted(p for p in (Path(data_dir) / split / name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        paths += files
        labels += [index] * len(files)
    images = np.concatenate([load_image(p).astype(np.uint8) for p in paths])
    return images, np.array(labels), paths
