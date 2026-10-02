"""
data.py — Dataset constants, loading, augmentation, and single-image
preprocessing for the lung chest X-ray module.

Expected dataset layout (one folder per class inside each split):

    <data-dir>/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"

# Alphabetical, so it matches the index order Keras infers from folder names
CLASS_NAMES = [
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
    "Tuberculosis",
]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def split_dir(data_dir: str | Path, split: str) -> Path:
    """Resolve and validate `<data-dir>/<split>`."""
    path = Path(data_dir) / split
    if not path.is_dir():
        raise FileNotFoundError(f"Split directory not found: {path}")
    return path


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def count_images(data_dir: str | Path) -> pd.DataFrame:
    """Images per class per split, with totals — the notebook's item-count table."""
    counts = {
        split.capitalize(): {
            name: len(list_images(split_dir(data_dir, split) / name))
            for name in CLASS_NAMES
        }
        for split in SPLITS
    }
    table = pd.DataFrame(counts)
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


def compute_class_weights(data_dir: str | Path) -> dict[int, float]:
    """Balanced class weights (n_samples / (n_classes * n_class)) for model.fit()."""
    train = split_dir(data_dir, "train")
    counts = np.array([len(list_images(train / name)) for name in CLASS_NAMES])
    weights = counts.sum() / (NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_augmentation():
    """
    Training-time augmentation, applied on [0, 1] images.

    Ported from the notebook's ImageDataGenerator (removed in Keras 3):
    rotation 15°, zoom 0.1, horizontal flip, brightness. Two differences:
    RandomBrightness shifts by ±0.1 where the notebook scaled by 0.9–1.1,
    and the notebook's `shear_range=0.1` was in degrees — no visible
    effect — so it is left out.
    """
    import keras

    return keras.Sequential([
        keras.layers.RandomRotation(15 / 360, fill_mode="nearest", seed=SEED),
        keras.layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED),
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0), seed=SEED),
    ], name="augmentation")


def make_dataset(
    data_dir: str | Path,
    split: str,
    augment: bool = False,
    shuffle: bool | None = None,
    batch_size: int = BATCH_SIZE,
):
    """
    Batched tf.data pipeline yielding (images in [0, 1], one-hot labels).

    Only the train split is shuffled by default, so val/test predictions line
    up with `dataset.file_paths` and `dataset_labels(dataset)`.
    """
    import keras
    import tensorflow as tf

    shuffle = (split == "train") if shuffle is None else shuffle
    raw = keras.utils.image_dataset_from_directory(
        split_dir(data_dir, split),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    file_paths = raw.file_paths

    ds = raw.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        augmentation = build_augmentation()
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds.file_paths = file_paths
    return ds


def dataset_labels(dataset) -> np.ndarray:
    """Integer labels for an unshuffled dataset, read from its file paths."""
    return np.array([CLASS_NAMES.index(Path(p).parent.name) for p in dataset.file_paths])


def load_image(source) -> np.ndarray:
    """
    Preprocess one image for inference: RGB, 224x224, scaled to [0, 1],
    with a batch axis — shape (1, 224, 224, 3).

    Accepts a file path, raw bytes, a file-like object (e.g. a Streamlit
    upload), a PIL image, or an HxW / HxWxC uint8 array.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]
