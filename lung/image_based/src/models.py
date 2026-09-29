"""
models.py — Architectures, fine-tuning helpers, and the registry of saved
.keras checkpoints for the lung chest X-ray module.

Every transfer model is Sequential([backbone, GlobalAveragePooling2D,
Dense(256), Dropout(0.4), Dense(6, softmax)]), with the pretrained backbone
nested as layer 0. explain.py and predict.py rely on that layout.
"""

from pathlib import Path
from .data import IMG_SIZE, NUM_CLASSES

MODULE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = MODULE_DIR / "models"

# Phase 1 (frozen backbone) and Phase 2 (fine-tuned, "_FT") checkpoints,
# named exactly as train.py's ModelCheckpoint writes them.
MODEL_FILES = {
    "Custom_CNN": "Custom_CNN_best.keras",
    "ResNet50": "ResNet50_best.keras",
    "ResNet50_FT": "ResNet50_FT_best.keras",
    "EfficientNetB0": "EfficientNetB0_best.keras",
    "EfficientNetB0_FT": "EfficientNetB0_FT_best.keras",
    "DenseNet121": "DenseNet121_best.keras",
    "DenseNet121_FT": "DenseNet121_FT_best.keras",
}

# The final version of each architecture — what the notebook compared
FINAL_MODELS = ["Custom_CNN", "ResNet50_FT", "EfficientNetB0_FT", "DenseNet121_FT"]
BEST_MODEL = "DenseNet121_FT"
ENSEMBLE_MEMBERS = ["ResNet50_FT", "EfficientNetB0_FT", "DenseNet121_FT"]

TRANSFER_BACKBONES = ["ResNet50", "EfficientNetB0", "DenseNet121"]
NUM_UNFREEZE = 30

INPUT_SHAPE = (*IMG_SIZE, 3)
_model_cache: dict = {}


def build_custom_cnn(num_classes: int = NUM_CLASSES):
    """Three conv blocks + dense head, trained from scratch (the baseline)."""
    import keras
    from keras import layers

    return keras.Sequential([
        keras.Input(INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Custom_CNN")


def build_transfer_model(backbone: str, num_classes: int = NUM_CLASSES):
    """
    ImageNet backbone (frozen for Phase 1) + the shared classification head.
    Returns (model, base) so train.py can unfreeze `base` for Phase 2.
    """
    import keras
    from keras import layers

    constructors = {
        "ResNet50": keras.applications.ResNet50,
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "DenseNet121": keras.applications.DenseNet121,
    }
    base = constructors[backbone](weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False

    model = keras.Sequential([
        keras.Input(INPUT_SHAPE),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name=backbone)
    return model, base


def unfreeze_top_layers(base, num_unfreeze: int = NUM_UNFREEZE) -> None:
    """Phase 2: make only the last `num_unfreeze` backbone layers trainable."""
    base.trainable = True
    for layer in base.layers[:-num_unfreeze]:
        layer.trainable = False


def model_path(name: str, model_dir: str | Path = MODEL_DIR) -> Path:
    if name not in MODEL_FILES:
        raise KeyError(f"Unknown model '{name}'. Choose from: {list(MODEL_FILES)}")
    return Path(model_dir) / MODEL_FILES[name]


def load_model(name: str = BEST_MODEL, model_dir: str | Path = MODEL_DIR):
    """Load (and cache) a saved checkpoint by registry name."""
    import keras

    path = model_path(name, model_dir)
    key = str(path.resolve())
    if key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {path}. The .keras files are not in git — "
                "see the module README for where to get them."
            )
        _model_cache[key] = keras.models.load_model(path, compile=False)
    return _model_cache[key]


def available_models(model_dir: str | Path = MODEL_DIR) -> list[str]:
    return [name for name in MODEL_FILES if model_path(name, model_dir).exists()]


def has_nested_backbone(model) -> bool:
    import keras
    return isinstance(model.layers[0], keras.Model)


def extract_embeddings(model, images, batch_size: int = 64):
    """
    Pooled backbone features (the GlobalAveragePooling2D output) for a batch
    of preprocessed images — the embedding space the OOD detector lives in.
    """
    import keras
    import numpy as np

    if not has_nested_backbone(model):
        raise ValueError("Embeddings need a transfer model with a nested backbone.")
    backbone, pooling = model.layers[0], model.layers[1]
    chunks = [
        keras.ops.convert_to_numpy(pooling(backbone(images[i:i + batch_size], training=False)))
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(chunks)
