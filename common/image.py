"""
image.py — The image cascade shared by the chest X-ray, ECG and kidney CT modules.

Every image passes through the whole cascade:

    Stage 1 (OOD Gate):        the best model's pooled embedding is compared with the
                               training images (Isolation Forest or kNN distance, whichever
                               the module's notebook validated). Inputs that are not the
                               expected kind of image are rejected and never classified.
    Stage 2 (Classification):  the best model, or the soft-voting ensemble, scores the classes.
    Stage 3 (Confidence):      a top-class probability below the threshold is flagged for
                               manual review.
    Stage 4 (Explainability):  Grad-CAM heatmap of the top class, and its overlay.

Keras runs on the PyTorch backend when torch is installed: TensorFlow has no GPU support
on native Windows, and the ECG and CT models were trained that way. Grad-CAM works on
both backends, and the chest X-ray checkpoints (trained with TensorFlow) load on either.
"""

import contextlib
import importlib.util
import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional, Union

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from .prediction import ACCEPTED, REJECTED, REVIEW, Prediction

# Must happen before keras is first imported; an explicit KERAS_BACKEND still wins
if importlib.util.find_spec("torch") is not None:
    os.environ.setdefault("KERAS_BACKEND", "torch")

BATCH_SIZE = 16          # 8 GB of GPU memory runs out at 32 on the 448x256 ECG inputs
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


# ---------- running the models ----------
# Every model here is a straight chain (input scaling -> backbone -> pooling -> head), so it
# can be run layer by layer. That gives the pooled embedding and the probabilities in one
# pass, and lets Grad-CAM stop the gradient at the backbone's feature map.

def _chain(model) -> list:
    import keras

    return [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]


def _no_grad():
    import keras

    if keras.backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


def forward(model, images: np.ndarray, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Preprocessed images (N, H, W, 3) -> (pooled embeddings for the OOD gate, class probabilities)."""
    import keras

    layers = _chain(model)
    pooling = [i for i, layer in enumerate(layers) if isinstance(layer, keras.layers.GlobalAveragePooling2D)]
    if not pooling:
        raise ValueError(f"{model.name} has no GlobalAveragePooling2D layer to take the OOD-gate embedding from")
    split = pooling[0] + 1
    embeddings, probabilities = [], []
    with _no_grad():
        for start in range(0, len(images), batch_size):
            x = keras.ops.convert_to_tensor(np.asarray(images[start:start + batch_size], dtype="float32"))
            for layer in layers[:split]:
                x = layer(x, training=False)
            embeddings.append(keras.ops.convert_to_numpy(x))
            for layer in layers[split:]:
                x = layer(x, training=False)
            probabilities.append(keras.ops.convert_to_numpy(x))
    return np.concatenate(embeddings), np.concatenate(probabilities)


def backbone(model):
    """The pre-trained network nested inside a transfer model; None for a from-scratch CNN."""
    import keras

    return next((layer for layer in model.layers if isinstance(layer, keras.Model)), None)


def gradcam(model, image: np.ndarray, class_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM for one preprocessed image, shape (1, H, W, 3). Returns (heatmap in [0, 1] at
    the feature-map resolution, class probabilities from the same forward pass).

    The gradient is taken with respect to the backbone's output (the last Conv2D for a
    from-scratch CNN), cut from the graph, so it works with frozen backbones too.
    """
    import keras

    target = backbone(model)
    if target is None:
        target = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)][-1]
    layers = _chain(model)
    backend = keras.backend.backend()

    if backend == "torch":
        import torch

        x = keras.ops.convert_to_tensor(np.asarray(image, dtype="float32"))
        with torch.enable_grad():
            for layer in layers:
                x = layer(x, training=False)
                if layer is target:
                    features = x.detach().requires_grad_(True)
                    x = features
            index = int(torch.argmax(x[0])) if class_index is None else class_index
            grads = torch.autograd.grad(x[:, index].sum(), features)[0]
        features, grads, probabilities = (keras.ops.convert_to_numpy(t.detach()) for t in (features, grads, x))
    elif backend == "tensorflow":
        import tensorflow as tf

        x = tf.convert_to_tensor(np.asarray(image, dtype="float32"))
        with tf.GradientTape() as tape:
            for layer in layers:
                x = layer(x, training=False)
                if layer is target:
                    features = x
                    tape.watch(features)
            index = int(tf.argmax(x[0])) if class_index is None else class_index
            score = x[:, index]
        grads = tape.gradient(score, features)
        features, grads, probabilities = features.numpy(), grads.numpy(), x.numpy()
    else:
        raise NotImplementedError(f"Grad-CAM supports the torch and tensorflow backends, not {backend}")

    weights = grads.mean(axis=(0, 1, 2))                  # how much each feature-map channel matters
    heatmap = np.maximum(features[0] @ weights, 0)        # positive influence only
    return heatmap / (heatmap.max() + 1e-8), probabilities[0]


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Jet-coloured heatmap blended onto an (H, W, 3) uint8 image; returns uint8 RGB."""
    import matplotlib

    h, w = image.shape[:2]
    resized = np.asarray(Image.fromarray(heatmap.astype(np.float32)).resize((w, h), Image.BILINEAR))
    jet = matplotlib.colormaps["jet"](np.uint8(255 * np.clip(resized, 0, 1)))[..., :3] * 255
    return np.clip(jet * alpha + image * (1 - alpha), 0, 255).astype(np.uint8)


def is_out_of_distribution(embeddings: np.ndarray, gate: dict) -> np.ndarray:
    """
    True where an embedding lies outside the training images (the input is rejected).
    Reads the three saved gate formats: {'detector', 'embedding_model'} (lung, Isolation
    Forest), {'index', 'k', 'threshold'} (heart, kNN) and {'type', ...} (kidney, either).
    """
    kind = gate.get("type", "isolation_forest" if "detector" in gate else "knn")
    if kind == "isolation_forest":
        detector = gate["model"] if "model" in gate else gate["detector"]
        return detector.predict(embeddings) == -1              # -1 = outlier
    # Mean cosine distance to the k nearest training embeddings
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return gate["index"].kneighbors(unit, n_neighbors=gate["k"])[0].mean(axis=1) > gate["threshold"]


class ImageModel:
    """One module's cascade: OOD gate -> classifier -> confidence check -> Grad-CAM."""

    def __init__(self, organ: str, model_dir: Union[str, Path], gate_file: str,
                 preprocess: Callable[..., np.ndarray], input_max: float, expected_input: str,
                 class_names: Optional[list] = None, normal_class: str = "Normal",
                 model_files: Optional[dict] = None):
        self.organ = organ
        self.model_dir = Path(model_dir)
        self.gate_path = self.model_dir / gate_file
        self.preprocess = preprocess              # image source -> float32 (1, H, W, 3), as the model expects
        self.input_max = input_max                # 1.0 when preprocess scales to [0, 1], 255.0 for raw pixels
        self.expected_input = expected_input      # "a chest X-ray", for the rejection message
        self.class_names = list(class_names) if class_names else None
        self.normal_class = normal_class
        self.model_files = model_files or {}      # name -> file name; default "<name>_best.keras"
        self.metadata = None
        self._gate = None
        self._models = {}
        self._lock = threading.Lock()

    def load(self) -> "ImageModel":
        """Read metadata.json and the OOD gate (once per process). Checkpoints load on first use."""
        with self._lock:
            if self.metadata is None:
                metadata = json.loads((self.model_dir / "metadata.json").read_text())
                if self.class_names and metadata["class_names"] != self.class_names:
                    raise ValueError(f"metadata.json classes {metadata['class_names']} != {self.class_names}")
                self.class_names = metadata["class_names"]
                self._gate = joblib.load(self.gate_path)
                self.metadata = metadata
        return self

    @property
    def best_model(self) -> str:
        return self.load().metadata["best_model"]

    @property
    def ensemble_members(self) -> list:
        return self.load().metadata.get("ensemble_members", [])

    @property
    def threshold(self) -> float:
        """Stage 3 confidence threshold (the notebook chose it on the validation set)."""
        return float(self.load().metadata.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))

    def model(self, name: Optional[str] = None):
        """Load (and cache) a checkpoint by name, e.g. 'DenseNet121_FT'; default: the best model."""
        import keras

        name = name or self.best_model
        with self._lock:
            if name not in self._models:
                path = self.model_dir / self.model_files.get(name, f"{name}_best.keras")
                if not path.exists():
                    raise FileNotFoundError(f"Missing checkpoint {path}. The .keras files are not in git; "
                                            "see the module README for where to get them.")
                self._models[name] = keras.models.load_model(path, compile=False)
        return self._models[name]

    def score(self, images: np.ndarray, ensemble: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Stages 1-2 for a stack of preprocessed images: (rejected by the OOD gate, class
        probabilities). Probabilities are returned for rejected images too, for evaluation.
        """
        self.load()
        embeddings, best = forward(self.model(), images)
        gate_model = self._gate.get("embedding_model", self.best_model)
        if gate_model != self.best_model:
            embeddings = forward(self.model(gate_model), images)[0]
        rejected = is_out_of_distribution(embeddings, self._gate)
        if not ensemble:
            return rejected, best
        members = [best if m == self.best_model else forward(self.model(m), images)[1]
                   for m in self.ensemble_members]
        return rejected, np.mean(members, axis=0)

    def display_image(self, image: np.ndarray) -> np.ndarray:
        """A preprocessed (H, W, 3) model input as uint8, for overlays."""
        return np.clip(np.asarray(image) * (255.0 / self.input_max), 0, 255).astype(np.uint8)

    def predict(self, image, explain: bool = True, ensemble: bool = False) -> Prediction:
        """Run the cascade for one image (a path, bytes, a file-like upload, a PIL image or an array)."""
        batch = self.preprocess(image)
        rejected, probabilities = self.score(batch, ensemble=ensemble)
        threshold = self.threshold
        details = {"model": "soft-voting ensemble" if ensemble else self.best_model, "threshold": threshold}
        if rejected[0]:
            return Prediction(organ=self.organ, modality="image", status=REJECTED,
                              message=f"Not {self.expected_input}: rejected by the out-of-distribution gate",
                              details=details)

        probs = probabilities[0]
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        label, confidence = self.class_names[top], float(probs[top])
        if confidence >= threshold:
            status, message = ACCEPTED, f"{label} ({confidence:.0%} confidence)"
        else:
            status = REVIEW
            message = f"{label} at {confidence:.0%} confidence, below the {threshold:.0%} threshold: manual review"

        explanation = {}
        if explain:     # from the single best model, also when the ensemble classified
            heatmap, _ = gradcam(self.model(), batch, class_index=top)
            explanation = {"method": f"Grad-CAM ({self.best_model}) for {label}", "heatmap": heatmap,
                           "overlay": overlay(self.display_image(batch[0]), heatmap)}

        return Prediction(
            organ=self.organ, modality="image", status=status, label=label,
            probability=1.0 - float(probs[self.class_names.index(self.normal_class)]),
            positive=label != self.normal_class, explanation=explanation, message=message,
            details={**details, "confidence": confidence,
                     "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
                     "top_3": [(self.class_names[i], float(probs[i])) for i in order[:3]]},
        )

    def predict_batch(self, images: list, ensemble: bool = False, chunk_size: int = 64) -> pd.DataFrame:
        """The cascade for many images, without heatmaps: one row per image."""
        self.load()
        threshold, normal = self.threshold, self.class_names.index(self.normal_class)
        rows = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start:start + chunk_size]
            rejected, probabilities = self.score(np.concatenate([self.preprocess(i) for i in chunk]), ensemble)
            for offset, (source, is_rejected, probs) in enumerate(zip(chunk, rejected, probabilities)):
                top = int(probs.argmax())
                scored = not is_rejected
                rows.append({
                    "image": str(source) if isinstance(source, (str, Path)) else start + offset,
                    "status": REJECTED if is_rejected else ACCEPTED if probs[top] >= threshold else REVIEW,
                    "label": self.class_names[top] if scored else None,
                    "probability": 1.0 - float(probs[normal]) if scored else np.nan,
                    "positive": (self.class_names[top] != self.normal_class) if scored else None,
                    "confidence": float(probs[top]) if scored else np.nan,
                })
        frame = pd.DataFrame(rows, columns=["image", "status", "label", "probability", "positive", "confidence"])
        frame["positive"] = frame["positive"].astype("boolean")
        return frame


def run_cli(model: ImageModel, description: str, argv=None):
    """`python -m src.predict --image a.png b.png [--ensemble] [--heatmap-dir out/]`."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", nargs="+", required=True, help="One or more image files.")
    parser.add_argument("--ensemble", action="store_true", help="Soft-vote the ensemble members in metadata.json.")
    parser.add_argument("--heatmap-dir", default=None, help="Save the Grad-CAM overlays here.")
    args = parser.parse_args(argv)

    for path in args.image:
        result = model.predict(path, explain=args.heatmap_dir is not None, ensemble=args.ensemble)
        print(f"\n{Path(path).name}: {result.status} | {result.message}")
        if not result.scored:
            continue
        print(f"   P(abnormal) = {result.probability:.1%}")
        for name, p in result.details["top_3"]:
            print(f"   {name:24s} {p:6.1%}")
        if args.heatmap_dir:
            out = Path(args.heatmap_dir) / f"{Path(path).stem}_gradcam.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.explanation["overlay"]).save(out)
            print(f"   Grad-CAM -> {out}")
