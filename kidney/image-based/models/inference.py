"""
inference.py - Stand-alone inference for the kidney CT classifier, with no code from the
GitHub repo needed. Runs stages 1-3 of the cascade exactly as the repo's src/predict.py does:

    Stage 1 (OOD gate):        Isolation Forest on the pooled embedding (GlobalAveragePooling
                               output) of DenseNet121_FT. Outliers -> rejected, not a kidney CT.
    Stage 2 (Classification):  DenseNet121_FT, or the soft-voting ensemble with --ensemble.
    Stage 3 (Confidence):      top-class probability below 0.75 -> manual review.

Grad-CAM (stage 4) lives in the GitHub repo (common/image.py).

    python inference.py slice1.png slice2.jpg                   # files next to this script
    python inference.py slice.png --repo FarhanKO/kidney-ct-classifier   # download from the Hub

    from inference import KidneyCTClassifier
    clf = KidneyCTClassifier(".")                               # or KidneyCTClassifier("FarhanKO/kidney-ct-classifier")
    clf.predict("slice.png")   # {'status': 'accepted', 'label': 'Stone', 'confidence': 0.97, ...}

The OOD gate is a scikit-learn object saved with joblib (a pickle): only load it from a
source you trust.
"""

import argparse
import json
import os
from pathlib import Path

# Trained on Keras 3 with the PyTorch backend; must be set before keras is imported
os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GATE_FILE = "kidney_ct_ood_detector.joblib"
BATCH_SIZE = 16


def _local_or_hub(source, filename: str) -> Path:
    """A file from a local folder, or downloaded (and cached) from a Hugging Face repo id."""
    local = Path(source) / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(source), filename=filename))


def _no_grad():
    if keras.backend.backend() == "torch":
        import torch

        return torch.no_grad()
    import contextlib

    return contextlib.nullcontext()


class KidneyCTClassifier:
    def __init__(self, source=Path(__file__).resolve().parent, ensemble: bool = False):
        """`source`: a folder holding the model files, or a Hugging Face repo id."""
        self.metadata = json.loads(_local_or_hub(source, "metadata.json").read_text())
        self.class_names = self.metadata["class_names"]
        self.img_size = tuple(self.metadata["img_size"])            # (height, width)
        self.threshold = float(self.metadata["confidence_threshold"])
        self.best = self.metadata["best_model"]
        self.members = self.metadata["ensemble_members"] if ensemble else [self.best]

        gate = joblib.load(_local_or_hub(source, GATE_FILE))
        self.gate = gate["model"] if "model" in gate else gate["detector"]

        self.models = {name: keras.models.load_model(_local_or_hub(source, f"{name}_best.keras"), compile=False)
                       for name in dict.fromkeys([self.best] + self.members)}

    def preprocess(self, image) -> np.ndarray:
        """A path, PIL image or uint8 array -> float32 (224, 224, 3) in 0-255: grayscale, padded to a
        square with black, BOX-resized, repeated to 3 channels. Do not divide by 255: every model
        rescales its input internally."""
        img = image if isinstance(image, Image.Image) else (
            Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image))
        img = img.convert("L")
        side = max(img.size)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = canvas.resize((self.img_size[1], self.img_size[0]), Image.BOX)
        return np.repeat(np.asarray(img, dtype=np.float32)[..., None], 3, axis=-1)

    def _run(self, name: str, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pooled embeddings, class probabilities), running the model layer by layer."""
        layers = [l for l in self.models[name].layers if not isinstance(l, keras.layers.InputLayer)]
        split = next(i for i, l in enumerate(layers) if isinstance(l, keras.layers.GlobalAveragePooling2D)) + 1
        embeddings, probabilities = [], []
        with _no_grad():
            for start in range(0, len(batch), BATCH_SIZE):
                x = keras.ops.convert_to_tensor(batch[start:start + BATCH_SIZE])
                for layer in layers[:split]:
                    x = layer(x, training=False)
                embeddings.append(keras.ops.convert_to_numpy(x))
                for layer in layers[split:]:
                    x = layer(x, training=False)
                probabilities.append(keras.ops.convert_to_numpy(x))
        return np.concatenate(embeddings), np.concatenate(probabilities)

    def predict_batch(self, images: list) -> list[dict]:
        batch = np.stack([self.preprocess(i) for i in images])
        embeddings, probabilities = self._run(self.best, batch)
        rejected = self.gate.predict(embeddings) == -1              # -1 = outlier
        if len(self.members) > 1:
            probabilities = np.mean([probabilities if m == self.best else self._run(m, batch)[1]
                                     for m in self.members], axis=0)

        results = []
        for is_rejected, probs in zip(rejected, probabilities):
            if is_rejected:
                results.append({"status": "rejected", "message": "Not a kidney CT slice (OOD gate)"})
                continue
            top = int(probs.argmax())
            label, confidence = self.class_names[top], float(probs[top])
            results.append({
                "status": "accepted" if confidence >= self.threshold else "review",
                "label": label,
                "confidence": confidence,
                "p_abnormal": 1.0 - float(probs[self.class_names.index("Normal")]),
                "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
            })
        return results

    def predict(self, image) -> dict:
        return self.predict_batch([image])[0]


def main():
    parser = argparse.ArgumentParser(description="Classify kidney CT slices (Cyst, Normal, Stone, Tumor).")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent),
                        help="Folder with the model files, or a Hugging Face repo id.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Soft-vote ResNet50V2, EfficientNetV2B0, DenseNet121 and ConvNeXtTiny.")
    args = parser.parse_args()

    clf = KidneyCTClassifier(args.repo, ensemble=args.ensemble)
    for path, result in zip(args.images, clf.predict_batch(args.images)):
        if result["status"] == "rejected":
            print(f"{Path(path).name}: rejected | {result['message']}")
            continue
        print(f"{Path(path).name}: {result['status']} | {result['label']} ({result['confidence']:.1%})")
        for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"   {name:10s} {p:6.1%}")


if __name__ == "__main__":
    main()
