"""
inference.py - Stand-alone inference for the chest X-ray classifier, with no code from the
GitHub repo needed. Runs stages 1-3 of the cascade exactly as the repo's src/predict.py does:

    Stage 1 (OOD gate):        Isolation Forest on the pooled embedding (GlobalAveragePooling
                               output) of DenseNet121_FT. Outliers -> rejected, not a chest X-ray.
    Stage 2 (Classification):  DenseNet121_FT, or the soft-voting ensemble with --ensemble.
    Stage 3 (Confidence):      top-class probability below 0.75 -> manual review.

Grad-CAM (stage 4) lives in the GitHub repo (common/image.py).

    python inference.py xray1.jpg xray2.png                     # files next to this script
    python inference.py xray.jpg --repo FarhanKO/lung-xray-classifier    # download from the Hub

    from inference import ChestXrayClassifier
    clf = ChestXrayClassifier(".")                              # or ChestXrayClassifier("FarhanKO/lung-xray-classifier")
    clf.predict("xray.jpg")   # {'status': 'accepted', 'label': 'Tuberculosis', 'confidence': 0.99, ...}

The checkpoints were trained with TensorFlow; they run on the PyTorch backend when torch is
installed (as in the repo) and on TensorFlow otherwise. Set KERAS_BACKEND to choose.

The OOD gate is a scikit-learn object saved with joblib (a pickle): only load it from a
source you trust.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

# Must be set before keras is imported; an explicit KERAS_BACKEND still wins
if importlib.util.find_spec("torch") is not None:
    os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GATE_FILE = "lung_ood_detector.joblib"
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


class ChestXrayClassifier:
    def __init__(self, source=Path(__file__).resolve().parent, ensemble: bool = False):
        """`source`: a folder holding the model files, or a Hugging Face repo id."""
        self.metadata = json.loads(_local_or_hub(source, "metadata.json").read_text())
        self.class_names = self.metadata["class_names"]
        self.img_size = tuple(self.metadata["img_size"])            # (height, width)
        self.threshold = float(self.metadata["confidence_threshold"])
        self.best = self.metadata["best_model"]
        self.members = self.metadata["ensemble_members"] if ensemble else [self.best]

        gate = joblib.load(_local_or_hub(source, GATE_FILE))
        self.gate = gate["detector"]
        self.gate_model = gate.get("embedding_model", self.best)

        self.models = {name: keras.models.load_model(_local_or_hub(source, f"{name}_best.keras"), compile=False)
                       for name in dict.fromkeys([self.best, self.gate_model] + self.members)}

    def preprocess(self, image) -> np.ndarray:
        """A path, PIL image or uint8 array -> float32 (224, 224, 3) in [0, 1]: RGB, bilinear
        resize, divided by 255. Unlike the heart and kidney models, these checkpoints have no
        rescaling layer, so the division is required."""
        img = image if isinstance(image, Image.Image) else (
            Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image))
        img = img.convert("RGB").resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

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
        if self.gate_model != self.best:
            embeddings = self._run(self.gate_model, batch)[0]
        rejected = self.gate.predict(embeddings) == -1              # -1 = outlier
        if len(self.members) > 1:
            probabilities = np.mean([probabilities if m == self.best else self._run(m, batch)[1]
                                     for m in self.members], axis=0)

        results = []
        for is_rejected, probs in zip(rejected, probabilities):
            if is_rejected:
                results.append({"status": "rejected", "message": "Not a chest X-ray (OOD gate)"})
                continue
            order = np.argsort(probs)[::-1]
            label, confidence = self.class_names[int(order[0])], float(probs[order[0]])
            results.append({
                "status": "accepted" if confidence >= self.threshold else "review",
                "label": label,
                "confidence": confidence,
                "p_abnormal": 1.0 - float(probs[self.class_names.index("Normal")]),
                "top_3": [(self.class_names[int(i)], float(probs[i])) for i in order[:3]],
                "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
            })
        return results

    def predict(self, image) -> dict:
        return self.predict_batch([image])[0]


def main():
    parser = argparse.ArgumentParser(description="Classify chest X-rays (Covid-19, Emphysema, Normal, "
                                                 "Pneumonia-Bacterial, Pneumonia-Viral, Tuberculosis).")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent),
                        help="Folder with the model files, or a Hugging Face repo id.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Soft-vote ResNet50, EfficientNetB0 and DenseNet121 (less calibrated; see README).")
    args = parser.parse_args()

    clf = ChestXrayClassifier(args.repo, ensemble=args.ensemble)
    for path, result in zip(args.images, clf.predict_batch(args.images)):
        if result["status"] == "rejected":
            print(f"{Path(path).name}: rejected | {result['message']}")
            continue
        print(f"{Path(path).name}: {result['status']} | {result['label']} ({result['confidence']:.1%})")
        for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"   {name:20s} {p:6.1%}")


if __name__ == "__main__":
    main()
