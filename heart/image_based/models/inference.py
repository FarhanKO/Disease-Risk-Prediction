"""
inference.py - Stand-alone inference for the 12-lead ECG classifier, with no code from the
GitHub repo needed. Runs stages 1-3 of the cascade exactly as the repo's src/predict.py does:

    Stage 1 (OOD gate):        mean distance from the image's embedding (unit-normalised
                               2048-d pooled features of ResNet50V2_FT) to its 5 nearest
                               training ECGs. Above the threshold -> rejected, not an ECG.
    Stage 2 (Classification):  ResNet50V2_FT, or the soft-voting ensemble with --ensemble.
    Stage 3 (Confidence):      top-class probability below 0.75 -> manual review.

Grad-CAM (stage 4) lives in the GitHub repo (common/image.py).

    python inference.py ecg1.png ecg2.jpg                       # files next to this script
    python inference.py ecg.png --repo <user>/<repo>            # download from the Hugging Face Hub

    from inference import ECGClassifier
    clf = ECGClassifier(".")                                    # or ECGClassifier("<user>/<repo>")
    clf.predict("ecg.png")   # {'status': 'accepted', 'label': 'Normal', 'confidence': 0.97, ...}
"""

import argparse
import json
import os
from pathlib import Path

# Trained on Keras 3 with the PyTorch backend; must be set before keras is imported
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PRINTOUT_SIZE = (2213, 1572)               # a full scanned printout from the source dataset
GRID_BOX = (68, 283, 2177, 1518)           # crop to the ECG grid: drops patient ID, date, heart rate
BATCH_SIZE = 16


def _local_or_hub(source, filename: str) -> Path:
    """A file from a local folder, or downloaded (and cached) from a Hugging Face repo id."""
    local = Path(source) / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(source), filename=filename))


class ECGClassifier:
    def __init__(self, source=Path(__file__).resolve().parent, ensemble: bool = False):
        """`source`: a folder holding the model files, or a Hugging Face repo id."""
        self.source = source
        self.metadata = json.loads(_local_or_hub(source, "metadata.json").read_text())
        self.class_names = self.metadata["class_names"]
        self.img_size = tuple(self.metadata["img_size"])            # (height, width)
        self.threshold = float(self.metadata["confidence_threshold"])
        self.best = self.metadata["best_model"]
        self.members = self.metadata["ensemble_members"] if ensemble else [self.best]

        gate = np.load(_local_or_hub(source, "ood_gate.npz"))
        if str(gate["embedding_model"]) != self.best:
            raise ValueError(f"ood_gate.npz was built on {gate['embedding_model']}, not {self.best}")
        self.gate_embeddings = gate["embeddings"].astype(np.float64)
        self.gate_k, self.gate_threshold = int(gate["k"]), float(gate["threshold"])

        # Each model -> (pooled embedding, class probabilities) in one pass
        self.models = {}
        for name in dict.fromkeys([self.best] + self.members):
            model = keras.models.load_model(_local_or_hub(source, f"{name}_best.keras"), compile=False)
            self.models[name] = keras.Model(model.input, [model.get_layer("gap").output, model.outputs[0]])

    def preprocess(self, image) -> np.ndarray:
        """A path, PIL image or uint8 array -> float32 (H, W, 3) in 0-255. Do not divide by 255:
        every model rescales its input internally."""
        img = image if isinstance(image, Image.Image) else (
            Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image))
        img = img.convert("RGB")
        if img.size == PRINTOUT_SIZE:
            img = img.crop(GRID_BOX)
        # BOX (area) resampling keeps the thin ECG traces visible when shrinking
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BOX)
        return np.asarray(img, dtype=np.float32)

    def _run(self, name: str, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = [self.models[name].predict_on_batch(batch[i:i + BATCH_SIZE])
                   for i in range(0, len(batch), BATCH_SIZE)]
        return (np.concatenate([keras.ops.convert_to_numpy(e) for e, _ in outputs]),
                np.concatenate([keras.ops.convert_to_numpy(p) for _, p in outputs]))

    def _out_of_distribution(self, embeddings: np.ndarray) -> np.ndarray:
        unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        sq = (unit ** 2).sum(1)[:, None] + (self.gate_embeddings ** 2).sum(1)[None] - 2 * unit @ self.gate_embeddings.T
        nearest = np.sort(np.sqrt(np.maximum(sq, 0)), axis=1)[:, :self.gate_k]
        return nearest.mean(axis=1) > self.gate_threshold

    def predict_batch(self, images: list) -> list[dict]:
        batch = np.stack([self.preprocess(i) for i in images])
        embeddings, probabilities = self._run(self.best, batch)
        rejected = self._out_of_distribution(embeddings)
        if len(self.members) > 1:
            probabilities = np.mean([probabilities if m == self.best else self._run(m, batch)[1]
                                     for m in self.members], axis=0)

        results = []
        for is_rejected, probs in zip(rejected, probabilities):
            if is_rejected:
                results.append({"status": "rejected", "message": "Not a 12-lead ECG printout (OOD gate)"})
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
    parser = argparse.ArgumentParser(description="Classify 12-lead ECG printouts.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent),
                        help="Folder with the model files, or a Hugging Face repo id.")
    parser.add_argument("--ensemble", action="store_true", help="Soft-vote ResNet50V2, EfficientNetB0 and DenseNet121.")
    args = parser.parse_args()

    clf = ECGClassifier(args.repo, ensemble=args.ensemble)
    for path, result in zip(args.images, clf.predict_batch(args.images)):
        if result["status"] == "rejected":
            print(f"{Path(path).name}: rejected | {result['message']}")
            continue
        print(f"{Path(path).name}: {result['status']} | {result['label']} ({result['confidence']:.1%})")
        for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"   {name:24s} {p:6.1%}")


if __name__ == "__main__":
    main()
