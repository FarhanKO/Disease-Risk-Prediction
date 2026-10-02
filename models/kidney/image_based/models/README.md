**See the models here: https://huggingface.co/FarhanKO/kidney-disease-prediction**

# Kidney CT Classifier (DenseNet121)

Classifies a kidney CT slice into one of four classes: **Cyst, Normal, Stone,
Tumor**.

> **Research and education only.** This is not a medical device. It was trained
> on 123 CT scans from hospitals in one city and must not be used for diagnosis
> or treatment decisions.

## Quick start

```bash
pip install -r requirements.txt     # keras 3, torch, scikit-learn, numpy, pillow, huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util

REPO_ID = "FarhanKO/kidney-disease-prediction"
spec = importlib.util.spec_from_file_location("inference", hf_hub_download(REPO_ID, "inference.py"))
inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)

clf = inference.KidneyCTClassifier(REPO_ID)     # downloads metadata, OOD gate and DenseNet121_FT (~90 MB)
clf.predict("slice.png")
# {'status': 'accepted', 'label': 'Cyst', 'confidence': 0.943, 'p_abnormal': 0.988,
#  'probabilities': {'Cyst': 0.943, 'Normal': 0.012, 'Stone': 0.019, 'Tumor': 0.026}}
```

Or from the command line (`--ensemble` soft-votes the four fine-tuned models):

```bash
python inference.py slice1.png slice2.jpg --repo FarhanKO/kidney-disease-prediction
```

`status` is `accepted` (top-class probability ≥ 0.75), `review` (below 0.75:
flag for a clinician) or `rejected` (the OOD gate says this is not a kidney CT
slice; it is not classified). On the full test split, `inference.py`
reproduces the notebook exactly: accuracy 0.8942, 3.64 % of slices rejected by
the gate.

The OOD gate is a scikit-learn object saved with joblib, which is a pickle:
load it only from a source you trust, and use scikit-learn 1.8 (pinned in
`requirements.txt`).

## Download into the GitHub repo

To use the full pipeline (with Grad-CAM) from a clone of the GitHub repo, put
the files in this folder, where `src/predict.py` looks for them:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# Everything (~1 GB): all nine checkpoints + gate + metadata
snapshot_download("FarhanKO/kidney-disease-prediction", local_dir="models")

# Only what the deployed cascade needs (~90 MB)
snapshot_download("FarhanKO/kidney-disease-prediction", local_dir="models",
                  allow_patterns=["DenseNet121_FT_best.keras", "kidney_ct_ood_detector.joblib", "metadata.json"])
```

Then, from `kidney/image_based/`:

```bash
python -m src.predict --image slice.png --heatmap-dir out/
python -m src.evaluate      # must reproduce the test metrics in metadata.json
```

## Using a checkpoint directly

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # before importing keras; trained on the torch backend
import keras
model = keras.models.load_model("DenseNet121_FT_best.keras", compile=False)
probs = model.predict(x)   # x: (N, 224, 224, 3) float32 in 0-255; classes in metadata.json order
```

**Input preprocessing** (the same as `load_image()` in `src/data.py`):
grayscale → pad to square with black → resize to **224 × 224** (PIL `BOX`) →
repeat to 3 channels → float32 in **0–255**. **Do not divide by 255**: every
model rescales its input internally.

## Files

| File | Size | What it is |
|---|---|---|
| `DenseNet121_FT_best.keras` | 84 MB | **Deployed model.** ImageNet DenseNet121, fully fine-tuned |
| `ResNet50V2_FT_best.keras` | 276 MB | ResNet50V2, fine-tuned (ensemble member) |
| `EfficientNetV2B0_FT_best.keras` | 72 MB | EfficientNetV2B0, fine-tuned (ensemble member) |
| `ConvNeXtTiny_FT_best.keras` | 321 MB | ConvNeXtTiny, fine-tuned (ensemble member) |
| `DenseNet121_best.keras` · `ResNet50V2_best.keras` · `EfficientNetV2B0_best.keras` · `ConvNeXtTiny_best.keras` | 31 / 97 / 27 / 109 MB | Phase-1 checkpoints (frozen backbone, head only), for comparison |
| `Custom_CNN_best.keras` | 5 MB | Small CNN trained from scratch (baseline) |
| `kidney_ct_ood_detector.joblib` | 4 MB | Stage 1 OOD gate: Isolation Forest on DenseNet121_FT embeddings |
| `metadata.json` | – | Class order, input size, preprocessing, deployed model, ensemble members, confidence threshold, test metrics |
| `inference.py` | – | Stand-alone cascade (stages 1–3) |
| `requirements.txt` | – | Dependencies for `inference.py` |

Every `.keras` file takes a `(N, 224, 224, 3)` input and outputs softmax
probabilities for the 4 classes, in the order `Cyst, Normal, Stone, Tumor`.

## How the cascade uses them

| Stage | What | Outcome |
|---|---|---|
| 1. OOD gate | Isolation Forest on DenseNet121_FT embeddings | Not a kidney CT slice → `rejected` |
| 2. Classification | DenseNet121_FT | Probabilities for the 4 classes |
| 3. Confidence check | Top-class probability < 0.75 | `review` (flag for a clinician) |
| 4. Explainability | Grad-CAM on the top class | Heatmap and overlay (in the GitHub repo) |

## Training data

| Split | Slices | Scans |
|---|---|---|
| Train | 8,353 | 123 |
| Val | 1,789 | 30 |
| Test | 1,787 | 36 |

## Training

- **Framework:** Keras 3 on the PyTorch backend (TensorFlow has no GPU support
  on native Windows), trained on an RTX 5060 (8 GB). Seed 42.
- **Backbones:** ImageNet-pretrained ResNet50V2, EfficientNetV2B0, DenseNet121
  and ConvNeXtTiny from Keras Applications, each applying its own input
  scaling, plus a small from-scratch CNN.
- **Phase 1:** 5-epoch head warm-up with the backbone frozen.
- **Phase 2 (`_FT`):** full fine-tuning of every layer, with BatchNorm frozen.
  AdamW (lr 3e-5, weight decay 0.05), label smoothing 0.1, class weights, and
  CT-safe augmentation (small rotation, zoom and translation; **no flips**).
- **Model selection:** highest **validation** macro-F1, then lowest validation
  loss. The test set was not used for selection.

## Results

Held-out test set: 1,787 slices from 36 scans never seen in training.

| Model | Val macro-F1 | Test acc. | Test macro-F1 | Test macro ROC-AUC |
|---|---|---|---|---|
| Custom CNN | 0.816 | 0.730 | 0.696 | 0.928 |
| ResNet50V2 (fine-tuned) | 0.844 | 0.868 | 0.826 | 0.973 |
| EfficientNetV2B0 (fine-tuned) | 0.817 | 0.806 | 0.737 | 0.965 |
| **DenseNet121 (fine-tuned) — deployed** | **0.889** | **0.894** | **0.867** | **0.976** |
| ConvNeXtTiny (fine-tuned) | 0.794 | 0.851 | 0.798 | 0.968 |
| Soft-voting ensemble (4 × FT) | 0.868 | 0.877 | 0.841 | 0.983 |
| Phase-1 (frozen backbone) models | 0.53–0.70 | 0.56–0.64 | 0.47–0.53 | 0.76–0.87 |

Scan-level 95 % bootstrap CIs for the deployed model: accuracy 0.82–0.96,
macro-F1 0.70–0.91.

Per class, deployed model, test set:

| Class | Precision | Recall | F1 | Slices |
|---|---|---|---|---|
| Cyst | 0.87 | 0.99 | 0.93 | 493 |
| Normal | 0.90 | 0.98 | 0.93 | 749 |
| Stone | 0.95 | 0.77 | 0.85 | 204 |
| Tumor | 0.90 | 0.66 | 0.76 | 341 |

**Confidence check (0.75):** 81 % of test slices are accepted, and 96.3 % of
those are correct. The 19 % sent to review are 59.6 % correct.

**OOD gate:** rejects 3.6 % of real test slices, and 100 % of random noise,
black images, grey speckle, chest X-rays and ECG printouts.

The ensemble is not deployed: it scores lower on validation (0.868 vs 0.889),
and its tumor recall drops to 0.50.

## Limitations

- **Few independent scans.** Only 4 tumor scans each in val and test, so
  per-class tumor numbers move a lot with one scan. Tumor recall (0.66) is the
  weakest point.
- **Acquisition shortcuts.** Body brightness differs by class (contrast phase,
  display window), and Grad-CAM often spreads over the central abdomen and
  spine, so part of the signal may be how the scan was taken.
- **One source, one format.** All scans come from Dhaka hospitals, exported as
  windowed JPEGs, not Hounsfield-unit DICOM. Other scanners and windows are
  untested and may be rejected by the gate.
- **Single label per slice**, and scans were grouped from image similarity
  because the dataset has no patient IDs.
