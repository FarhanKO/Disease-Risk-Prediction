**See the models: https://huggingface.co/FarhanKO/lung-disease-prediction**

# Chest X-ray Classifier (DenseNet121)

Classifies a chest X-ray into one of six classes: **Covid-19, Emphysema,
Normal, Pneumonia-Bacterial, Pneumonia-Viral, Tuberculosis**.

> **Research and education only.** This is not a medical device and must not
> be used for diagnosis or treatment decisions.

## Quick start (no GitHub code needed)

```bash
pip install -r requirements.txt     # keras 3, torch, scikit-learn, numpy, pillow, huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util

REPO_ID = "FarhanKO/lung-disease-prediction"
spec = importlib.util.spec_from_file_location("inference", hf_hub_download(REPO_ID, "inference.py"))
inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)

clf = inference.ChestXrayClassifier(REPO_ID)    # downloads metadata, OOD gate and DenseNet121_FT (~38 MB)
clf.predict("xray.jpg")
# {'status': 'review', 'label': 'Covid-19', 'confidence': 0.627, 'p_abnormal': 0.934,
#  'top_3': [('Covid-19', 0.627), ('Emphysema', 0.134), ('Normal', 0.066)],
#  'probabilities': {'Covid-19': 0.627, 'Emphysema': 0.134, 'Normal': 0.066, ...}}
```

Or from the command line:

```bash
python inference.py xray1.jpg xray2.png --repo FarhanKO/lung-disease-prediction
```

`status` is one of:

| Status     | Meaning                                                           |
| ---------- | ----------------------------------------------------------------- |
| `accepted` | Top-class probability ≥ 0.75                                      |
| `review`   | Top-class probability < 0.75: flag for a clinician                |
| `rejected` | The OOD gate says this is not a chest X-ray; it is not classified |

On the full test split, `inference.py` reproduces the repo's PyTorch-backend
results: accuracy 0.8456 on scored images, 1 of 1,737 X-rays rejected by the
gate, and 34 of 34 noise and blank images rejected.

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

# Everything (~700 MB): all seven checkpoints + gate + metadata
snapshot_download("FarhanKO/lung-disease-prediction", local_dir="models")

# Only what the deployed cascade needs (~38 MB)
snapshot_download("FarhanKO/lung-disease-prediction", local_dir="models",
                  allow_patterns=["DenseNet121_FT_best.keras", "lung_ood_detector.joblib", "metadata.json"])
```

Then, from `lung/image_based/`:

```bash
python -m src.predict --image xray.jpg --heatmap-dir out/
python -m src.evaluate --data-dir data      # must reproduce the test metrics in metadata.json
```

## Using a checkpoint directly

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # optional: trained with TensorFlow, runs on either
import keras
model = keras.models.load_model("DenseNet121_FT_best.keras", compile=False)
probs = model.predict(x)   # x: (N, 224, 224, 3) float32 in [0, 1]; classes in metadata.json order
```

**Input preprocessing** (the same as `load_image()` in `src/data.py`):
RGB → resize to **224 × 224** (PIL `BILINEAR`) → float32 → **divide by 255**.

The division is required here. The heart and kidney models from the same
project take raw 0–255 pixels, but these checkpoints have no rescaling layer
and were trained on [0, 1] inputs. Do not add a backbone-specific
`preprocess_input` either.

## Files

| File                                                                           | Size            | What it is                                                                                                   |
| ------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------------ |
| `DenseNet121_FT_best.keras`                                                    | 36 MB           | **Deployed model.** ImageNet DenseNet121, top 30 layers fine-tuned                                           |
| `ResNet50_FT_best.keras`                                                       | 207 MB          | ResNet50, fine-tuned (ensemble member)                                                                       |
| `EfficientNetB0_FT_best.keras`                                                 | 31 MB           | EfficientNetB0, fine-tuned (ensemble member; did not converge, see below)                                    |
| `DenseNet121_best.keras` · `ResNet50_best.keras` · `EfficientNetB0_best.keras` | 31 / 97 / 20 MB | Phase-1 checkpoints (frozen backbone, head only), for comparison                                             |
| `Custom_CNN_best.keras`                                                        | 255 MB          | 3-block CNN trained from scratch (baseline)                                                                  |
| `lung_ood_detector.joblib`                                                     | 1.8 MB          | Stage 1 OOD gate: Isolation Forest on DenseNet121_FT embeddings                                              |
| `metadata.json`                                                                | –               | Class order, input size, preprocessing, deployed model, ensemble members, confidence threshold, test metrics |
| `inference.py`                                                                 | –               | Stand-alone cascade (stages 1–3)                                                                             |
| `requirements.txt`                                                             | –               | Dependencies for `inference.py`                                                                              |

Every `.keras` file takes a `(N, 224, 224, 3)` input in [0, 1] and outputs
softmax probabilities for the 6 classes, in the order `Covid-19, Emphysema,
Normal, Pneumonia-Bacterial, Pneumonia-Viral, Tuberculosis`.

## How the cascade uses them

| Stage               | What                                           | Outcome                                  |
| ------------------- | ---------------------------------------------- | ---------------------------------------- |
| 1. OOD gate         | Isolation Forest on DenseNet121_FT embeddings  | Not a chest X-ray → `rejected`           |
| 2. Classification   | DenseNet121_FT                                 | Probabilities for the 6 classes          |
| 3. Confidence check | Top-class probability < 0.75                   | `review` (flag for a clinician)          |
| 4. Explainability   | Grad-CAM on the top class + top-3 differential | Heatmap and overlay (in the GitHub repo) |

## Training data

18,036 chest X-rays at 224 × 224, one folder per class. The split sizes are
those of the published dataset:

| Class               | Test images                               |
| ------------------- | ----------------------------------------- |
| Covid-19            | 300                                       |
| Emphysema           | 250                                       |
| Normal              | 300                                       |
| Pneumonia-Bacterial | 300                                       |
| Pneumonia-Viral     | 300                                       |
| Tuberculosis        | 287                                       |
| **Split totals**    | **Train 14,551 · Val 1,748 · Test 1,737** |

## Training

- **Framework:** TensorFlow / Keras on Google Colab. Seed 42.
- **Backbones:** ImageNet-pretrained ResNet50, EfficientNetB0 and DenseNet121
  from Keras Applications, plus a small from-scratch CNN.
- **Head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Dense(6,
  softmax).
- **Phase 1:** backbone frozen, Adam at 1e-4, up to 15 epochs.
- **Phase 2 (`_FT`):** the last 30 backbone layers unfrozen, Adam at 1e-5, up
  to 10 epochs.
- **Common to both phases:**
  - Batch size 32, categorical cross-entropy, balanced class weights.
  - EarlyStopping on val loss (patience 8, restores the best weights).
  - ReduceLROnPlateau (factor 0.3, patience 4).
  - The checkpoint with the best val accuracy is kept.
- **Augmentation:** rotation ±15°, zoom ±10 %, horizontal flip, brightness.
- **OOD gate:** Isolation Forest (100 trees, contamination 0.01) fitted on the
  DenseNet121_FT pooled embeddings (1,024-d) of 2,000 un-augmented training
  images.
- **Deployed model:** DenseNet121_FT, which has the highest validation
  accuracy (0.839) as well as the best test scores.

## Results

The held-out test set has 1,737 images.

| Model                                   | Val acc.  | Test acc. | Test macro-F1 | Test macro ROC-AUC | Test log-loss |
| --------------------------------------- | --------- | --------- | ------------- | ------------------ | ------------- |
| Custom CNN                              | 0.762     | 0.794     | 0.789         | 0.968              | 0.513         |
| ResNet50 (fine-tuned)                   | 0.761     | 0.787     | 0.785         | 0.967              | 0.503         |
| EfficientNetB0 (fine-tuned)             | 0.285     | 0.273     | 0.134         | 0.843              | 1.766         |
| **DenseNet121 (fine-tuned) — deployed** | **0.839** | **0.847** | **0.846**     | **0.979**          | **0.384**     |
| Soft-voting ensemble (3 × FT)           | –         | 0.864     | 0.863         | 0.980              | 0.665         |

These are TensorFlow-backend numbers. On the PyTorch backend, 2 of the 1,737
predictions change and accuracy is 0.8457.

Per-class results for the deployed model on the test set:

| Class               | Precision | Recall | F1   | Support |
| ------------------- | --------- | ------ | ---- | ------- |
| Covid-19            | 0.87      | 0.89   | 0.88 | 300     |
| Emphysema           | 0.87      | 0.85   | 0.86 | 250     |
| Normal              | 0.85      | 0.96   | 0.90 | 300     |
| Pneumonia-Bacterial | 0.74      | 0.74   | 0.74 | 300     |
| Pneumonia-Viral     | 0.76      | 0.65   | 0.70 | 300     |
| Tuberculosis        | 1.00      | 0.99   | 0.99 | 287     |

Most errors fall in two pairs: Pneumonia-Bacterial ↔ Pneumonia-Viral (128
swaps) and Covid-19 ↔ Emphysema (63 swaps).

**Confidence check (threshold 0.75):**

- 72 % of test images are accepted, and 93.1 % of those are correct.
- The 28 % flagged for review are 62.6 % correct.

**OOD gate:**

- Rejects 1 of 1,737 real test X-rays (0.06 %).
- Rejects 100 % of random-noise and blank images. Other image types (CT
  slices, ECG printouts, photos) have not been measured against this gate.

**The ensemble is not deployed.** It includes EfficientNetB0_FT, whose output
is close to uniform, so it pulls the average towards 1/6. As a result:

- Its log-loss is worse than DenseNet121 alone (0.67 vs 0.38).
- Its top-class probability almost never reaches 0.75. With `--ensemble`,
  nearly every image is sent to `review`: 0 of 1,737 test images were
  accepted in `inference.py`.

## Limitations

- **EfficientNetB0 did not train.** Keras's EfficientNetB0 rescales its input
  internally and expects 0–255 pixels. It was fed [0, 1] images, so it saw
  near-black inputs and stayed near chance. ResNet50 expects caffe-style
  0–255 input, which also held it back. Retraining each backbone with its own
  input range should fix both.
- **Pneumonia subtypes.** Bacterial vs viral pneumonia is hard to tell from a
  radiograph alone. F1 for those two classes is 0.70–0.74.
- **Split by image, not by patient.** The dataset has no patient IDs, so
  images of the same patient may appear in both train and test. The test
  scores may be optimistic.
- **Possible source shortcuts.** Six-class X-ray datasets are usually pooled
  from several public collections, often one per disease. The model may
  partly learn which collection an image came from (hospital, scanner,
  labels, borders) rather than the disease. In the repo's Grad-CAM samples,
  some of the heat falls outside the lung fields (image edges, diaphragm).
- **Coarse explanations.** DenseNet121's Grad-CAM is a 7 × 7 map. Treat it as
  a sanity check, not as a lesion location.
- **Untested inputs.** Lateral views, paediatric X-rays, phone photos of
  films and DICOM exports with other windowing are untested and may be
  rejected by the gate or misclassified.
