**See the models here:** https://huggingface.co/FarhanKO/heart-disease-prediction

# 12-Lead ECG Classifier (ResNet50V2)

Classifies a scanned 12-lead ECG printout into one of four classes:
**Abnormal_Heartbeat, History_of_MI, Myocardial_Infarction, Normal**.

The model runs as a cascade:

1. An out-of-distribution (OOD) gate rejects images that are not ECG printouts.
2. The classifier scores the four classes.
3. A confidence check sends uncertain cases to manual review.
4. In the full pipeline, Grad-CAM shows which part of the trace drove the answer.

These models are the image half of the heart module in
[Disease-Risk-Prediction](https://github.com/FarhanKO/Disease-Risk-Prediction).

> **Research and education only.** This is not a medical device. It was trained on
> 491 ECGs from a single hospital and must not be used for diagnosis or treatment
> decisions.

## Quick start

```bash
pip install -r requirements.txt     # keras 3, torch, numpy, pillow, huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util

REPO_ID = "<your-hf-username>/heart-ecg-classifier"
spec = importlib.util.spec_from_file_location("inference", hf_hub_download(REPO_ID, "inference.py"))
inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)

clf = inference.ECGClassifier(REPO_ID)          # downloads metadata, OOD gate and ResNet50V2_FT (~210 MB)
clf.predict("ecg.png")
# {'status': 'accepted', 'label': 'Normal', 'confidence': 0.985, 'p_abnormal': 0.015,
#  'probabilities': {'Abnormal_Heartbeat': 0.0003, 'History_of_MI': 0.014, 'Myocardial_Infarction': 0.001, 'Normal': 0.985}}
```

The command line works too, either from a clone of this repo or with `--repo`:

```bash
python inference.py ecg1.png ecg2.jpg
python inference.py ecg.png --repo <your-hf-username>/heart-ecg-classifier --ensemble
```

```text
MI(12).png: review | Normal (51.5%)
   Normal                    51.5%
   Myocardial_Infarction     48.3%
   Abnormal_Heartbeat         0.1%
   History_of_MI              0.1%
Normal(104).png: accepted | Normal (98.5%)
   Normal                    98.5%
   History_of_MI              1.4%
   Myocardial_Infarction      0.1%
   Abnormal_Heartbeat         0.0%
```

The first ECG above is a real infarction that the model is unsure about. It is
sent to review rather than being reported as Normal, which is the job of the
confidence check.

`status` is one of:

| Status | Meaning |
|---|---|
| `accepted` | Top-class probability ≥ 0.75 |
| `review` | Top-class probability < 0.75: flag for a clinician |
| `rejected` | The OOD gate says this is not a 12-lead ECG printout; it is not classified |

### Input

- **Accepted formats:** an RGB image of the ECG grid. That can be a crop like
  the training data (about 1.7 : 1), or a full 2213 × 1572 printout from the
  source dataset, which is cropped to the grid automatically.
- **Preprocessing:** RGB → crop (full printouts only) → resize to **448 × 256**
  (width × height) with PIL `BOX` resampling → float32 in **0–255**.
- **Do not divide by 255.** Each model has its own rescaling layer built in.

If you load a `.keras` file directly instead of using `inference.py`:

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # before importing keras
import keras
model = keras.models.load_model("ResNet50V2_FT_best.keras", compile=False)
probs = model.predict(x)   # x: (N, 256, 448, 3) float32, 0-255; classes in metadata.json order
```

## Files

| File | Size | What it is |
|---|---|---|
| `ResNet50V2_FT_best.keras` | 207 MB | **Deployed model.** ImageNet ResNet50V2, fine-tuned |
| `EfficientNetB0_FT_best.keras` | 31 MB | EfficientNetB0, fine-tuned (ensemble member) |
| `DenseNet121_FT_best.keras` | 36 MB | DenseNet121, fine-tuned (ensemble member) |
| `ResNet50V2_best.keras` · `EfficientNetB0_best.keras` · `DenseNet121_best.keras` | 97 / 20 / 31 MB | Phase-1 checkpoints (frozen backbone, head only), for comparison |
| `Custom_CNN_best.keras` | 5 MB | 4-block CNN trained from scratch (baseline) |
| `ood_gate.npz` | 1.2 MB | OOD gate: unit-normalised ResNet50V2_FT embeddings of the 343 training ECGs, `k = 5`, `threshold` |
| `metadata.json` | – | Class order, input size, crop box, confidence threshold, deployed model, ensemble members, test metrics |
| `inference.py` | – | Stand-alone cascade (stages 1–3) |
| `requirements.txt` | – | Dependencies for `inference.py` |

Every `.keras` file takes a `(N, 256, 448, 3)` input and outputs softmax
probabilities for the 4 classes, in the order listed in `metadata.json`.

## Training data

*ECG Images dataset of Cardiac Patients*, by Ali Haider Khan and Muzammil
Hussain (Mendeley Data v2, 2021,
[doi:10.17632/gwbz3fsgp8.2](https://data.mendeley.com/datasets/gwbz3fsgp8/2),
CC BY 4.0). The ECGs were collected at the Ch. Pervaiz Elahi Institute of
Cardiology, Multan, Pakistan.

The published dataset has 928 files, but only 491 distinct ECGs. The
Myocardial Infarction class has 239 files covering just 30 ECGs, and every
Normal and History-of-MI ECG appears twice. Exact duplicates were removed by
MD5 hash **before** splitting. Each printout was then cropped to the ECG grid,
which removes the patient ID, date and heart-rate text a CNN could use as a
shortcut. The data was split with stratification, seed 42:

| Class | Unique ECGs | Train | Val | Test |
|---|---|---|---|---|
| Abnormal_Heartbeat | 233 | 163 | 35 | 35 |
| History_of_MI | 86 | 60 | 13 | 13 |
| Myocardial_Infarction | 30 | 21 | 5 | 4 |
| Normal | 142 | 99 | 21 | 22 |
| **Total** | **491** | **343** | **74** | **74** |

The build script and a manifest (every file, its split and the duplicates
dropped) are in the GitHub repo under `heart/image_based/data/`.

## Training

- **Framework:** Keras 3.14 on the PyTorch 2.11 backend, trained on an
  RTX 5060 (8 GB). Seed 42.
- **Head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Dense(4,
  softmax). The backbones are ImageNet-pretrained ResNet50V2, EfficientNetB0
  and DenseNet121 from Keras Applications.
- **Phase 1:** the backbone is frozen. Adam at 1e-3, up to 40 epochs.
- **Phase 2 (`_FT`):** the last 30 backbone layers are unfrozen, with
  BatchNorm kept frozen. Adam at 1e-4, up to 25 epochs.
- **Common to both phases:**
  - Batch size 8, categorical cross-entropy, balanced class weights.
  - EarlyStopping on val loss (patience 10, restores the best weights).
  - ReduceLROnPlateau (factor 0.3, patience 4).
  - The checkpoint with the best val loss is kept.
- **Augmentation:** rotation ±0.005 turns (about ±2°), zoom ±5 %,
  translation ±3 %, contrast ±0.2, with white fill. **No flips**, because a
  mirrored ECG is physiologically impossible.
- **Model selection:** the deployed model has the highest **validation**
  macro-F1, with the lowest validation loss as a tie-breaker. The test set was
  not used for selection.
- **Confidence threshold:** 0.75.
- **OOD gate threshold:** each training ECG is scored by its mean distance to
  its 5 nearest other training ECGs (leave-one-out). The threshold is the 99th
  percentile of those scores.

## Results

The held-out test set has 74 ECGs.

| Model | Val macro-F1 | Test acc. | Test macro-F1 | Test macro ROC-AUC | Test log-loss |
|---|---|---|---|---|---|
| Custom CNN | 0.395 | 0.689 | 0.388 | 0.853 | 0.843 |
| ResNet50V2 (phase 1) | 0.586 | 0.622 | 0.589 | 0.938 | 0.878 |
| **ResNet50V2 (fine-tuned) — deployed** | **0.805** | **0.878** | **0.850** | **0.975** | **0.373** |
| EfficientNetB0 (phase 1) | 0.442 | 0.757 | 0.551 | 0.869 | 0.883 |
| EfficientNetB0 (fine-tuned) | 0.684 | 0.838 | 0.749 | 0.920 | 0.601 |
| DenseNet121 (phase 1) | 0.511 | 0.662 | 0.528 | 0.931 | 0.843 |
| DenseNet121 (fine-tuned) | 0.541 | 0.649 | 0.597 | 0.927 | 0.961 |
| Soft-voting ensemble (3 × FT) | 0.795 | 0.892 | 0.890 | 0.971 | 0.527 |

The 95 % bootstrap confidence intervals for the deployed model are 0.80–0.95
for accuracy and 0.66–0.95 for macro-F1.

The ensemble scores higher on test but lower on validation, and its
probabilities are less well calibrated. It is therefore available through
`--ensemble` but is not the default. Choosing it for its test score would mean
selecting on the test set.

Per-class results for the deployed model on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Abnormal_Heartbeat | 0.94 | 0.86 | 0.90 | 35 |
| History_of_MI | 0.92 | 0.85 | 0.88 | 13 |
| Myocardial_Infarction | 0.75 | 0.75 | 0.75 | 4 |
| Normal | 0.81 | 0.95 | 0.88 | 22 |

**Confidence check (threshold 0.75):**
- 72 % of test ECGs are accepted, and 96.2 % of those are correct.
- The 28 % flagged for review are 66.7 % correct.
- 7 of the 9 test errors are flagged.

**OOD gate:**
- Rejects 0 % of the validation and test ECGs.
- Rejects 100 % of random noise, blank pages, grey speckle, 60 chest X-rays and
  line-chart images.

`inference.py` reproduces the test accuracy exactly: 0.8784 for the single
model and 0.8919 for the ensemble.
