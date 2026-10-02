# Lung Disease — Image-Based Module (Chest X-ray)

Classifies a chest X-ray into six categories: **Covid-19, Emphysema, Normal,
Pneumonia-Bacterial, Pneumonia-Viral, Tuberculosis**. Part of the
`disease-risk-prediction` monorepo (heart / kidney / lung, each with tabular +
image submodules).

Ported from `notebook/Lung_Xray.ipynb`, refactored into an importable,
CLI-capable package. The notebook is retained for EDA only — `src/` is the
canonical pipeline.

## What it does

- Compares four CNNs: a Custom CNN baseline trained from scratch, and
  ImageNet-pretrained **ResNet50, EfficientNetB0, DenseNet121**.
- Two-phase transfer learning: Phase 1 trains only the new head with the
  backbone frozen (15 epochs, lr 1e-4); Phase 2 unfreezes the last 30 backbone
  layers (10 epochs, lr 1e-5). Balanced class weights and augmentation
  (rotation, zoom, flip, brightness) throughout.
- Wraps the best model (**fine-tuned DenseNet121**) in a 4-stage clinical
  cascade — every image goes through all of it:

  | Stage | What | Outcome |
  |---|---|---|
  | 1. OOD gate | IsolationForest on DenseNet121 embeddings | Non-X-ray input → `Rejected - Not a Chest X-ray` |
  | 2. Classification | DenseNet121_FT (or the soft-voting ensemble) | Probabilities for 6 classes |
  | 3. Confidence check | Top-class probability < 0.75 | `Low Confidence - Manual Review` |
  | 4. Explainability | Grad-CAM on the top class + top-3 differential | Heatmap and overlay |

## Structure

```
image_based/
├── requirements.txt
├── notebook/Lung_Xray.ipynb   # EDA + original experiments — not the canonical pipeline
├── src/
│   ├── data.py       # class list, tf.data loaders, augmentation, class weights, load_image()
│   ├── models.py     # architectures, fine-tune helper, checkpoint registry, embeddings
│   ├── train.py      # Phase 1 + Phase 2 training, fits the Stage 1 OOD gate
│   ├── evaluate.py   # comparison table, report, confusion matrix, ROC, ensemble, Grad-CAM
│   ├── explain.py    # Grad-CAM heatmap + overlay (from common/image.py)
│   └── predict.py    # the shared predict() / predict_batch() interface (common/image.py)
├── models/           # *.keras (not in git), lung_ood_detector.joblib, metadata.json
├── results/          # model_comparison.csv, classification_report.csv
└── images/           # plots (EDA ones extracted from the notebook, the rest from evaluate.py)
```

## Setup

**Checkpoints.** The seven `.keras` files (~700 MB total; the largest are over
GitHub's 100 MB limit) are not in git. Download them from
[Hugging Face](https://huggingface.co/FarhanKO/lung-disease-prediction)
into `models/`:

```
Custom_CNN_best.keras
ResNet50_best.keras        ResNet50_FT_best.keras
EfficientNetB0_best.keras  EfficientNetB0_FT_best.keras
DenseNet121_best.keras     DenseNet121_FT_best.keras
```

Serving needs only `DenseNet121_FT_best.keras` (plus the two other `_FT`
files for `--ensemble`). `lung_ood_detector.joblib` and `metadata.json` are
committed.

**Dataset** (only for training/evaluation). 18,036 images, 224×224, one folder
per class in each split:

```
data/
├── train/<class>/   14,551
├── val/<class>/      1,748
└── test/<class>/     1,737
```

Put it at `data/` or pass `--data-dir` to any command.

## Usage

Run from this folder (`lung/image_based/`):

```bash
pip install -r requirements.txt

# Train all four architectures (Phase 1 + 2), then fit the OOD gate — GPU recommended
python -m src.train --data-dir data

# Refit only the OOD gate from the saved checkpoints (a few minutes on CPU)
python -m src.train --data-dir data --ood-only

# Evaluate — writes results/, images/ and models/metadata.json
python -m src.evaluate --data-dir data

# Predict; --heatmap-dir saves the Grad-CAM overlays
python -m src.predict --image xray1.jpg xray2.png --heatmap-dir out/
```

```python
# Or import directly (what Streamlit does; the same interface as the other five modules)
from src.predict import predict, predict_batch

result = predict("xray.jpg")         # also accepts bytes, file objects, PIL images, arrays
result.status        # 'accepted', 'review' (confidence < 0.75) or 'rejected' (not a chest X-ray)
result.label         # e.g. 'Pneumonia-Viral'
result.probability   # 1 - P(Normal)
result.details       # {'confidence': 0.77, 'probabilities': {...}, 'top_3': [('Pneumonia-Viral', 0.77), ...], ...}
result.explanation   # {'heatmap': <7x7 array>, 'overlay': <224x224x3 uint8>, ...}
predict("xray.jpg", ensemble=True)   # soft-voting ensemble instead of DenseNet121 alone
```

## Results

From `python -m src.evaluate` on the held-out test set (1,737 images). Full
table in `results/model_comparison.csv`.

| Model | Test acc. | Macro F1 | Macro ROC-AUC | Val acc. |
|---|---|---|---|---|
| Custom CNN | 0.794 | 0.789 | 0.968 | 0.762 |
| ResNet50 (fine-tuned) | 0.787 | 0.785 | 0.967 | 0.761 |
| EfficientNetB0 (fine-tuned) | 0.273 | 0.134 | 0.843 | 0.286 |
| **DenseNet121 (fine-tuned)** | **0.847** | **0.846** | **0.979** | **0.839** |
| Ensemble (ResNet + EffNet + DenseNet) | 0.864 | 0.863 | 0.980 | — |

**DenseNet121_FT per class** (`results/classification_report.csv`): Tuberculosis
0.99 F1 and Normal 0.90 are near-solved. The errors cluster in two pairs:
Pneumonia-Bacterial ↔ Pneumonia-Viral (128 swaps) and Covid-19 ↔ Emphysema
(63 swaps). See `images/confusion_matrix.png`.

**Cascade behaviour on the test set:**
- Stage 1 rejected 1 of 1,737 real X-rays (0.06%) and 64 of 64 random-noise
  images.
- Stage 3 at 0.75 accepts 72% of images, and those are **93.1%** correct. The
  28% flagged for review are 62.6% correct, so the threshold does separate
  reliable predictions from shaky ones.

## Notes

- **Preprocessing is `RGB → 224×224 → ÷255`**, the same as the checkpoints were
  trained with. `data.load_image()` is the single place it happens; do not add
  a model-specific `preprocess_input` at inference.
- **Backend.** The checkpoints were trained with TensorFlow (Colab). `src.predict`
  runs them on the PyTorch backend when torch is installed, like the ECG and CT
  modules, so one process can serve all three. On the 1,737 test images the two
  backends disagree on 2 predictions (accuracy 0.8457 on PyTorch, 0.8469 on
  TensorFlow). `KERAS_BACKEND=tensorflow` switches back.
- **`predict` and `evaluate` decode JPEGs differently.** `load_image()` decodes with
  PIL, as the notebook's training generator did; `evaluate.py` uses
  `image_dataset_from_directory` (TensorFlow's decoder). The pixel differences move
  the headline numbers slightly: through `predict`, DenseNet121_FT scores accuracy
  0.8457–0.8469 and log-loss 0.383 (vs 0.8469 / 0.384 in the table above).
- **Why EfficientNetB0 failed.** Keras's EfficientNetB0 already contains
  `Rescaling(1/255)` + normalization layers and expects 0–255 input. The
  notebook divided by 255 first, so EfficientNet saw near-black images and
  stayed at ~17% (chance for 6 classes). ResNet50 expects caffe-style
  0–255 mean-subtracted input, which is why its Phase 1 was slow too.
  Retraining each backbone with its own expected input range should fix both.
  That means different preprocessing per model, which `load_image()` would
  then need to handle.
- **Differences from the notebook:** Keras 3 removed `ImageDataGenerator`, so
  loading uses `image_dataset_from_directory` and augmentation layers. Grad-CAM
  runs the model layer by layer, because the notebook's `layer.output_shape`
  lookup fails in Keras 3. The OOD gate is fitted on un-augmented training
  images and saved to disk; the notebook fitted it in memory only.
- The ensemble includes EfficientNetB0 (as the notebook did) despite its
  accuracy — see above. Its near-uniform outputs also make the ensemble's
  probabilities less calibrated (test log-loss 0.67 vs DenseNet's 0.38), so
  predict.py defaults to DenseNet121 alone.
- Grad-CAM on DenseNet121 is a coarse 7×7 map. In `images/gradcam_samples.png`
  some of the heat lands outside the lung fields (image edges, diaphragm), so
  treat the overlay as a sanity check, not a lesion location.
