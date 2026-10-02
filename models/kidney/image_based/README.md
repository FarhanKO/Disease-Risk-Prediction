# Kidney Disease — Image-Based Module (CT)

Classifies a kidney CT slice into four categories: **Cyst, Normal, Stone,
Tumor**. Part of the `disease-risk-prediction` monorepo (heart / kidney / lung,
each with tabular + image submodules). Structured like `lung/image_based`.

## What it does

- **Evaluates on unseen scans.** The 12,446 images are consecutive slices of
  only 189 CT scans, and neighbouring slices are near-identical. The dataset
  build removes 517 exact duplicates and splits by **scan series**, so no scan
  is in more than one split. A memorising 1-NN classifier scores 100% on a
  random image split and 45% on this split: random splits (common in
  published work on this dataset) mostly measure memorisation.
- Compares five CNNs: a Custom CNN baseline and ImageNet-pretrained
  **ResNet50V2, EfficientNetV2B0, DenseNet121, ConvNeXtTiny**. Each backbone
  applies its own input scaling inside the model.
- Two-phase training: a 5-epoch head warm-up on the frozen backbone, then
  **full fine-tuning** of every layer (BatchNorm frozen) with AdamW (lr 3e-5,
  weight decay 0.05), label smoothing 0.1 and strong CT-safe augmentation (no
  flips). The lung recipe (frozen head + top-30-layer fine-tune) was tried
  first and overfitted to the training scans: best validation macro-F1 0.59.
  Full fine-tuning was chosen in validation-only experiments.
- Wraps the best model (**fine-tuned DenseNet121**, chosen on validation
  macro-F1) in a 4-stage clinical cascade:

  | Stage | What | Outcome |
  |---|---|---|
  | 1. OOD gate | Isolation Forest on DenseNet121 embeddings (chosen over a kNN gate by a validation rule) | Non-CT input → `Rejected - Not a kidney CT` |
  | 2. Classification | DenseNet121_FT | Probabilities for 4 classes |
  | 3. Confidence check | Top-class probability < 0.75 | `Low Confidence - Manual Review` |
  | 4. Explainability | Grad-CAM on the top class + top-3 differential | Heatmap and overlay |

## Structure

```
image_based/
├── requirements.txt
├── notebook/Kidney_CT.ipynb        # the full pipeline: leakage audit, EDA, training, evaluation, Grad-CAM, cascade
├── src/                            # the notebook's inference code, importable
│   ├── data.py                     # class list, load_image() (grayscale + pad + resize), split loader
│   ├── predict.py                  # the shared predict() / predict_batch() interface (common/image.py)
│   └── evaluate.py                 # re-scores the test split, checks it matches the notebook's metadata.json
├── data/
│   ├── build_kidney_dataset.py     # dedupe + scan-series grouping + series-level split + 256 px PNGs
│   ├── manifest.csv                # every kept slice: split, series_id, original ID/size, MD5, dropped duplicates
│   ├── raw/                        # the 5 source parquet files (not in git)
│   └── train/ val/ test/<class>/   # built by the script (not in git)
├── models/                         # *.keras (not in git), kidney_ct_ood_detector.joblib, metadata.json
├── results/                        # model_comparison.csv, classification_report.csv, training_history.json
└── images/                         # every plot the notebook produces
```

## Setup

**Dataset.** *CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone* (Islam et al.,
*Scientific Reports* 2022; Kaggle `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`).
The full-resolution originals with their file names are mirrored on Hugging
Face as `ryfkn/CT-Kidney-Dataset-{Cyst,Normal,Stone,Tumor}` (~1.6 GB). Save
each class's parquet file(s) as `data/raw/<Class>__<file>.parquet`, then:

```bash
pip install -r requirements.txt
python data/build_kidney_dataset.py
```

| Class | Raw files | Unique slices | Scan series | Train (scans) | Val (scans) | Test (scans) |
|---|---|---|---|---|---|---|
| Cyst | 3,709 | 3,284 | 64 | 2,298 (38) | 493 (14) | 493 (12) |
| Normal | 5,077 | 5,002 | 47 | 3,503 (32) | 750 (7) | 749 (8) |
| Stone | 1,377 | 1,360 | 55 | 952 (38) | 204 (5) | 204 (12) |
| Tumor | 2,283 | 2,283 | 23 | 1,600 (15) | 342 (4) | 341 (4) |
| **Total** | **12,446** | **11,929** | **189** | **8,353 (123)** | **1,789 (30)** | **1,787 (36)** |

**Running the notebook.** It uses Keras 3 on the **PyTorch** backend (set in
its first cell), because TensorFlow has no GPU support on native Windows.
Training all nine checkpoints takes ~1 h 50 min on an RTX 5060 (8 GB).
Training is resumable: models whose `models/*_best.keras` already exists are
loaded, not retrained. Start Jupyter with `RETRAIN=1` to retrain everything.

## Usage

The notebook trains the models; `src/` serves them. From this folder
(`kidney/image_based/`):

```bash
python -m src.predict --image slice1.png slice2.jpg --heatmap-dir out/   # prints the cascade result, saves Grad-CAM overlays
python -m src.evaluate            # re-scores the test split: must match the notebook's metadata.json
```

```python
# What Streamlit calls (the same interface as the other five modules; see the repo README)
from src.predict import predict, predict_batch

result = predict("slice.png")     # also bytes, a file-like upload, a PIL image or an array
result.status        # 'accepted', 'review' (confidence < 0.75) or 'rejected' (not a kidney CT slice)
result.label         # e.g. 'Stone'
result.probability   # 1 - P(Normal)
result.details["top_3"], result.explanation["overlay"]   # differential, Grad-CAM overlay (uint8 RGB)
```

`python -m src.evaluate` reproduces the notebook: accuracy 0.8942, macro-F1 0.8670,
macro ROC-AUC 0.9759, the same confidence coverage, and 3.64 % of test slices
rejected by the gate (the notebook's 3.6 %, rounded). `--ensemble` also matches.

## Results

Held-out test set: 1,787 slices from 36 scans that were never seen in
training. Full table in `results/model_comparison.csv`.

| Model | Val macro-F1 | Test acc. | Test macro-F1 | Test macro ROC-AUC |
|---|---|---|---|---|
| Custom CNN | 0.815 | 0.730 | 0.697 | 0.928 |
| ResNet50V2 (fine-tuned) | 0.844 | 0.868 | 0.826 | 0.973 |
| EfficientNetV2B0 (fine-tuned) | 0.817 | 0.806 | 0.737 | 0.965 |
| **DenseNet121 (fine-tuned)** | **0.889** | **0.894** | **0.867** | **0.976** |
| ConvNeXtTiny (fine-tuned) | 0.794 | 0.851 | 0.798 | 0.968 |
| Ensemble (4 fine-tuned) | 0.868 | 0.877 | 0.841 | 0.983 |
| *Frozen-backbone (Phase 1) models* | *0.53–0.71* | *0.56–0.64* | *0.47–0.53* | *0.76–0.87* |

- Scan-level 95% bootstrap CI for DenseNet121_FT: accuracy 0.82–0.96,
  macro-F1 0.70–0.91.
- Per class: Cyst recall 0.99, Normal 0.98, Stone 0.77, **Tumor 0.66**. Tumor
  errors concentrate in 1–2 of the only 4 tumor test scans; one stone scan is
  missed completely.
- The ensemble is not deployed: it is worse on validation (0.868 vs 0.889) and
  its tumor recall drops to 0.50.

**Cascade behaviour:**
- Stage 1: the Isolation Forest rejects 3.6% of real test slices and 100% of
  random noise, black images, grey speckle, 60 chest X-rays (lung dataset) and
  60 ECG printouts (heart module). The kNN gate from the heart module rejected
  19% of real test slices, so the selection rule picked the Isolation Forest.
- Stage 3 at 0.75 accepts 81% of test slices, and those are **96.3%**
  correct. The 19% flagged for review are 59.6% correct.

## Notes

- **Preprocessing is `grayscale → pad to square (black) → 224×224 (BOX) → 3 channels → float 0–255`**,
  implemented in `load_image()` in the notebook and in `src/data.py`. Scaling to each backbone's
  range happens inside the saved model, so do not divide by 255 at inference.
- **Acquisition shortcut risk:** body-pixel intensity differs by class (Cyst
  darker, Normal and Tumor brighter), which reflects contrast phase and
  display window rather than disease. Grad-CAM often spreads over the central
  abdomen and spine, not only the kidneys, so part of the signal may be
  "how the scan was taken".
- **Single-label data:** some slices show more than one finding (e.g. a stone
  next to a large low-density mass), but each image carries one label.
- **Source limits:** all scans come from hospitals in Dhaka, exported as
  windowed JPEGs, not raw HU DICOMs. Other scanners and windows are untested.
- **Speed on the PyTorch backend:** Keras's `RandomRotation` / `RandomZoom` /
  `RandomTranslation` take ~96 ms per batch there, so the notebook augments
  with one `torch` `affine_grid` / `grid_sample` call (~1 ms) instead.
