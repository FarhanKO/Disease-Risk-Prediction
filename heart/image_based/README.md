# Heart Disease — Image-Based Module (12-lead ECG)

Classifies a 12-lead ECG printout into four categories: **Abnormal Heartbeat,
History of MI, Myocardial Infarction, Normal**. Part of the
`disease-risk-prediction` monorepo (heart / kidney / lung, each with tabular +
image submodules). Built to mirror `lung/image_based`.

## What it does

- **Cleans the data first.** The published dataset repeats 437 of its 928
  files (Myocardial Infarction: 239 files, 30 distinct ECGs). Duplicates are
  removed before splitting, and the printout header (patient ID, date, heart
  rate) is cropped off so the models can only learn from the waveforms.
- Compares four CNNs: a Custom CNN baseline trained from scratch, and
  ImageNet-pretrained **ResNet50V2, EfficientNetB0, DenseNet121**. Each
  backbone applies its own input scaling inside the model, which fixes the
  EfficientNetB0 failure seen in the lung module.
- Two-phase transfer learning: Phase 1 trains only the head (lr 1e-3);
  Phase 2 unfreezes the last 30 backbone layers, BatchNorm excluded (lr 1e-4).
  Balanced class weights and ECG-safe augmentation (small rotation, zoom,
  shift, contrast; **no flips**, which would produce impossible ECGs).
- Wraps the best model (**fine-tuned ResNet50V2**, chosen on validation
  macro-F1) in a 4-stage clinical cascade:

  | Stage | What | Outcome |
  |---|---|---|
  | 1. OOD gate | Mean cosine distance to the 5 nearest training ECG embeddings | Non-ECG input → `Rejected - Not an ECG` |
  | 2. Classification | ResNet50V2_FT | Probabilities for 4 classes |
  | 3. Confidence check | Top-class probability < 0.75 | `Low Confidence - Manual Review` |
  | 4. Explainability | Grad-CAM on the top class + top-3 differential | Heatmap and overlay |

## Structure

```
image_based/
├── requirements.txt
├── notebook/Heart_ECG.ipynb      # the full pipeline: audit, EDA, training, evaluation, Grad-CAM, cascade
├── src/                          # the notebook's inference code, importable
│   ├── data.py                   # class list, load_image() (crop + resize), split loader
│   ├── predict.py                # the shared predict() / predict_batch() interface (common/image.py)
│   └── evaluate.py               # re-scores the test split, checks it matches the notebook's metadata.json
├── data/
│   ├── build_ecg_dataset.py      # dedupe + crop + stratified split of the Mendeley zip
│   ├── manifest.csv              # every kept image: split, source file, MD5, dropped duplicates
│   └── train/ val/ test/<class>/ # built by the script (not in git)
├── models/                       # *.keras (not in git), heart_ecg_ood_detector.joblib, metadata.json
├── results/                      # model_comparison.csv, classification_report.csv, training_history.json
└── images/                       # every plot the notebook produces
```

## Setup

**Dataset.** *ECG Images dataset of Cardiac Patients*, Khan & Hussain,
Mendeley Data v2, [doi:10.17632/gwbz3fsgp8.2](https://data.mendeley.com/datasets/gwbz3fsgp8/2)
(CC BY 4.0). Download the zip, then build the clean split:

```bash
pip install -r requirements.txt
python data/build_ecg_dataset.py --source path/to/ECG.zip   # also accepts the extracted folder
```

| Class | Raw files | Unique ECGs | Train | Val | Test |
|---|---|---|---|---|---|
| Abnormal_Heartbeat | 233 | 233 | 163 | 35 | 35 |
| History_of_MI | 172 | 86 | 60 | 13 | 13 |
| Myocardial_Infarction | 239 | 30 | 21 | 5 | 4 |
| Normal | 284 | 142 | 99 | 21 | 22 |
| **Total** | **928** | **491** | **343** | **74** | **74** |

**Running the notebook.** It uses Keras 3 on the **PyTorch** backend (set in
its first cell), because TensorFlow has no GPU support on native Windows.
Training all seven checkpoints takes ~15 minutes on an RTX 5060 (8 GB).
Training is resumable: models whose `models/*_best.keras` already exists are
loaded, not retrained. Start Jupyter with `RETRAIN=1` to retrain everything.

## Usage

The notebook trains the models; `src/` serves them. From this folder
(`heart/image_based/`):

```bash
python -m src.predict --image ecg1.png ecg2.jpg --heatmap-dir out/   # prints the cascade result, saves Grad-CAM overlays
python -m src.evaluate            # re-scores the test split: must match the notebook's metadata.json
```

```python
# What Streamlit calls (the same interface as the other five modules; see the repo README)
from src.predict import predict, predict_batch

result = predict("ecg.png")       # also bytes, a file-like upload, a PIL image or an array
result.status        # 'accepted', 'review' (confidence < 0.75) or 'rejected' (not an ECG printout)
result.label         # e.g. 'Myocardial_Infarction'
result.probability   # 1 - P(Normal)
result.details["top_3"], result.explanation["overlay"]   # differential, Grad-CAM overlay (uint8 RGB)
```

`python -m src.evaluate` reproduces the notebook exactly: accuracy 0.8784, macro-F1
0.8501, macro ROC-AUC 0.9746, the same confidence coverage and 0 % of test ECGs
rejected (`--ensemble` also matches).

## Results

Held-out test set (74 ECGs). Full table in `results/model_comparison.csv`.

| Model | Val macro-F1 | Test acc. | Test macro-F1 | Test macro ROC-AUC |
|---|---|---|---|---|
| Custom CNN | 0.395 | 0.689 | 0.388 | 0.853 |
| ResNet50V2 (Phase 1) | 0.586 | 0.622 | 0.589 | 0.938 |
| **ResNet50V2 (fine-tuned)** | **0.805** | **0.878** | **0.850** | **0.975** |
| EfficientNetB0 (fine-tuned) | 0.684 | 0.838 | 0.749 | 0.920 |
| DenseNet121 (Phase 1) | 0.511 | 0.662 | 0.528 | 0.931 |
| Ensemble (ResNet + EffNet + DenseNet, FT) | 0.795 | 0.892 | 0.890 | 0.971 |

- 95% bootstrap CI for ResNet50V2_FT: accuracy 0.80–0.95, macro-F1 0.66–0.95.
  With 4 MI test ECGs, per-class MI numbers are indicative only.
- The ensemble scores higher on test but lower on validation, and its
  probabilities are less calibrated (log-loss 0.53 vs 0.37), so it is not
  deployed. Picking it for its test score would be test-set selection.
- Errors: 4 of 35 Abnormal Heartbeat ECGs are called Normal, and 2 History of
  MI ECGs are called Abnormal Heartbeat. MI and History of MI are never
  confused with each other.

**Cascade behaviour:**
- Stage 1 (kNN gate) rejected 0% of validation/test ECGs and 100% of random
  noise, blank pages, grey speckle, 60 chest X-rays from the lung dataset, and
  line charts from the heart tabular module. The lung module's Isolation
  Forest was also tested: it let 100% of noise and blank pages through.
- Stage 3 at 0.75 accepts 72% of test ECGs, and those are **96.2%** correct.
  The 28% flagged for review are 66.7% correct. 7 of the 9 test errors are
  flagged.

## Notes

- **Preprocessing is `RGB → crop raw printouts to the grid → 448×256 (BOX) → float 0–255`**,
  implemented in `load_image()` in the notebook and in `src/data.py`. Scaling to each backbone's
  range happens inside the saved model, so do not divide by 255 at inference.
  Raw 2213×1572 printouts are cropped to `GRID_BOX = (68, 283, 2177, 1518)`
  automatically.
- **ResNet50V2 instead of ResNet50:** ResNet50 expects BGR, mean-subtracted
  input, and the channel swap would need a `Lambda` layer, which blocks
  safe-mode `.keras` loading. V2's [-1, 1] scaling is a plain `Rescaling` layer.
- **Grad-CAM caveat:** for MI, the heat sits on the ST-elevated chest leads.
  For some Normal and Abnormal Heartbeat ECGs it sits on the right edge of the
  printout, which may be a shortcut. Rule this out before any clinical use.
- **Single source:** all ECGs come from one hospital and one machine template.
  Other printers, paper speeds or phone photos are untested and may be
  rejected by the OOD gate.
