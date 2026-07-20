# Kidney Disease (CKD) — Tabular Module

Predicts presence of chronic kidney disease (CKD) from clinical/tabular patient
data (NHANES-derived). Part of the `disease-risk-prediction` monorepo (heart /
kidney / lung, each with tabular + image + fusion submodules, orchestrated by a
top-level cascade).

Ported from `Kidney_Disease.ipynb`, refactored into an importable, CLI-capable
package. `src/` is the canonical pipeline.

## What it does

- Engineers 5 clinical features on top of the raw NHANES columns:
  `pulse_pressure`, `map` (mean arterial pressure), `bun_cr_ratio`,
  `ca_p_product`, `bmi_bp_interaction`.
- Compares 9 classifiers (Logistic Regression, Naive Bayes, KNN, Decision Tree,
  Random Forest, XGBoost, LightGBM, SVM, Neural Network) via grid search,
  selecting on **recall first, precision second** — missing a real CKD case is
  treated as far costlier than a false alarm.
- Balances classes with SMOTE, calibrates the winning model with isotonic
  regression (`CalibratedClassifierCV`) so output probabilities are
  trustworthy, not just ranks.
- Picks a decision threshold via a clinical cost function (false negatives
  weighted 5x false positives), not a flat 0.5 cutoff.
- Validates with bootstrapped confidence intervals, Brier score, SHAP, and
  permutation importance.

**Adapted from the notebook, not a literal port:**
- The notebook trained on RAPIDS cuML / GPU XGBoost / GPU LightGBM (Colab
  GPU-runtime only). `train.py` uses their CPU sklearn/xgboost/lightgbm
  equivalents instead, so it runs anywhere.
- The notebook's "Decision Tree" was actually a single-tree GPU XGBoost
  workaround (cuML has no decision tree). `train.py` uses a real
  `DecisionTreeClassifier` instead.
- The notebook dynamically dropped any feature >0.85 correlated with another;
  on the reference NHANES export that dropped exactly `weight_kg`. `data.py`
  bakes that in as a static drop (see the docstring in `data.py`) so the
  feature schema stays stable across retrains, rather than being recomputed
  — and possibly changing — every run.
- SHAP in `evaluate.py` tries `TreeExplainer` first and falls back to a
  generic `shap.Explainer` if the winning model isn't tree-based (unlike
  heart, kidney's winner isn't guaranteed to be a tree model).

`ckd_stage` (multi-class staging) is dropped along with the target — this
module predicts binary `ckd_present` only. Stage-level prediction isn't
currently part of the pipeline.

## Structure

```
tabuler/
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py          # loading, cleaning, feature engineering, preprocessing pipeline
│   ├── train.py         # 9-model grid search, calibration, saves the model artifact
│   ├── evaluate.py      # metrics, threshold optimization, SHAP, permutation importance
│   └── predict.py       # stable inference interface — what Streamlit/cascade import
└── models/              # saved .joblib artifact + metadata.json
```

No `data/` or `notebook/` folder yet — add the raw CSV under `data/` before
running `train.py`, and drop a trimmed EDA-only notebook in here if you want
one alongside heart's.

## Usage

```bash
pip install -r requirements.txt

# Train and calibrate
python -m src.train --data-path data/kidney_disease.csv \
    --model-out models/kidney_tabular_calibrated.joblib

# Evaluate — also writes models/metadata.json with the optimal threshold
python -m src.evaluate --data-path data/kidney_disease.csv \
    --model-path models/kidney_tabular_calibrated.joblib

# Predict on new patients
python -m src.predict --input patient.json \
    --model-path models/kidney_tabular_calibrated.joblib
```

```python
# Or import directly (what Streamlit does)
from src.predict import predict_single

result = predict_single({
    "age": 28, "gender": "Female", "ethnicity": "Non-Hispanic Asian",
    "education_level": 5.0, "poverty_income_ratio": 4.5, "bmi": 22.0,
    "height_cm": 165.0, "bp_systolic": 110.0, "bp_diastolic": 70.0,
    "serum_creatinine": 0.8, "blood_urea_nitrogen": 12.0, "albumin_serum": 4.5,
    "phosphorus": 3.5, "bicarbonate": 24.0, "calcium": 9.5, "uric_acid": 4.0,
    "urine_creatinine": 120.0, "urine_albumin": 10.0,
    "albumin_creatinine_ratio": 8.0, "diabetes_diagnosed": 0.0,
    "insulin_use": 0.0, "diabetes_pills": 0.0, "ever_smoked": 0.0,
    "current_smoker": 0.0, "egfr": 110.0,
})
# {'ckd_probability': 0.04, 'prediction': 0, 'risk_level': 'Low Risk'}
```

## Notes

- `predict.py` only needs raw clinical columns — feature engineering and
  preprocessing happen inside the saved pipeline itself.
- Risk bands: <0.20 Low, <0.50 Medium, ≥0.50 High (fixed, independent of the
  cost-optimized decision threshold used for the binary `prediction` field).
- No anomaly-detection gate here — that's a Stage 1 concern for the top-level
  `cascade/` module, not this module's stable interface.
- **Current artifact naming mismatch:** the committed model is
  `models/kidney_disease_calibrated_model.joblib` (the notebook's original
  export name), but `predict.py`'s default path is
  `models/kidney_tabular_calibrated.joblib`. Either rename the file to match,
  or pass `--model-path models/kidney_disease_calibrated_model.joblib`
  explicitly until you retrain with `train.py`'s default naming.
- No `metadata.json` yet either, since `evaluate.py` hasn't been run against
  the committed artifact — until it is, `predict.py` falls back to a flat 0.5
  threshold rather than the clinical cost-optimal one.
