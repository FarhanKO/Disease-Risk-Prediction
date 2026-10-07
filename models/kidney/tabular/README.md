# Kidney Disease (CKD) — Tabular Module

Predicts chronic kidney disease (CKD) from routine clinical data — demographics,
blood pressure, a basic metabolic panel, and diabetes and smoking history —
**without** the tests that define CKD. A positive screen means "order an eGFR
and a urine albumin-to-creatinine ratio (ACR)". Part of the
`disease-risk-prediction` monorepo (heart / kidney / lung, each with tabular +
image submodules).

> **Label fix.** The raw NHANES export's `ckd_present` labelled 5,607
> unknown-status participants (mostly children) and everyone with eGFR 60–89 as
> CKD, and eGFR, creatinine and ACR were model inputs. The earlier ~99 % scores
> therefore measured age and missing labs, not kidney disease. The target is now
> the clinical **KDIGO** definition (eGFR < 60 or ACR ≥ 30) for adults 20+, and
> the five label-defining lab values are excluded. Details: `data/README.md`.

## What it does

- Rebuilds the target (KDIGO), resolves NHANES answer codes and skip-patterns,
  and drops the label-defining labs → **5,473 adults, 19.3 % CKD**.
- Engineers 4 clinical features: `pulse_pressure`, `map` (mean arterial
  pressure), `ca_p_product`, `bmi_bp_interaction`.
- Compares **17 classifiers** — Logistic Regression, Naive Bayes, KNN, Decision
  Tree, Random Forest, XGBoost, LightGBM, SVM, MLP, plus the modern CatBoost,
  Explainable Boosting Machine, TabNet, FT-Transformer, RealMLP, TabM, TabPFN
  and a Stacking Ensemble. They are tuned on 5-fold CV **PR-AUC**, and
  oversampling (SMOTENC) is switched on or off per model by cross-validation.
- Selects with the **one-standard-error rule**: the simplest model within one
  SE of the best CV score → **LightGBM**.
- Calibrates with isotonic regression, picks the decision threshold on
  **out-of-fold training predictions** (cost 5·FN + FP), and reports
  bootstrapped 95 % confidence intervals on the untouched test set.
- Wraps it in a two-stage cascade: an Isolation Forest gate (1 %, fitted on the
  training set) → calibrated LightGBM + SHAP explanation.

## Results

Test set: 1,095 adults, 211 with CKD (full table in `results/model_comparison.csv`).

| Model | CV PR-AUC (± SE) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| TabPFN | 0.632 ± 0.018 | 0.648 | 0.849 |
| **LightGBM** (selected) | 0.629 ± 0.021 | 0.620 | 0.829 |
| Stacking Ensemble | 0.627 ± 0.023 | 0.626 | 0.837 |
| CatBoost | 0.627 ± 0.023 | 0.623 | 0.831 |
| XGBoost | 0.618 ± 0.022 | 0.612 | 0.820 |
| Explainable Boosting | 0.599 ± 0.019 | 0.606 | 0.829 |
| Logistic Regression | 0.591 ± 0.015 | 0.604 | 0.820 |

A random model scores PR-AUC 0.19. The top five are statistically tied, and
LightGBM is the simplest of them.

**Calibrated LightGBM at the cost-optimal threshold 0.175** (chosen on
training data; the Bayes-optimal value for calibrated probabilities is 0.167):

| Metric | Value | 95 % CI |
|---|---|---|
| Recall (CKD caught) | 0.754 | 0.698 – 0.810 |
| Precision | 0.401 | 0.353 – 0.448 |
| ROC-AUC | 0.834 | 0.802 – 0.863 |
| PR-AUC | 0.625 | 0.559 – 0.687 |
| Brier score | 0.110 | reference 0.156 (always predict the prevalence) |

Top drivers (SHAP and permutation importance agree): age, blood urea nitrogen,
bicarbonate, uric acid, systolic blood pressure, diabetes, and education /
income. See `images/` for the comparison, one-SE selection, ROC/PR curves,
calibration, SHAP, clustering, and anomaly-detection plots.

## Structure

```
tabular/
├── requirements.txt
├── data/
│   ├── CKD_NHANES.csv          # raw NHANES 2021–2023 export (public domain)
│   └── README.md               # label bug, KDIGO correction, input columns
├── notebooks/
│   └── Kidney Disease.ipynb    # full analysis: audit, 17 models, calibration, SHAP, clustering, anomalies, cascade
├── src/
│   ├── data.py       # KDIGO label, NHANES cleaning, feature engineering, preprocessing pipeline
│   ├── models.py     # the 17 candidates and their grids (deep models imported only on request)
│   ├── train.py      # selected model (or --compare), calibration, out-of-fold threshold, anomaly gate
│   ├── evaluate.py   # test metrics at the saved threshold, bootstrap CIs, SHAP, permutation importance
│   └── predict.py    # the shared predict() / predict_batch() interface (common/tabular.py)
├── examples/patient.json   # sample input for `python -m src.predict`
├── models/           # kidney_disease_calibrated_model.joblib, kidney_anomaly_gate.joblib, metadata.json,
│                     # reference_profile.json (typical training patient, for the explanations)
├── results/          # model_comparison.csv, permutational_feature_importance.csv
└── images/           # plots saved by the notebook
```

## Usage

Run from this folder (`models/kidney/tabular/`):

```bash
pip install -r requirements.txt

# Re-train the selected model (LightGBM): calibration, threshold, anomaly gate, metadata.json
python -m src.train

# Re-run the full comparison (add --include-deep for the GPU models)
python -m src.train --compare --results-out results/model_comparison.csv

# Evaluate the saved cascade on the test split (updates models/metadata.json)
python -m src.evaluate

# Predict for new patients (JSON object or list)
python -m src.predict --input examples/patient.json
```

```python
# Or import directly (what Streamlit does; the same interface as the other five modules)
from src.predict import predict, predict_batch

result = predict({
    "age": 28, "gender": "Female", "ethnicity": "Non-Hispanic Asian",
    "education_level": 5, "poverty_income_ratio": 4.5, "bmi": 22.0, "height_cm": 165,
    "bp_systolic": 110, "bp_diastolic": 70, "blood_urea_nitrogen": 12,
    "albumin_serum": 4.5, "phosphorus": 3.5, "bicarbonate": 24, "calcium": 9.5,
    "uric_acid": 4.0, "diabetes": "No", "insulin_use": "No", "diabetes_pills": "No",
    "smoking_status": "Never",
})
result.status, result.label, result.probability, result.positive
# ('accepted', 'Low Risk', 0.032, False)
result.explanation["factors"]   # inputs that moved the risk most vs. the typical training patient
predict_batch(patients_df)      # many patients: status, label, probability, positive, message per row
```

## Notes

- **Inputs** are the 19 columns in `src.data.RAW_INPUT_COLUMNS`. `diabetes` is
  Yes / No / Borderline; `insulin_use` and `diabetes_pills` are Yes / No;
  `smoking_status` is Never / Former / Current.
- **Risk bands are anchored to the threshold:** Low < 0.175 ≤ Medium < 0.50 ≤
  High. `positive = True` means "order kidney tests".
- **Stage 1 withholds some real patients.** On the test set the gate withholds
  1.6 % of patients: 6.2 % of CKD patients vs 0.6 % of healthy ones, because
  advanced CKD produces extreme lab values. Withheld patients get
  `status = "review"` with no probability, not a low score. If every plausible
  patient should be scored, replace the gate with physiological range checks.
- **Explanations compare with a typical patient, not with "unknown".** Replacing an
  input with a missing value and letting the pipeline impute it would be simpler,
  but this LightGBM has learned to recognise imputed values (KNN imputation gives
  non-integer education levels, and missing education goes with higher CKD rates):
  a missing education level alone moves a 28-year-old from 3 % to 35 %. So each
  input is swapped for the training median or most common answer instead
  (`models/reference_profile.json`). The same effect means patients with missing
  inputs can score higher than their known values justify.
- **Which artifacts are served:** the notebook's joblib files reference
  functions defined inside the notebook. `src/train.py` re-trains the same
  pipeline with importable code, and the committed `models/` come from it
  (identical CV scores and threshold).
