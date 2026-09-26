# Heart Disease — Tabular Module

Predicts the presence of heart disease (coronary heart disease, angina, heart
attack or congestive heart failure) from demographic, vital-sign, laboratory,
lifestyle and chest-pain data. Part of the `disease-risk-prediction` monorepo
(heart / kidney / lung, each with tabular and image submodules).

## Dataset

`data/HEART_NHANES.csv`: **24,454 examined adults aged 40+** from six CDC NHANES
survey cycles (2007 to March 2020), 31 features, **12.5 % heart disease
prevalence**. It replaces the 920-patient UCI heart disease file, which was too
small and had `ca`, `thal` and `slope` 34–66 % missing.

The full data dictionary, the NHANES variables behind every column and the
caveats (self-reported diagnosis, cross-sectional design) are in
[`data/README.md`](data/README.md). The CSV is rebuilt from the CDC servers with:

```bash
python data/build_heart_nhanes.py      # from heart/tabular/; caches raw .xpt files in data/raw/
```

## What the notebook does

`notebook/Heart_Diseases.ipynb` is the full pipeline, cell for cell the same
workflow as the kidney and lung notebooks:

- **7 engineered features:** pulse pressure, non-HDL cholesterol, total/HDL
  ratio, waist-to-height ratio, eGFR (CKD-EPI 2021), log urine
  albumin-to-creatinine ratio and a count of classic risk factors.
- **Leakage-aware preprocessing:** scale → KNN-impute → SMOTENC (optional) →
  one-hot, all inside the cross-validation pipeline.
- **17 classifiers:** 9 baselines (Logistic Regression, Naive Bayes, KNN,
  Decision Tree, Random Forest, XGBoost, LightGBM, SVM, MLP) and 8 modern models
  (CatBoost, Explainable Boosting Machine, TabNet, FT-Transformer, RealMLP, TabM,
  TabPFN, Stacking Ensemble). Each is tuned on 5-fold cross-validated PR-AUC,
  and the grid also decides whether SMOTENC oversampling helps.
- **Model selection** by the one-standard-error rule: the simplest model whose
  CV PR-AUC is within one standard error of the best.
- **Isotonic calibration** and a **cost-based threshold** (a missed case costs
  5× a false alarm) chosen on out-of-fold training predictions, never on the
  test set.
- Bootstrapped 95 % confidence intervals, Brier score, SHAP and permutation
  importance.
- PCA / t-SNE, DBSCAN, GMM, K-Means, six anomaly detectors, and a two-stage
  cascade: an Isolation Forest gate (1 % contamination) followed by the
  calibrated model with SHAP explanations.

## Results

Test set: 4,891 patients never used for training or tuning (610 with heart disease).

| Model | CV PR-AUC (± SE) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| TabPFN (GPU, benchmark only) | 0.579 ± 0.011 | 0.563 | 0.886 |
| Stacking Ensemble | 0.575 ± 0.011 | 0.563 | 0.886 |
| TabM (GPU, benchmark only) | 0.572 ± 0.011 | 0.559 | 0.885 |
| CatBoost | 0.572 ± 0.011 | 0.564 | 0.884 |
| Explainable Boosting | 0.569 ± 0.010 | 0.560 | 0.885 |
| **Logistic Regression (selected)** | **0.567 ± 0.012** | **0.555** | **0.880** |
| Random model | — | 0.125 | 0.500 |

All 17 models are in [`results/model_comparison.csv`](results/model_comparison.csv).
The top ten are within about one standard error of each other, so the one-SE
rule picks the simplest deployable model: **logistic regression**. SMOTE was
switched off by cross-validation for every model.

Deployed model (calibrated logistic regression at the cost-optimal threshold
**0.200**), on the test set with 95 % bootstrap intervals:

| Metric | Value |
|---|---|
| Recall (sensitivity) | 0.732 (0.696–0.767) |
| Precision | 0.416 (0.386–0.446) |
| Specificity | 0.854 |
| ROC-AUC | 0.880 (0.866–0.893) |
| PR-AUC | 0.555 (0.515–0.597) |
| Brier score | 0.077 (0.109 for always predicting the prevalence) |

Top permutation importances: age, chest-pain type, severe chest pain, total
cholesterol, self-rated health, urine albumin and sex. Measured cholesterol and
blood pressure are *lower* in diagnosed patients (statins and blood-pressure
drugs), and the model reflects that. See the notebook's SHAP section and the
data caveats.

## Structure

```
tabular/
├── data/
│   ├── build_heart_nhanes.py   # downloads NHANES and builds the CSV
│   ├── HEART_NHANES.csv
│   ├── README.md               # data dictionary
│   └── raw/                    # cached NHANES .xpt files (gitignored)
├── notebook/
│   └── Heart_Diseases.ipynb    # training, evaluation, explainability, cascade
├── models/
│   ├── heart_disease_calibrated_model.joblib   # Stage 2: calibrated classifier
│   ├── heart_disease_anomaly_gate.joblib       # Stage 1: Isolation Forest gate
│   ├── metadata.json                           # threshold, features, test metrics
│   └── reference_profile.json                  # typical training patient, for the explanations
├── results/                    # model_comparison.csv, permutational_feature_importance.csv
├── images/                     # plots saved by the notebook
├── src/
│   ├── data.py                 # schema, feature engineering, preprocessing pipeline, anomaly gate
│   ├── train.py                # tune -> one-SE selection -> calibrate -> OOF threshold -> save artifacts
│   ├── evaluate.py             # test metrics, bootstrap CIs, Brier, grouped SHAP, permutation importance
│   └── predict.py              # the shared predict() / predict_batch() interface (common/tabular.py)
├── examples/patient.json      # sample input for `python -m src.predict`
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python data/build_heart_nhanes.py        # rebuild the dataset from CDC NHANES

# Full analysis: 17 models, plots, observations (several hours, GPU recommended)
jupyter nbconvert --to notebook --execute --inplace notebook/Heart_Diseases.ipynb

# Or just the deployed model, from heart/tabular/ (minutes, CPU only)
python -m src.train                      # add --compare to also tune LightGBM / XGBoost / CatBoost / EBM
python -m src.evaluate                   # adds test metrics to models/metadata.json
python -m src.predict --input examples/patient.json
```

Set the `N_JOBS` environment variable (notebook) or `--n-jobs` (scripts) to limit
the CPU workers the grid searches use (default: all cores but two).

```python
# What Streamlit calls: raw clinical columns in, cascade decision out
# (the same interface as the other five modules; see the repo README)
from src.predict import predict, predict_batch

result = predict({
    "age": 67, "gender": "Male", "ethnicity": "Non-Hispanic White", "education_level": 2,
    "poverty_income_ratio": 1.3, "bmi": 31.5, "waist_cm": 112, "height_cm": 175,
    "systolic_bp": 158, "diastolic_bp": 84, "resting_heart_rate": 82,
    "total_cholesterol": 240, "hdl_cholesterol": 34, "hba1c": 7.9, "serum_creatinine": 1.4,
    "uric_acid": 7.6, "albumin_creatinine_ratio": 85, "white_blood_cells": 8.9,
    "hemoglobin": 13.4, "rdw": 14.8, "smoking_status": "Current", "physically_active": "No",
    "hypertension_history": "Yes", "high_cholesterol_history": "Yes", "diabetes_status": "Yes",
    "stroke_history": "No", "family_history_chd": "Yes", "general_health": 4,
    "chest_pain_type": "Exertional", "severe_chest_pain": "Yes", "sob_on_exertion": "Yes",
})
result.status        # 'accepted'  ('review' if the anomaly gate withholds the profile)
result.label         # 'High Risk'
result.probability   # 0.924
result.positive      # True: at or above the cost-optimal threshold (0.200)
result.message       # 'High Risk (92%): refer for cardiology work-up (ECG, stress test)'
result.explanation["factors"][:2]
# [{'feature': 'chest_pain_type', 'value': 'Exertional', 'typical': 'No pain', 'effect': 0.256},
#  {'feature': 'severe_chest_pain', 'value': 'Yes', 'typical': 'No', 'effect': 0.180}]

predict_batch(patients_df)   # many patients: status, label, probability, positive, message per row
```

The explanation swaps one input at a time for the typical training patient's value
(`models/reference_profile.json`: medians and most common answers) and reports how
much the probability changes; positive effects raised this patient's risk.

## Notes

- `src/` and the notebook build identical pipelines (same features, preprocessing,
  grids, seeds and threshold rule), so `python -m src.train` reproduces the
  notebook's deployed model: same CV score, threshold (0.2003) and test metrics,
  with individual probabilities within 0.003. The notebook-saved pipelines reference
  `add_custom_features` from the notebook; `common/artifacts.py` points that name at
  `src.data.add_custom_features` while loading, so both kinds of artifacts load the
  same way, also next to the other five modules in one process.
- Missing values are allowed in any input; the pipelines impute them (KNN for
  numbers, most frequent answer for categories).
- Risk bands are anchored to the cost-optimal threshold: Low (below it), Medium
  (threshold to 50 %), High (50 % or more).
