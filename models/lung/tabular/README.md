# Lung Disease (COPD) — Tabular Module

Predicts doctor-diagnosed **COPD** (emphysema or chronic bronchitis) from
demographics, tobacco exposure (questionnaire and serum cotinine), breathlessness,
comorbidities and a blood count. A positive screen means "refer for spirometry".
Part of the `disease-risk-prediction` monorepo (heart / kidney / lung, each with
tabular and image submodules).

## Dataset

`data/COPD_NHANES.csv`: **24,522 examined adults aged 40+** from six CDC NHANES
survey cycles (2007 to March 2020), **9.9 % COPD prevalence**. The data dictionary,
the NHANES variables behind every column and the caveats (self-reported diagnosis,
the 2017 change in question wording) are in [`data/README.md`](data/README.md).

```bash
python data/build_copd_nhanes.py      # from lung/tabular/; caches raw .xpt files in data/raw/
```

## Status

- `notebook/Lung_Disease.ipynb` is the full analysis: 17 models, calibration,
  threshold, SHAP, clustering and anomaly detection, cell for cell the same
  workflow as the heart and kidney notebooks. **It has been run end to end**
  (2026-10-01), with every observation cell written from its outputs. Tuned
  models are cached in `notebook/grid_cache/` (not in git), so a re-run takes about
  25 minutes. Set `RERUN_GRID=1` to tune everything again (about 2.5 hours with
  `N_JOBS=3`).
- The notebook and `python -m src.train --compare` select the same model
  (L1 logistic regression, C = 0.1, SMOTENC off). The artifacts in `models/` are the
  notebook's, with test metrics and gate rates added by `python -m src.evaluate`.

## What `src/` does

- Uses the notebook's inputs: 21 raw columns (`waist_cm` is dropped, r = 0.89 with
  BMI) plus 4 engineered features: pack-years, log cotinine, neutrophil-to-lymphocyte
  ratio and a cardiopulmonary comorbidity count.
- Same leakage-aware pipeline as the heart and kidney modules: scale → KNN-impute →
  SMOTENC (switched on or off per model by cross-validation) → one-hot, all inside
  the cross-validation folds.
- Tunes each candidate on 5-fold CV PR-AUC and picks the simplest model within one
  standard error of the best.
- Isotonic calibration, and a cost-based threshold (a missed case costs 5× a false
  alarm) chosen on out-of-fold training predictions, never on the test set.
- Two-stage cascade: an Isolation Forest gate (1 %, fitted on the training set)
  withholds implausible profiles for manual review, then the calibrated model.

## Results

From `notebook/Lung_Disease.ipynb` (full table in `results/model_comparison.csv`).
Test set: 4,905 patients never used for training or tuning (486 with COPD).

| Model | CV PR-AUC (± SE) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| TabPFN | 0.466 ± 0.013 | 0.494 | 0.868 |
| Stacking Ensemble (LR + EBM + CatBoost + XGBoost) | 0.463 ± 0.012 | 0.495 | 0.867 |
| **Logistic Regression (selected)** | **0.461 ± 0.012** | **0.490** | **0.865** |
| Explainable Boosting | 0.460 ± 0.011 | 0.492 | 0.867 |
| TabM | 0.457 ± 0.015 | 0.489 | 0.863 |
| CatBoost | 0.457 ± 0.013 | 0.489 | 0.866 |
| XGBoost | 0.456 ± 0.011 | 0.492 | 0.866 |
| Neural Network (MLP) | 0.453 ± 0.014 | 0.486 | 0.864 |
| RealMLP · LightGBM · FT-Transformer · Random Forest | 0.450–0.452 | 0.476–0.493 | 0.860–0.864 |
| KNN · TabNet · Decision Tree · SVM · Naive Bayes | 0.384–0.417 | 0.384–0.447 | 0.722–0.849 |
| Random model | 0.099 | 0.099 | 0.5 |

Eight models are within one standard error of the best (cutoff 0.4526), so the
one-SE rule picks the simplest: **logistic regression**. TabPFN's +0.005 is a third
of its own standard error and would add a 260 MB GPU model. Cross-validation switched
SMOTENC off for every model except Naive Bayes.

Deployed model (calibrated logistic regression at the cost-optimal threshold
**0.164**; the Bayes-optimal value for calibrated probabilities is 0.167), on the
test set with 95 % bootstrap intervals:

| Metric | Value |
|---|---|
| Recall (sensitivity) | 0.689 (0.651–0.734) |
| Precision | 0.346 (0.319–0.376) |
| Specificity | 0.857 |
| ROC-AUC | 0.864 (0.847–0.880) |
| PR-AUC | 0.480 (0.434–0.527) |
| Brier score | 0.067 (0.089 for always predicting the prevalence) |

At this threshold 151 of the 486 COPD patients are missed and 632 of the 4,419
others are referred. Top permutation importances: asthma history, years smoked,
breathlessness on exertion, self-rated health and ethnicity.

## Structure

```
tabular/
├── data/
│   ├── build_copd_nhanes.py    # downloads NHANES and builds the CSV
│   ├── COPD_NHANES.csv
│   ├── README.md               # data dictionary
│   └── raw/                    # cached NHANES .xpt files (gitignored)
├── notebook/
│   └── Lung_Disease.ipynb      # full analysis: 17 models, calibration, SHAP, clustering, anomalies, cascade
├── models/
│   ├── lung_copd_calibrated_model.joblib   # Stage 2: calibrated classifier
│   ├── lung_copd_anomaly_gate.joblib       # Stage 1: Isolation Forest gate
│   ├── metadata.json                       # threshold, features, CV scores, test metrics
│   └── reference_profile.json              # typical training patient, for the explanations
├── results/                    # model_comparison.csv, permutational_feature_importance.csv
├── images/                     # plots saved by the notebook
├── src/
│   ├── data.py                 # schema, feature engineering, preprocessing pipeline, anomaly gate
│   ├── train.py                # tune -> one-SE selection -> calibrate -> OOF threshold -> save artifacts
│   ├── evaluate.py             # test metrics, bootstrap CIs, Brier, gate rates, permutation importance
│   └── predict.py              # the shared predict() / predict_batch() interface (common/tabular.py)
├── examples/patient.json       # sample input for `python -m src.predict`
└── requirements.txt
```

## Usage

From this folder (`lung/tabular/`):

```bash
pip install -r requirements.txt

python -m src.train --compare --n-jobs 6   # ~30 min on CPU; without --compare: Logistic Regression only
python -m src.evaluate                     # adds test metrics to models/metadata.json
python -m src.predict --input examples/patient.json
```

```python
# What Streamlit calls (the same interface as the other five modules; see the repo README)
from src.predict import predict, predict_batch

result = predict({
    "age": 68, "gender": "Male", "ethnicity": "Non-Hispanic White", "education_level": 2,
    "poverty_income_ratio": 1.1, "bmi": 21.5, "height_cm": 175, "smoking_status": "Current",
    "cigarettes_per_day": 25, "smoking_years": 48, "household_smokers": 1, "serum_cotinine": 350,
    "asthma_history": "Yes", "heart_failure": "No", "coronary_heart_disease": "Yes",
    "sob_on_exertion": "Yes", "general_health": 4, "eosinophils": 0.4, "neutrophils": 6.5,
    "lymphocytes": 1.4, "hemoglobin": 16.2,
})
result.status, result.label, result.probability, result.positive
# ('accepted', 'High Risk', 0.887, True)
result.message       # 'High Risk (89%): refer for spirometry'
result.explanation["factors"][:3]
# [{'feature': 'asthma_history', 'value': 'Yes', 'typical': 'No', 'effect': 0.399},
#  {'feature': 'smoking_years', 'value': 48.0, 'typical': 0.0, 'effect': 0.347},
#  {'feature': 'sob_on_exertion', 'value': 'Yes', 'typical': 'No', 'effect': 0.283}]

predict_batch(patients_df)   # many patients: status, label, probability, positive, message per row
```

## Notes

- Missing values are allowed in any input; the pipelines impute them (KNN for
  numbers, most frequent answer for categories). For never-smokers,
  `cigarettes_per_day` and `smoking_years` are 0, not missing (see `data/README.md`).
- Risk bands are anchored to the cost-optimal threshold: Low (below it), Medium
  (threshold to 50 %), High (50 % or more).
- **Stage 1 withholds some real patients.** On the test set the gate withholds
  1.2 % of patients: 6.2 % of COPD patients vs 0.6 % of the others, the same pattern
  as the kidney module (disease produces extreme values). Withheld patients get
  `status = "review"` with no probability, not a low score.
- The explanation swaps one input at a time for the typical training patient's value
  (`models/reference_profile.json`) and reports the change in probability.
- The label is *diagnosed* COPD. COPD is widely under-diagnosed, so some label-0
  participants have undiagnosed airflow obstruction.
