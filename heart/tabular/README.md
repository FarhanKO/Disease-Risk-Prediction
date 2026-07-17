# Heart Disease — Tabular Module

Predicts presence of heart disease from clinical/tabular patient data. Part of the
`disease-risk-prediction` monorepo (heart / kidney / lung, each with tabular + image +
fusion submodules, orchestrated by a top-level cascade).

Ported from `Heart_Diseases.ipynb`, refactored into an importable, CLI-capable
package. The notebook is retained for EDA only — `src/` is the canonical pipeline.

## What it does

- Engineers 3 clinical features on top of the raw UCI heart disease columns:
  `max_hr_ratio`, `bp_hr_index`, `age_st_interaction`.
- Compares 8 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest,
  Naive Bayes, MLP, XGBoost, CatBoost) via grid search, selecting on **recall first,
  precision second** — missing a real case of heart disease is treated as far
  costlier than a false alarm.
- Balances classes with SMOTE, calibrates the winning model with isotonic regression
  (`CalibratedClassifierCV`) so output probabilities are trustworthy, not just ranks.
- Picks a decision threshold via a clinical cost function (false negatives weighted
  5x false positives), not a flat 0.5 cutoff.
- Validates with bootstrapped confidence intervals, Brier score, SHAP, and
  permutation importance.

## Structure

```
tabular/
├── requirements.txt
├── notebook.ipynb      # EDA only — not the canonical pipeline
├── src/
│   ├── data.py          # loading, cleaning, feature engineering, preprocessing pipeline
│   ├── train.py         # 8-model grid search, calibration, saves the model artifact
│   ├── evaluate.py       # metrics, threshold optimization, SHAP, permutation importance
│   └── predict.py       # stable inference interface — what Streamlit/cascade import
└── models/              # saved .joblib artifacts + metadata.json (gitignored/LFS if large)
```

## Usage

```bash
pip install -r requirements.txt

# Train and calibrate
python -m src.train --data-path data/heart_disease.csv \
    --model-out models/heart_tabular_calibrated.joblib

# Evaluate — also writes models/metadata.json with the optimal threshold
python -m src.evaluate --data-path data/heart_disease.csv \
    --model-path models/heart_tabular_calibrated.joblib

# Predict on new patients
python -m src.predict --input patient.json \
    --model-path models/heart_tabular_calibrated.joblib
```

```python
# Or import directly (what Streamlit does)
from src.predict import predict_single

result = predict_single({
    "age": 63, "sex": "Male", "dataset": "Cleveland", "cp": "typical angina",
    "trestbps": 145, "chol": 233, "fbs": True, "restecg": "normal",
    "thalch": 150, "exang": False, "oldpeak": 2.3, "slope": "downsloping",
    "ca": 0, "thal": "fixed defect",
})
# {'heart_disease_probability': 0.71, 'prediction': 1, 'risk_level': 'High Risk'}
```

## Notes

- `predict.py` only needs raw clinical columns — feature engineering and
  preprocessing happen inside the saved pipeline itself.
- Risk bands: <0.20 Low, <0.50 Medium, ≥0.50 High (fixed, independent of the
  cost-optimized decision threshold used for the binary `prediction` field).
- No anomaly-detection gate here — that's a Stage 1 concern for the top-level
  `cascade/` module, not this module's stable interface.
