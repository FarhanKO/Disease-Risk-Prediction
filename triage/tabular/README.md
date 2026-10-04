# Symptom Triage — Tabular Module (Layer 1 Router)

Hears a patient's symptoms and decides which organ modules should look at them next:
**heart**, **lung**, **kidney**, or none of them (**other**). A routed patient then goes
to that organ's tabular risk model and, with a scan or printout, its image model:

```
symptoms, age, sex (+ vital signs if known)
        │
        ▼
  Symptom triage ──► Heart  ──► models/heart/tabular (heart disease risk) ──► models/heart/image_based (ECG)
  (this module)  ──► Lung   ──► models/lung/tabular (COPD risk)           ──► models/lung/image_based (chest X-ray)
                 ──► Kidney ──► models/kidney/tabular (CKD risk)          ──► models/kidney/image_based (CT)
                 ──► Other: no heart, lung or kidney signal
```

A patient can be routed to more than one organ (pneumonia with heart failure), so the
router has **three binary heads**, one per organ, each with its own calibrated
probability and cost-optimal threshold. Part of the `disease-risk-prediction` monorepo.

## Dataset

`data/TRIAGE_NHAMCS.csv`: **184,629 adult emergency-department visits** from thirteen
public CDC NHAMCS surveys (2010–2022), where the patient gave at least one symptom as a
reason for coming.

- **Inputs:** what a patient can tell. Age, sex, up to three symptoms (128 plain-language names grouped from NCHS reason-for-visit codes), diabetes, whether it is an injury, and seven vital signs if known.
- **Labels:** the emergency physician's first three diagnoses (ICD-9 / ICD-10), grouped into **heart (4.3 %)**, **lung (9.2 %)** and **kidney (3.9 %)**. 83.8 % of visits are in none: those are **other**.

The data dictionary, the exact ICD code ranges behind each group and the caveats are in
[`data/README.md`](data/README.md). The CSV is rebuilt from the CDC servers with:

```bash
python data/build_triage_nhamcs.py      # from triage/tabular/; caches the raw archives in data/raw/ (~45 MB)
```

## What the notebook does

`notebooks/Symptom_Triage.ipynb` is the full pipeline, following the same workflow as the
heart, kidney and lung notebooks:

- **Feature engineering:** the three symptoms become 128 order-free 0/1 columns; plus the number of symptoms, pulse pressure, the shock index and four triage flags (fever, low oxygen, fast heart rate, fast breathing).
- **Vital-sign dropout:** patients at home rarely have vital signs, so a random half of the training visits have all of theirs removed. Each head is then accurate and calibrated with or without them, and every test metric is reported both ways.
- **9 classifiers per head:** Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM, MLP, CatBoost and the Explainable Boosting Machine. Each is tuned on 5-fold cross-validated PR-AUC and selected by the one-standard-error rule. Grid results are cached, so a stopped run resumes.
- **Isotonic calibration** and a **cost-based threshold** per head (a missed organ costs 5× an unnecessary routing), chosen on out-of-fold training predictions.
- **Routing metrics** for the three heads together, bootstrapped 95 % confidence intervals, Brier scores, SHAP and permutation importance.
- PCA / t-SNE, DBSCAN, GMM, K-Means, six anomaly detectors, and a two-stage cascade: an Isolation Forest gate on the vital signs, then the three calibrated heads with SHAP explanations.

## Results

Test set: 36,926 visits never used for training or tuning (1,596 heart, 3,391 lung and 1,423 kidney).

**Model selection** (5-fold CV PR-AUC; a random head scores its prevalence: 0.043 heart, 0.092 lung, 0.039 kidney):

| Head | Best CV PR-AUC | Selected by the one-SE rule | Its CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|
| Heart | CatBoost 0.350 ± 0.004 | **CatBoost** | 0.350 | 0.349 | 0.896 |
| Lung | LightGBM 0.564 ± 0.005 | **Explainable Boosting** | 0.560 | 0.588 | 0.890 |
| Kidney | XGBoost 0.277 ± 0.007 | **LightGBM** | 0.275 | 0.296 | 0.856 |

All 27 models (9 per head) are in [`results/model_comparison.csv`](results/model_comparison.csv).
Gradient-boosted trees lead every head within about 0.01 of each other.

**Deployed heads** (calibrated, at the cost-optimal thresholds), on the test set *with / without* vital signs:

| Head | Threshold | Recall | Precision | Specificity | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Heart | 0.151 | 55.3 % / 54.0 % | 29.9 % / 29.0 % | 94.2 % / 94.0 % | 0.352 / 0.314 | 0.896 / 0.889 | 0.033 / 0.034 |
| Lung | 0.162 | 73.1 % / 71.5 % | 47.0 % / 46.1 % | 91.7 % / 91.5 % | 0.588 / 0.549 | 0.890 / 0.883 | 0.053 / 0.055 |
| Kidney | 0.154 | 32.0 % / 29.3 % | 33.6 % / 36.0 % | 97.5 % / 97.9 % | 0.296 / 0.278 | 0.856 / 0.843 | 0.032 / 0.032 |

95 % bootstrap intervals for recall (with vital signs):
- heart 0.529–0.577
- lung 0.717–0.746
- kidney 0.297–0.345

All heads are calibrated with and without vital signs: the mean predicted probability matches the prevalence within 0.07 points.

**The router as a whole** (with / without vital signs):

| | |
|---|---|
| Visits sent to at least one organ module | 22.0 % / 21.5 % |
| Organ modules run per visit | 0.26 / 0.25 |
| Visits with an organ diagnosis routed to at least one of their organs | 61.7 % / 59.9 % |
| … routed to all of their organs | 58.5 % / 56.9 % |
| Visits with no organ diagnosis left alone ("other") | 86.8 % / 87.1 % |

What drives it:
- **Symptoms.** Shuffling the first symptom costs 0.17–0.32 PR-AUC, more than any other input.
- **Age, for the heart.** Oxygen saturation and breathing rate for the lungs.
- **Two kinds of kidney patient.** Flank pain (stones, infections) and age plus diabetes (kidney failure, CKD).

Kidney stones and renal colic reach the kidney module 58 % of the time. Acute kidney failure (6 %) and chronic kidney disease (10 %) mostly come without kidney complaints: a symptom router cannot find them, which is the CKD risk model's job.

**Why vital-sign dropout.** A head trained only on complete ED records, then used without vital signs, underestimates risk by 10–24 %. Trained with half the vital signs removed, the same head is within 0.004 PR-AUC of a symptoms-only model when there are none, and within 0.008 of a vital-signs model when there are.

**The thresholds are a policy.** The 5:1 cost of a missed organ matches the other modules. Running one extra organ module is cheap, so a product may prefer 20:1 (threshold ≈ 0.05). That gives about 81 % heart, 85 % lung and 73 % kidney recall, at the price of routing about a fifth of the other visits to each module. The notebook's operating table shows the whole trade-off, and each threshold is one number in `models/metadata.json`.

## Structure

```
tabular/
├── data/
│   ├── build_triage_nhamcs.py   # downloads NHAMCS and builds the three files below
│   ├── TRIAGE_NHAMCS.csv        # one row per visit
│   ├── symptom_vocabulary.csv   # the 128 symptom names and the NCHS codes behind them
│   ├── diagnosis_groups.csv     # every diagnosis code in each organ group (label audit)
│   ├── README.md                # data dictionary and ICD code ranges
│   └── raw/                     # cached NHAMCS archives (gitignored)
├── notebooks/
│   ├── Symptom_Triage.ipynb     # training, evaluation, explainability, cascade
│   └── cv_cache/                # grid-search results, so a stopped run resumes (gitignored)
├── models/
│   ├── triage_heart_calibrated_model.joblib    # Stage 2: one calibrated head per organ
│   ├── triage_lung_calibrated_model.joblib
│   ├── triage_kidney_calibrated_model.joblib
│   ├── triage_anomaly_gate.joblib              # Stage 1: Isolation Forest gate
│   ├── metadata.json                           # thresholds, models, features, vocabulary, test metrics
│   └── reference_profile.json                  # typical patient, for the explanations
├── results/                     # model_comparison.csv, permutational_feature_importance.csv
├── images/                      # plots saved by the notebook
├── src/
│   ├── data.py                  # schema, symptom vocabulary, features, preprocessing, vital-sign dropout, gate
│   ├── models.py                # the nine candidates, their grids and the notebook's picks
│   ├── routing.py               # cost-optimal thresholds, the routing rule, router metrics
│   ├── train.py                 # tune -> one-SE -> calibrate -> OOF threshold -> save artifacts
│   ├── evaluate.py              # test metrics with / without vital signs, bootstrap CIs, routing, permutation importance
│   └── predict.py               # predict() / predict_batch(): the shared interface (common/)
├── examples/                    # patient.json, patients.json for `python -m src.predict`
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python data/build_triage_nhamcs.py        # rebuild the dataset from CDC NHAMCS

# Full analysis: 9 models x 3 heads, plots, observations (about two hours on a CPU)
jupyter nbconvert --to notebook --execute --inplace notebooks/Symptom_Triage.ipynb

# Or just the deployed heads, from triage/tabular/
python -m src.train                       # add --compare to re-run the nine-model comparison per head
python -m src.evaluate                    # adds test metrics to models/metadata.json
python -m src.predict --input examples/patient.json
python -m src.predict --input examples/patients.json   # several patients
python -m src.predict --list-symptoms     # the 128 names the router accepts
```

Set the `N_JOBS` environment variable (notebook) or `--n-jobs` (scripts) to limit the
CPU workers. By default they are chosen from the free RAM (about 0.6 GB per worker).

```python
from src.predict import predict, predict_batch      # or: from common.registry import route

result = predict({"age": 64, "sex": "Male",
                  "symptoms": ["Chest pain", "Shortness of breath", "Excessive sweating"],
                  "diabetes": "Yes"})                 # vital signs are optional
result.label               # 'Heart'                (or e.g. 'Heart + Lung', or 'Other')
result.details["route"]    # ['heart']              organ modules to run next, in order
result.details["probabilities"]   # {'heart': 0.474, 'lung': 0.126, 'kidney': 0.069}
result.message             # 'Route to Heart (47%): run its organ module next'
result.explanation["factors"]["heart"][:2]
# [{'feature': 'symptom_1', 'value': 'Chest pain', 'typical': None, 'effect': 0.150},
#  {'feature': 'age', 'value': 64.0, 'typical': 45.0, 'effect': 0.121}]

predict_batch(patients)    # many patients: status, label, route, p_heart, p_lung, p_kidney, positive, message
```

Only age, sex and one to three symptoms are required. Everything else may be left out
and is then scored as typical. Symptom names must come from
`data/symptom_vocabulary.csv`; a misspelled name raises an error listing the closest names.

The explanation swaps one input at a time for a typical patient's (median measurements,
most common answers, no symptom) and reports how much that head's probability changes.
Positive effects raised it.

## Notes

- **Different patients from the organ modules.** The router is trained on emergency-department visits and the organ models on NHANES participants and image datasets. How accurate the whole chain is (router, then organ model, then image model) cannot be measured on one set of patients; each layer is evaluated on its own.
- **The router is not a diagnosis.** It decides which organ modules to run, and it errs toward sending a patient to too many: a missed organ costs five times an unnecessary one. Chest pain with no organ cause found by the emergency physician counts as "not heart", so the probabilities are those of *diagnosed* organ disease among ED patients with those complaints.
- `src/` and the notebook build identical pipelines (same features, vital-sign dropout, grids, seeds and threshold rule). `python -m src.train` reproduces the notebook's deployed heads exactly: the same CV scores, the same thresholds (0.1509 / 0.1617 / 0.1544) and identical probabilities on all 36,926 test visits. The notebook-saved pipelines reference `add_custom_features` from the notebook; `common/artifacts.py` points that name at `src.data.add_custom_features` while loading.
- The notebook calibrates in its own process on purpose. Pipelines fitted in parallel workers come back holding a copy of the notebook's `add_custom_features`, which `joblib.dump` cannot save by name.
- The vital signs use wide plausibility limits at build time; the Stage 1 gate withholds extreme combinations. A patient who gives no vital signs always passes it.
