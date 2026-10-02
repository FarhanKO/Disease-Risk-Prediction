# HEART_NHANES.csv — Data Dictionary

**24,454 rows × 34 columns.** One row per examined adult aged 40+, from six
public CDC NHANES survey cycles. Target: `heart_disease` (12.5 % positive).

| Cycle | Adults | Heart disease prevalence |
|---|---|---|
| 2007–2008 | 3,843 | 13.1 % |
| 2009–2010 | 4,002 | 12.2 % |
| 2011–2012 | 3,423 | 11.3 % |
| 2013–2014 | 3,694 | 11.7 % |
| 2015–2016 | 3,585 | 13.0 % |
| 2017–March 2020 (pre-pandemic) | 5,907 | 13.1 % |

It replaces the UCI heart disease file (`Heart Diseases.csv`), which had only 920
patients from four hospitals, with `ca` 66 % missing, `thal` 53 % missing and
`slope` 34 % missing.

## Source and rebuild

National Health and Nutrition Examination Survey (NHANES), National Center for
Health Statistics, CDC — <https://wwwn.cdc.gov/nchs/nhanes/>. NHANES data are a
U.S. government work in the public domain.

```bash
# from heart/tabular/ — downloads the raw .xpt files into data/raw/ (cached, not in git)
python data/build_heart_nhanes.py
```

Inclusion: examined participants (`RIDSTATR = 2`), age ≥ 40 (the chest-pain
questionnaire starts at 40), answered the heart-disease questions. Components
used per cycle: DEMO, BMX, BPX (BPXO in 2017–2020), BPQ, MCQ, CDQ, SMQ, DIQ,
PAQ, HUQ, TCHOL, HDL, GHB, BIOPRO, ALB_CR and CBC.

## Columns

| Column | Type | Meaning | NHANES variable(s) |
|---|---|---|---|
| `participant_id` | int | Respondent sequence number, unique across cycles | `SEQN` |
| `survey_cycle` | text | NHANES cycle — metadata, not a model feature | — |
| `age` | years | Age at screening (80 = 80 and over) | `RIDAGEYR` |
| `gender` | text | Male / Female | `RIAGENDR` |
| `ethnicity` | text | Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, Other/Multi-Racial | `RIDRETH1` |
| `education_level` | 1–5 | 1 < 9th grade · 2 9–11th grade · 3 high school/GED · 4 some college · 5 college graduate | `DMDEDUC2` |
| `poverty_income_ratio` | 0–5 | Family income ÷ poverty threshold (5 = 5 or more) | `INDFMPIR` |
| `bmi` | kg/m² | Body-mass index | `BMXBMI` |
| `waist_cm` | cm | Waist circumference | `BMXWAIST` |
| `height_cm` | cm | Standing height | `BMXHT` |
| `systolic_bp` | mm Hg | Mean of the repeated systolic readings | `BPXSY1–4` (2007–16), `BPXOSY1–3` (2017–20) |
| `diastolic_bp` | mm Hg | Mean of the repeated diastolic readings (a reading of 0 is excluded) | `BPXDI1–4`, `BPXODI1–3` |
| `resting_heart_rate` | beats/min | 60-second pulse (2007–16) / mean oscillometric pulse (2017–20) | `BPXPLS`, `BPXOPLS1–3` |
| `total_cholesterol` | mg/dL | Serum total cholesterol | `LBXTC` |
| `hdl_cholesterol` | mg/dL | HDL cholesterol | `LBDHDD` |
| `hba1c` | % | Glycohemoglobin | `LBXGH` |
| `serum_creatinine` | mg/dL | Serum creatinine | `LBXSCR` |
| `uric_acid` | mg/dL | Serum uric acid | `LBXSUA` |
| `albumin_creatinine_ratio` | mg/g | Urine albumin ÷ urine creatinine | `URDACT`; 2007–08: `URXUMA` / `URXUCR` × 100 |
| `white_blood_cells` | 10³/µL | White blood cell count | `LBXWBCSI` |
| `hemoglobin` | g/dL | Hemoglobin | `LBXHGB` |
| `rdw` | % | Red cell distribution width | `LBXRDW` |
| `smoking_status` | text | Never (< 100 cigarettes in life) / Former / Current | `SMQ020`, `SMQ040` |
| `physically_active` | Yes/No | Any moderate or vigorous recreational activity (sports, fitness) in a typical week | `PAQ650`, `PAQ665` |
| `hypertension_history` | Yes/No | Ever told they had high blood pressure | `BPQ020` |
| `high_cholesterol_history` | Yes/No | Ever told their blood cholesterol was high | `BPQ080` (+ `BPQ060`) |
| `diabetes_status` | text | Doctor told they have diabetes: Yes / No / Borderline | `DIQ010` |
| `stroke_history` | Yes/No | Ever told they had a stroke | `MCQ160F` |
| `family_history_chd` | Yes/No | A parent or sibling had a heart attack or angina before age 50 | `MCQ300A` |
| `general_health` | 1–5 | Self-rated health: 1 excellent … 5 poor | `HUQ010` |
| `chest_pain_type` | text | Rose questionnaire: No pain / Non-exertional (pain, but not when hurrying or walking uphill) / Exertional | `CDQ001`, `CDQ002`, `CDQ003` |
| `severe_chest_pain` | Yes/No | Ever had severe pain across the front of the chest lasting 30 minutes or more | `CDQ008` |
| `sob_on_exertion` | Yes/No | Short of breath when hurrying on level ground or walking up a slight hill | `CDQ010` |
| `heart_disease` | 0/1 | **Target.** Ever told by a doctor they had coronary heart disease, angina, a heart attack or congestive heart failure | `MCQ160C`, `MCQ160D`, `MCQ160E`, `MCQ160B` |

## How raw codes were resolved

- Refused / don't-know codes (7, 9, 77, 99, ...) → missing.
- Zeros stored by the SAS transport format as 5.4e-79 → 0.
- **Survey skip patterns are not missing data.** The follow-up chest-pain
  questions (`CDQ002`, `CDQ008`) are only asked after "yes" to `CDQ001`, so a
  patient who never had chest pain gets `chest_pain_type = "No pain"` and
  `severe_chest_pain = "No"` instead of blanks that an imputer would fill with
  invented symptoms. Until 2016, `BPQ080` was only asked if the cholesterol had
  ever been checked (`BPQ060`); never checked means never told it was high, so
  it is "No". Every blank left in the file is a genuine unknown.
- `CDQ002 = 3` ("never walks uphill or hurries") counts as exertional pain if
  the pain comes on walking at an ordinary pace (`CDQ003 = 1`), and as
  non-exertional otherwise.
- The category is "No pain", not "None": pandas reads the string "None" as a
  missing value.
- Blood pressure is the mean of the available readings. A diastolic of 0
  (Korotkoff sounds heard down to zero) is not a measurement and is excluded.
- The target is 1 if any heart-disease question was answered "yes", and 0 only
  if all four were answered "no". Otherwise the row is dropped.

## Caveats

- **Self-reported diagnosis.** Some label-0 participants have undiagnosed
  coronary disease. The model learns *diagnosed* heart disease.
- **Cross-sectional, not prospective.** Vitals and labs are measured after the
  diagnosis, so treatment shows up in them: statins lower the cholesterol of
  many heart disease patients. This is a diagnostic (prevalence) model, not a
  10-year risk score such as the Pooled Cohort Equations or QRISK3.
- **Blood-pressure method changed in 2017**, from auscultatory (mercury) to
  oscillometric readings. The two agree to within a few mm Hg on average.
- Survey weights are not applied. The rows are not a nationally representative
  estimate, which is fine for model training but not for prevalence reporting.
- Stroke is a feature, not part of the target: it is cerebrovascular disease
  and belongs with a future brain module.
- Prescription medications (statins, aspirin, blood-pressure drugs) and "age
  when first told" are left out because they follow from the diagnosis (target
  leakage).
- LDL, triglycerides and fasting glucose exist only for the morning fasting
  subsample (about half), and hs-CRP was not measured in 2011–2014, so they are
  left out rather than creating large or cycle-shaped gaps. Non-HDL cholesterol
  (engineered in the notebook) stands in for LDL.
- The 2021–2023 cycle is left out: it dropped the chest-pain questionnaire,
  the family-history question and the recreational-activity questions.
