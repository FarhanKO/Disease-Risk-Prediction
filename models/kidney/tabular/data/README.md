# CKD_NHANES.csv — Data Notes

Raw export of **NHANES August 2021 – August 2023**: 11,933 participants × 29
columns (demographics, body measures, blood pressure, standard biochemistry
profile, urine albumin/creatinine, diabetes and smoking questionnaires).
NHANES data are a U.S. government work in the public domain —
<https://wwwn.cdc.gov/nchs/nhanes/>.

## The stored target is not used

`ckd_present` in this file is **not** a CKD label, so it is never used directly:

| Problem | Effect |
|---|---|
| 5,607 participants with no kidney labs (`ckd_stage = "Unknown"`, mostly children) are labelled `1` | 67 % of the "positives" have unknown status; every participant aged 0–11 is "CKD" |
| Everyone with eGFR 60–89 (`Stage 2`) is labelled `1`, whatever their urine albumin | KDIGO does not define a mildly reduced eGFR without kidney damage as CKD |

On rows with a known status, the stored label agrees with the clinical
definition only 73.5 % of the time.

## How the notebook and `src/` rebuild it

`src/data.py` (`load_raw_data`) and the notebook do the same thing:

1. **Target (KDIGO):** CKD = eGFR < 60 mL/min/1.73 m² **or** urine
   albumin-to-creatinine ratio ≥ 30 mg/g.
2. **Population:** adults 20+ with a determinable status (both tests present,
   or one test alone already meeting a CKD criterion). This leaves **5,473
   adults, 19.3 % CKD** (334 by eGFR only, 578 by albuminuria only, 143 by both).
3. **Leakage removal:** `egfr`, `albumin_creatinine_ratio`, `serum_creatinine`,
   `urine_albumin` and `urine_creatinine` define the label, so they are dropped,
   along with `ckd_stage` and `participant_id`.
4. **NHANES codes:** 7/9 (refused / don't know) → missing. SAS zeros stored as
   `5.4e-79` → 0. Skip-patterns resolved: the insulin and diabetes-pill questions
   are only asked of people with (pre)diabetes, so "not asked" → `"No"`, and the
   "smoking now" question → `smoking_status` (Never / Former / Current).
5. `weight_kg` is dropped (r = 0.89 with BMI).

## Model inputs after cleaning (19 columns)

| Column | Type | Notes |
|---|---|---|
| `age` | years | 20–80 (80 = 80 and over) |
| `gender`, `ethnicity` | text | `ethnicity` uses the six NHANES groups including Non-Hispanic Asian |
| `education_level` | 1–5 | < 9th grade … college graduate |
| `poverty_income_ratio` | 0–5 | family income ÷ poverty threshold |
| `bmi`, `height_cm` | numeric | |
| `bp_systolic`, `bp_diastolic` | mmHg | |
| `blood_urea_nitrogen`, `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid` | numeric | routine biochemistry |
| `diabetes` | Yes / No / Borderline | |
| `insulin_use`, `diabetes_pills` | Yes / No | |
| `smoking_status` | Never / Former / Current | |
