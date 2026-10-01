# COPD_NHANES.csv — Data Dictionary

**24,522 rows × 25 columns.** One row per examined adult aged 40+, from six
public CDC NHANES survey cycles. Target: `copd_present` (9.9 % positive).

| Cycle | Adults | COPD prevalence |
|---|---|---|
| 2007–2008 | 3,851 | 10.5 % |
| 2009–2010 | 4,015 | 7.7 % |
| 2011–2012 | 3,428 | 7.8 % |
| 2013–2014 | 3,700 | 10.0 % |
| 2015–2016 | 3,593 | 10.7 % |
| 2017–March 2020 (pre-pandemic) | 5,935 | 11.7 % |

## Source and rebuild

National Health and Nutrition Examination Survey (NHANES), National Center for
Health Statistics, CDC — <https://wwwn.cdc.gov/nchs/nhanes/>. NHANES data are a
U.S. government work in the public domain.

```bash
# from lung/tabular/ — downloads the raw .xpt files into data/raw/ (cached, not in git)
python data/build_copd_nhanes.py
```

Inclusion: examined participants (`RIDSTATR = 2`), age ≥ 40, answered the
COPD questions. Components used per cycle: DEMO, BMX, SMQ, SMQFAM, MCQ, CDQ,
CBC, HUQ, and cotinine (COTNAL until 2012, COT afterwards).

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
| `smoking_status` | text | Never (< 100 cigarettes in life) / Former / Current | `SMQ020`, `SMQ040` |
| `cigarettes_per_day` | count | Current: cigarettes per smoking day × days smoked ÷ 30. Former: per day when quitting. Never: 0 | `SMD650`, `SMD641`, `SMD057` |
| `smoking_years` | years | Years of regular smoking. Current: age − age started. Former: age quit − age started. Never / never regularly: 0 | `SMD030`, `SMD055`, `SMQ050Q/U` |
| `household_smokers` | 0–2 | Smokers living in the household (2 = 2 or more) | `SMD410`/`SMD415` (2007–12), `SMD460` (2013+) |
| `serum_cotinine` | ng/mL | Nicotine metabolite; results below the detection limit are reported as LOD/√2 ≈ 0.011 | `LBXCOT` |
| `asthma_history` | Yes/No | Ever told they had asthma | `MCQ010` |
| `heart_failure` | Yes/No | Ever told they had congestive heart failure | `MCQ160B` |
| `coronary_heart_disease` | Yes/No | Ever told they had coronary heart disease | `MCQ160C` |
| `sob_on_exertion` | Yes/No | Short of breath when hurrying on level ground or walking up a slight hill | `CDQ010` |
| `general_health` | 1–5 | Self-rated health: 1 excellent … 5 poor | `HUQ010` |
| `eosinophils` | 10³/µL | Eosinophil count | `LBDEONO` |
| `neutrophils` | 10³/µL | Segmented neutrophil count | `LBDNENO` |
| `lymphocytes` | 10³/µL | Lymphocyte count | `LBDLYMNO` |
| `hemoglobin` | g/dL | Hemoglobin | `LBXHGB` |
| `copd_present` | 0/1 | **Target.** Ever told by a doctor they had COPD, emphysema, or chronic bronchitis | `MCQ160G/K/O` (2007–16), `MCQ160P` (2017–20) |

## How raw codes were resolved

- Refused / don't-know codes (7, 9, 77, 99, 777, 999, 77777, 99999) → missing.
- Zeros stored by the SAS transport format as 5.4e-79 → 0.
- **Survey skip patterns are not missing data.** A never-smoker is never asked
  how much they smoke, so `cigarettes_per_day` and `smoking_years` are set to 0
  for them rather than left blank; otherwise an imputer would invent a smoking
  history. Every blank left in the file is a genuine unknown.
- 2017–2020 dropped "age last smoked regularly", so a former smoker's quit age is
  age − time since quitting (`SMQ050Q`/`SMQ050U`, "50+ years" = 50). The
  resulting smoking-years distribution matches the earlier cycles.
- Household smokers are top-coded at "2 or more" in 2017–2020, so all cycles are
  capped at 2.
- The target is 1 if any COPD question was answered "yes", and 0 only if every
  question asked in that cycle was answered "no". Otherwise the row is dropped.

## Caveats

- **Self-reported diagnosis.** COPD is widely under-diagnosed, so some label-0
  participants have undiagnosed airflow obstruction. The model learns
  *diagnosed* COPD.
- **Question wording changed in 2017.** Separate emphysema / chronic bronchitis /
  COPD questions became one combined question, which partly explains the higher
  2017–2020 prevalence.
- Survey weights are not applied. The rows are not a nationally representative
  estimate, which is fine for model training but not for prevalence reporting.
- Spirometry and respiratory-symptom questionnaires exist only for 2007–2012, so
  they are left out rather than creating cycle-shaped gaps. Prescription
  medications and "age at diagnosis" are left out because they follow from the
  diagnosis (target leakage).
