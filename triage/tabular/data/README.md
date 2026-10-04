# TRIAGE_NHAMCS.csv — Data Dictionary

**184,629 rows × 24 columns.** One row per adult (18+) emergency-department visit
where the patient gave at least one symptom as a reason for coming and the physician
recorded at least one codable diagnosis. It comes from thirteen public CDC NHAMCS
surveys (2010–2022). Targets: `heart` (4.3 %), `lung` (9.2 %) and `kidney` (3.9 %). A
visit can be in several groups (1.1 % are). 83.8 % of visits are in none of them: those are "other".

| Year | Visits | Heart | Lung | Kidney | Diagnosis coding |
|---|---|---|---|---|---|
| 2010 | 21,525 | 4.2 % | 8.3 % | 3.4 % | ICD-9-CM |
| 2011 | 19,737 | 4.3 % | 8.6 % | 3.4 % | ICD-9-CM |
| 2012 | 18,675 | 3.8 % | 8.1 % | 3.5 % | ICD-9-CM |
| 2013 | 15,690 | 4.0 % | 9.2 % | 3.5 % | ICD-9-CM |
| 2014 | 15,106 | 4.7 % | 8.9 % | 3.8 % | ICD-9-CM |
| 2015 | 13,420 | 4.3 % | 9.3 % | 3.6 % | ICD-9-CM |
| 2016 | 12,615 | 4.2 % | 8.8 % | 4.3 % | ICD-10-CM |
| 2017 | 10,732 | 4.1 % | 9.0 % | 3.9 % | ICD-10-CM |
| 2018 | 13,336 | 4.9 % | 10.1 % | 4.2 % | ICD-10-CM |
| 2019 | 12,460 | 4.6 % | 9.2 % | 4.1 % | ICD-10-CM |
| 2020 | 9,938 | 4.7 % | 10.1 % | 4.5 % | ICD-10-CM |
| 2021 | 11,021 | 4.6 % | 10.7 % | 4.7 % | ICD-10-CM |
| 2022 | 10,374 | 4.0 % | 11.2 % | 4.4 % | ICD-10-CM |

Lung rises from 2020 because COVID-19 (U07.1) is a lung diagnosis.

## Source and rebuild

National Hospital Ambulatory Medical Care Survey (NHAMCS), emergency-department
public-use files, National Center for Health Statistics, CDC:
<https://www.cdc.gov/nchs/nhamcs/>. The files are free to download without
registration. NCHS data are a U.S. government work in the public domain.

```bash
# from triage/tabular/ — downloads the raw archives into data/raw/ (cached, not in git; ~45 MB)
python data/build_triage_nhamcs.py
```

The script reads the Stata files for 2011–2022. 2010's Stata file ships only as a
self-extracting `.exe`, so the SPSS file is read instead (needs `pyreadstat`). It writes
three files:

| File | Content |
|---|---|
| `TRIAGE_NHAMCS.csv` | one row per visit (below) |
| `symptom_vocabulary.csv` | the 128 symptom names: body system, number of visits, and the NCHS reason-for-visit codes and labels behind each name |
| `diagnosis_groups.csv` | every diagnosis code that put a visit in an organ group, with its number of visits (the label audit) |

Inclusion rules:
- Age 18 or older.
- At least one of the first three reasons for the visit is a symptom (RFV symptom module, codes 1001–1999).
- At least one codable diagnosis. Blank, "left before being seen", "no diagnosis" and the other survey codes (V99x in the ICD-9 years, ZZZx in the ICD-10 years) do not count.

## Columns

| Column | Type | Meaning | NHAMCS variable(s) |
|---|---|---|---|
| `visit_id` | int | Row number, unique across years (NHAMCS has no patient identifier) | — |
| `survey_year` | int | Survey year: metadata, not a model feature | file |
| `age` | years | Age at the visit (top-coded at 93–100 depending on the year) | `AGE` |
| `sex` | text | Female / Male | `SEX` |
| `symptom_1` … `symptom_3` | text | The patient's symptoms, in the order given: the first three reasons for the visit that are symptoms, as names from `symptom_vocabulary.csv`. Blank when fewer were given | `RFV1`–`RFV3` |
| `injury_reason` | Yes/No | One of the first three reasons is an injury, poisoning or adverse effect (RFV injury module) | `RFV1`–`RFV3` |
| `diabetes` | Yes/No | Diabetes on the chronic-condition checklist (any type) | `DIABETES` (2010–13), `DIABTYP0/1/2` (2014–22), `NOCHRON` |
| `temperature_f` | °F | Initial temperature | `TEMPF` ÷ 10 |
| `heart_rate` | /min | Initial heart rate | `PULSE` |
| `respiratory_rate` | /min | Initial breathing rate | `RESPR` |
| `systolic_bp` | mm Hg | Initial systolic blood pressure | `BPSYS` |
| `diastolic_bp` | mm Hg | Initial diastolic blood pressure | `BPDIAS` |
| `oxygen_saturation` | % | Initial pulse oximetry | `POPCT` |
| `pain_score` | 0–10 | Pain scale at triage | `PAINSCALE` |
| `icd_version` | 9 / 10 | Diagnosis coding of the year (ICD-10-CM from 2016) | — |
| `diagnosis_1` … `diagnosis_3` | text | The physician's first three diagnoses, with the decimal point restored (`I50.9`, `427.31`). Outcome data: never a model input | `DIAG1`–`DIAG3` |
| `heart` | 0/1 | **Target.** A diagnosis in the heart group | derived |
| `lung` | 0/1 | **Target.** A diagnosis in the lung group | derived |
| `kidney` | 0/1 | **Target.** A diagnosis in the kidney group | derived |
| `primary_organ` | text | Group of the first organ diagnosis listed (heart / lung / kidney), or `other` | derived |

## The organ groups

A visit is in a group when any of its first three diagnoses falls in one of these
ranges. The ICD-9 and ICD-10 ranges are equivalent. NHAMCS keeps four ICD-10
characters (`I50.9`), which is enough for every range. `diagnosis_groups.csv` lists
all codes that matched and how often.

| Group | ICD-10-CM (2016–2022) | ICD-9-CM (2010–2015) | What it covers |
|---|---|---|---|
| **heart** | I05–I09, I11, I13, I20–I25, I30–I52; Z95.0–Z95.5, Z94.1 | 393–398, 402, 404, 410–414, 420–429; V45.0, V45.81, V45.82, V43.3, V42.1, V42.2 | rheumatic and hypertensive heart disease, angina and infarction, coronary disease, pericarditis, cardiomyopathy, arrhythmias, heart failure; pacemaker, bypass, valve and stent status, heart transplant |
| **lung** | J09–J22, J40–J99, C33–C34, A15–A16, U07, Z94.2 | 466, 480–488, 490–496, 500–508, 510–519, 162, 010–012, V42.6 | influenza and pneumonia, acute bronchitis and other lower respiratory infections, COPD, asthma, bronchiectasis, lung disease from external agents, interstitial and pleural disease, respiratory failure, lung cancer, tuberculosis, COVID-19, lung transplant |
| **kidney** | N00–N29, I12–I13, C64–C65, Q60–Q63, Z49, E08.2–E13.2, Z99.2, Z94.0 | 580–593, 403–404, 189.0–189.1, 753.0–753.3, 250.4, 788.0, V56, V45.1, V42.0 | glomerular disease, kidney infections, acute and chronic kidney failure, stones and renal colic, hydronephrosis, cysts and other kidney disorders, kidney cancer, congenital kidney disease, diabetic and hypertensive kidney disease, dialysis, kidney transplant |

Deliberately **not** in a group:
- Colds, sinusitis, sore throats and other upper-airway infections (J00–J06, J30–J39 / 460–465, 470–478). They are not lung disease, and none of the lung models looks at them.
- Pulmonary embolism and pulmonary hypertension (I26–I28 / 415–417).
- Bladder infections and other lower urinary tract problems (N30–N39 / 595–599).
- Stroke (I60–I69 / 430–438). It belongs with a future brain module.
- Symptom codes such as "chest pain, unspecified" (R07.9 / 786.50). When the emergency physician found no organ cause, the visit counts as *not* in the group. The router therefore learns how often a complaint turns out to be that organ's disease, not how often it is investigated for it.

Renal colic is a kidney diagnosis in both versions: N23 in ICD-10, and 788.0 (which ICD-9
files under symptoms) in ICD-9. Visits coded "probable", "questionable" or "rule out"
count like confirmed ones; they are under 1 % of diagnoses.

## How raw codes were resolved

- Blank and unknown codes (−9, −8, −7) → missing. A Doppler pulse or a palpated diastolic pressure (998) is not a number → missing. Temperature is stored in tenths of a degree.
- Physiologically impossible readings are entry errors and become missing, using wide limits: temperature 85–110 °F, heart rate 20–250, breathing rate 4–80, systolic 50–300, diastolic 20–200, oxygen saturation 50–100 %. This affects 1,117 visits (0.6 %): 535 oxygen saturations, 386 breathing rates, 236 heart rates and 22 blood pressures.
- **Symptom names.** NHAMCS records up to three reasons for the visit (five from 2014; the first three are used for every year alike) as NCHS reason-for-visit codes. Only symptom-module codes are used as symptoms, in the order given. Diagnoses the patient named ("my asthma") and treatments ("medication refill") are not inputs.
  - Codes that a patient could not tell apart share a name. "Shortness of breath", "Labored or difficult breathing" and "Breathing problems" are all *Shortness of breath*; "Side pain, flank pain" and "Kidney pain" are *Flank (side) or kidney pain*.
  - A name keeps its own column only if it appears in at least 200 visits. Rarer codes, and the survey's own "other symptoms referable to …" codes, become *Other \<body system\> symptom* (e.g. *Other digestive symptom*).
  - The result is 128 names, listed with their codes in `symptom_vocabulary.csv`.
- `injury_reason` is "Yes" when one of the first three reasons comes from the RFV injury module (injuries, poisoning, adverse effects of medicines). It is what the patient said, not the physician's injury assessment.
- **Diabetes.** The chronic-condition checklist uses one box until 2013 and three (type 1, type 2, unspecified) from 2014. When the whole checklist was left blank (`NOCHRON = 2`), diabetes is unknown, not "No".

## Caveats

- **Emergency-department patients, not the general public.** Everyone here was worried enough to come to an emergency department. Prevalences and probabilities are those of ED visits, and the model has never seen people with mild symptoms who stay at home.
- **A visit, not a person.** A patient can appear more than once. NHAMCS has no patient identifier, and repeat visits are a small share of a national sample.
- **Diagnoses at the ED, not follow-up.** The labels are the emergency physician's diagnoses at that visit. Some are provisional, and conditions found later are not included. The first three diagnoses include chronic conditions noted at the visit (CKD in a patient seen for a fall), so a head also learns "this patient has that organ's disease", not only "this organ explains today's symptoms".
- **Vital signs are measured in the ED.** A patient at home usually has none. The notebook trains with half the visits' vital signs removed and reports every metric with and without them.
- **ICD-9 to ICD-10 in 2016.** The code ranges are equivalent, but coding habits changed. Kidney prevalence steps from 3.6 % (2015) to 4.3 % (2016), mostly through more renal-colic and kidney-infection codes (N23, N10, N12 against 788.0 and 590.80), then settles around 4 %.
- **Survey weights are not applied.** The rows are not a nationally representative estimate, which is fine for model training but not for prevalence reporting.
- **Only diabetes from the medical history.** The chronic-condition checklist gained hypertension, CKD, COPD, heart failure and others only in 2014, so using them would leave year-shaped gaps. The organ conditions would also leak the labels: an ED physician lists a known CKD as a diagnosis.
