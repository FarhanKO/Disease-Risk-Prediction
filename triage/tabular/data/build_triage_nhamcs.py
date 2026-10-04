"""
build_triage_nhamcs.py — Builds TRIAGE_NHAMCS.csv from the public CDC NHAMCS
emergency-department files (National Hospital Ambulatory Medical Care Survey).

Thirteen survey years (2010-2022), adult visits (18+) where the patient gave at
least one symptom as a reason for coming, one row per visit. The inputs are what
a patient can tell: age, sex, up to three symptoms, diabetes, whether it is about
an injury, and the vital signs and pain score if known. The three targets say
which organ groups the emergency physician's diagnoses fall in:

    heart   ischemic heart disease, arrhythmias, heart failure, other heart disease
    lung    influenza and pneumonia, lower respiratory infections, COPD and asthma,
            other lung disease, lung cancer, tuberculosis, COVID-19
    kidney  kidney failure and CKD, kidney infections, kidney stones and renal colic,
            other kidney disorders, kidney cancer, dialysis and transplant status

A visit can be in several groups (pneumonia with heart failure); a visit in none
is "other". The groups follow the three organ modules of this repo: the heart,
lung and kidney tabular and image models are the next layer after this one.

Every raw NHAMCS code is resolved here, so the CSV holds only real values or
genuine unknowns (NaN):
    - blank / unknown codes (-9, -8, -7) -> NaN; a Doppler pulse or palpated
      diastolic pressure (998) is not a number -> NaN
    - temperature is stored in tenths of a degree (986 = 98.6 °F)
    - physiologically impossible readings (a pulse of 3, an oxygen saturation of
      5 %: entry errors) -> NaN, using wide limits (PLAUSIBLE_RANGE)
    - symptoms: the patient's first three reasons for visit (RFV1-3) that are
      symptoms (RFV symptom module, codes 1001-1999), in the order given. Codes
      seen in at least 200 adult visits keep their own name (SYMPTOM_NAMES);
      rarer ones are grouped by body system ("Other digestive symptom")
    - diabetes: the chronic-condition checklist; an entirely blank checklist is
      unknown, not "No"
    - diagnoses: the first three, ICD-9-CM until 2015 and ICD-10-CM from 2016,
      mapped to the organ groups with equivalent code ranges in both versions
      (ORGAN_CODES). NHAMCS keeps the first four ICD-10 characters, which is
      enough for every range used here

Visits without a symptom among the first three reasons, or without a codable
diagnosis (left before being seen, "no diagnosis", blank), are left out.

The public-use files are free to download (no registration); NCHS data are a
U.S. government work in the public domain.

CLI (run from triage/tabular/):
    python data/build_triage_nhamcs.py
"""

import argparse
import io
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "TRIAGE_NHAMCS.csv"
VOCABULARY_PATH = DATA_DIR / "symptom_vocabulary.csv"
DIAGNOSIS_GROUPS_PATH = DATA_DIR / "diagnosis_groups.csv"

BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/NHAMCS/{folder}/{name}"
MIN_AGE = 18

# Stata files from 2011 on; 2010 ships Stata only as a self-extracting .exe, so its SPSS file is used
FILES = {
    2010: ("spss", "ed2010-spss.zip"),
    2011: ("stata", "ed2011-stata.zip"),
    2012: ("stata", "ed2012-stata.zip"),
    2013: ("stata", "ed2013-stata.zip"),
    2014: ("stata", "ed2014-stata.zip"),
    2015: ("stata", "ED2015-stata.zip"),
    2016: ("stata", "ED2016-stata.zip"),
    2017: ("stata", "ed2017-stata.zip"),
    2018: ("stata", "ED2018-stata.zip"),
    2019: ("stata", "ED2019-stata.zip"),
    2020: ("stata", "ed2020-stata.zip"),
    2021: ("stata", "ed2021-stata.zip"),
    2022: ("stata", "ed2022-stata.zip"),
}
FIRST_ICD10_YEAR = 2016

REASONS = ["RFV1", "RFV2", "RFV3"]          # 2014+ record five; the first three keep every year alike
DIAGNOSES = ["DIAG1", "DIAG2", "DIAG3"]
VITALS = ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE"]
CHECKLIST = ["DIABETES", "DIABTYP0", "DIABTYP1", "DIABTYP2", "NOCHRON"]   # DIABETES until 2013, DIABTYP* from 2014
COLUMNS = ["AGE", "SEX"] + VITALS + REASONS + DIAGNOSES + CHECKLIST

# Inclusive plausibility limits; values outside them (a pulse of 3, an oxygen saturation of 5 %) become NaN
PLAUSIBLE_RANGE = {
    "temperature_f": (85, 110), "heart_rate": (20, 250), "respiratory_rate": (4, 80), "systolic_bp": (50, 300),
    "diastolic_bp": (20, 200), "oxygen_saturation": (50, 100), "pain_score": (0, 10),
}

SYMPTOM_MODULE = (10000, 20000)             # RFV codes 1000.0-1999.9, stored x10
INJURY_MODULE = (50000, 60000)              # injuries, poisoning and adverse effects
BLANK_CODES = [-9, -8, -7]

# The RFV symptom module is ordered by body system; rare symptoms fall back to their system
BODY_SYSTEMS = [
    (10000, 11000, "general"),
    (11000, 12000, "psychological"),
    (12000, 12600, "nervous system"),
    (12600, 13000, "circulation and lymph"),
    (13000, 14000, "eye and ear"),
    (14000, 15000, "respiratory"),
    (15000, 16400, "digestive"),
    (16400, 18300, "urinary and genital"),
    (18300, 19000, "skin, nail and hair"),
    (19000, 20000, "muscle and joint"),
]
MIN_SYMPTOM_VISITS = 200

# Client-facing names for the reason-for-visit symptom codes (RFV code x 10: 10501 = 1050.1).
# Codes that a patient could not tell apart share a name ("Shortness of breath" and "Labored or
# difficult breathing"; a category's general code and its main entry). A name keeps its own column
# only if its codes appear in at least MIN_SYMPTOM_VISITS adult visits (2010-2022); every other code,
# and the RFV's own "other symptoms referable to ..." codes, become "Other <body system> symptom".
SYMPTOM_NAMES = {
    # general
    10050: "Chills",
    10100: "Fever",
    10150: "Tiredness or exhaustion",
    10200: "General weakness",
    10250: "Feeling generally unwell",
    10300: "Fainting",
    10350: "Fluid retention or imbalance",
    10351: "Swelling (edema)",
    10352: "Excessive sweating",
    10400: "Weight gain",
    10460: "Face symptoms",
    10500: "Chest pain", 10501: "Chest pain",
    10502: "Chest pressure or tightness",
    10503: "Burning in the chest",
    10550: "Pain at another specific site",
    10551: "Rib pain",
    10552: "Flank (side) or kidney pain",
    10553: "Groin pain",
    10554: "Facial pain",
    10600: "Pain, site not specified", 10601: "Pain, site not specified",
    10602: "Cramps or spasms",
    10700: "Bleeding, site not specified",
    10900: "Allergic reaction",
    10950: "Unsteady walking or poor coordination",
    # psychological
    11000: "Anxiety or nervousness",
    11100: "Depression",
    11300: "Behavior changes", 11302: "Hostile or aggressive behavior",
    11351: "Trouble sleeping",
    11352: "Excessive sleepiness",
    11450: "Alcohol-related problem",
    11500: "Drug use problem",
    11550: "Delusions or hallucinations",
    11650: "Other psychological symptom",
    # nervous system
    12000: "Tremor or involuntary movements",
    12050: "Seizure (convulsions)",
    12100: "Headache",
    12150: "Memory problems",
    12201: "Numbness",
    12203: "Tingling (pins and needles)",
    12250: "Dizziness or vertigo",
    12300: "Weakness of a limb or one side",
    12350: "Speech difficulty", 12352: "Speech difficulty",
    12400: "Other nervous system symptom",
    # circulation and lymph
    12600: "Palpitations",
    12601: "Fast heartbeat",
    12603: "Irregular heartbeat",
    12800: "Other circulation and lymph symptom",
    # eye and ear
    13050: "Vision problems", 13052: "Vision problems",
    13100: "Eye discharge",
    13201: "Eye pain",
    13301: "Red or discolored eyes",
    13353: "Eye swelling",
    13350: "Other eye and ear symptom", 13650: "Other eye and ear symptom",
    13451: "Hearing loss",
    13551: "Earache",
    # respiratory
    14000: "Nasal congestion",
    14051: "Nosebleed",
    14100: "Sinus pain or congestion", 14101: "Sinus pain or congestion", 14103: "Sinus pain or congestion",
    14150: "Shortness of breath", 14200: "Shortness of breath", 14300: "Shortness of breath",
    14250: "Wheezing",
    14400: "Cough",
    14450: "Head cold",
    14501: "Flu symptoms",
    14550: "Sore throat", 14551: "Sore throat", 14552: "Sore throat",
    14555: "Throat swelling",
    14701: "Coughing up blood",
    14703: "Coughing up phlegm",
    14750: "Chest congestion",
    14850: "Other respiratory symptom",
    # digestive
    15000: "Toothache or gum problem", 15001: "Toothache or gum problem",
    15050: "Lip symptoms",
    15100: "Mouth pain or sores", 15101: "Mouth pain or sores",
    15200: "Difficulty swallowing",
    15250: "Nausea",
    15300: "Vomiting",
    15350: "Heartburn or indigestion",
    15450: "Abdominal pain", 15451: "Abdominal pain",
    15452: "Lower abdominal pain",
    15453: "Upper abdominal pain",
    15651: "Abdominal swelling or bloating", 15653: "Abdominal swelling or bloating",
    15702: "Loss of appetite",
    15800: "Blood in stool or rectal bleeding", 15801: "Blood in stool or rectal bleeding",
    16052: "Blood in stool or rectal bleeding",
    15802: "Vomiting blood",
    15900: "Constipation",
    15950: "Diarrhea",
    16003: "Change in stools",
    16051: "Rectal pain",
    16150: "Other digestive symptom",
    # urinary and genital
    16401: "Blood in urine",
    16403: "Unusual urine color or smell",
    16450: "Frequent or urgent urination",
    16500: "Painful urination",
    16551: "Urinary incontinence",
    16600: "Difficulty urinating", 16601: "Difficulty urinating",
    16701: "Flank (side) or kidney pain",
    16750: "Suspected urinary infection",
    16800: "Other urinary and genital symptom",
    17000: "Penile pain or discharge", 17001: "Penile pain or discharge", 17050: "Penile pain or discharge",
    17151: "Testicle or scrotum pain or swelling", 17152: "Testicle or scrotum pain or swelling",
    17550: "Vaginal bleeding", 17551: "Vaginal bleeding",
    17600: "Vaginal discharge",
    17650: "Vaginal pain or itching", 17651: "Vaginal pain or itching", 17653: "Vaginal pain or itching",
    17751: "Pelvic pain",
    17900: "Pregnancy-related problem", 17901: "Pregnancy-related problem",
    17902: "Pregnancy-related problem", 17903: "Pregnancy-related problem",
    18000: "Breast pain",
    # skin, nail and hair
    18350: "Skin redness or discoloration",
    18400: "Skin infection", 18402: "Skin infection", 18403: "Skin infection",
    18550: "Skin lesion or growth", 18650: "Skin lesion or growth",
    18600: "Skin rash",
    18700: "Itchy or irritated skin", 18702: "Itchy or irritated skin",
    18750: "Skin swelling",
    18800: "Other skin, nail and hair symptom",
    # muscle and joint
    19001: "Neck pain",
    19050: "Back pain", 19051: "Back pain", 19052: "Back pain",
    19101: "Low back pain",
    19151: "Hip pain",
    19200: "Leg pain", 19201: "Leg pain", 19202: "Leg pain",
    19204: "Leg weakness",
    19205: "Leg swelling",
    19251: "Knee pain or swelling", 19255: "Knee pain or swelling",
    19301: "Ankle pain or swelling", 19305: "Ankle pain or swelling",
    19350: "Foot or toe pain or swelling", 19351: "Foot or toe pain or swelling", 19355: "Foot or toe pain or swelling",
    19401: "Shoulder pain",
    19450: "Arm pain or swelling", 19451: "Arm pain or swelling", 19455: "Arm pain or swelling",
    19501: "Elbow pain or swelling", 19505: "Elbow pain or swelling",
    19551: "Wrist pain or swelling", 19555: "Wrist pain or swelling",
    19600: "Hand or finger pain or swelling", 19601: "Hand or finger pain or swelling",
    19605: "Hand or finger pain or swelling",
    19651: "Muscle aches or cramps", 19652: "Muscle aches or cramps",
    19701: "Joint pain",
    19800: "Other muscle and joint symptom",
}

# ---------- diagnoses -> organ groups ----------
# Ranges are inclusive and compare the code's first three characters (its category); entries
# with four characters match that exact subcategory. ICD-9 and ICD-10 ranges are equivalent.
ORGAN_CODES = {
    "heart": {
        10: [("I05", "I09"), ("I11", "I11"), ("I13", "I13"), ("I20", "I25"), ("I30", "I52"),
             "Z950", "Z951", "Z952", "Z953", "Z954", "Z955",    # pacemaker, bypass, valve, stent status
             "Z941"],                                           # heart transplant
        9: [("393", "398"), ("402", "402"), ("404", "404"), ("410", "414"), ("420", "429"),
            "V450", "V4581", "V4582", "V433", "V421", "V422"],   # the same statuses
    },
    "lung": {
        10: [("J09", "J22"), ("J40", "J99"), ("C33", "C34"), ("A15", "A16"), ("U07", "U07"),
             "Z942"],                                           # lung transplant
        9: [("466", "466"), ("480", "488"), ("490", "496"), ("500", "508"), ("510", "519"),
            ("162", "162"), ("010", "012"), "V426"],
    },
    "kidney": {
        10: [("N00", "N29"), ("I12", "I13"), ("C64", "C65"), ("Q60", "Q63"), ("Z49", "Z49"),
             "E082", "E092", "E102", "E112", "E132",            # diabetes with kidney complications
             "Z992", "Z940"],                                   # dialysis dependence, kidney transplant
        9: [("580", "593"), ("403", "404"), ("V56", "V56"),
            "2504", "1890", "1891", "7530", "7531", "7532", "7533",
            "7880",                                             # renal colic (N23 in ICD-10)
            "V451", "V420"],
    },
}
ORGANS = list(ORGAN_CODES)


# ---------- download / read ----------

def download(url: str, dest: Path) -> None:
    """curl first (uses the OS certificate store on Windows), urllib otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    if shutil.which("curl"):
        subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(partial), url], check=True)
    else:
        urllib.request.urlretrieve(url, partial)
    if partial.read_bytes()[:2] != b"PK":
        partial.unlink()
        raise ValueError(f"{url} did not return a zip archive")
    partial.replace(dest)


def raw_archive(year: int) -> Path:
    folder, name = FILES[year]
    path = RAW_DIR / name
    if not path.exists():
        print(f"  downloading {name}")
        download(BASE_URL.format(folder=folder, name=name), path)
    return path


def read_year(year: int) -> tuple[pd.DataFrame, dict]:
    """The columns this build uses (whichever the year has) and the reason-for-visit code labels."""
    archive = zipfile.ZipFile(raw_archive(year))
    folder, _ = FILES[year]
    suffix = ".dta" if folder == "stata" else ".sav"
    member = next(i for i in archive.infolist() if i.filename.lower().endswith(suffix))

    if folder == "stata":
        reader = pd.io.stata.StataReader(io.BytesIO(archive.read(member)))
        df = reader.read(convert_categoricals=False)
        df = df[[c for c in COLUMNS if c in df.columns]]
        # The reason-for-visit label set is the one that names code 1050.1, "Chest pain"
        rfv_labels = next((labels for labels in reader.value_labels().values() if 10501 in labels), {})
    else:
        import pyreadstat
        sav = RAW_DIR / Path(member.filename).name
        if not sav.exists():
            sav.write_bytes(archive.read(member))
        _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
        present = [c for c in COLUMNS if c in meta.column_names]
        df, meta = pyreadstat.read_sav(str(sav), usecols=present)
        rfv_labels = meta.variable_value_labels.get("RFV1", {})

    for column in REASONS + [c for c in VITALS + CHECKLIST + ["AGE", "SEX"] if c in df.columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in DIAGNOSES:
        df[column] = df[column].astype(str).str.strip()
    rfv_labels = {int(code): str(label).strip() for code, label in rfv_labels.items()}
    return df.reset_index(drop=True), rfv_labels


def clean(series: pd.Series, missing_codes=BLANK_CODES) -> pd.Series:
    return series.where(~series.isin(missing_codes))


# ---------- per-domain derivations ----------

def vital_signs(df: pd.DataFrame) -> pd.DataFrame:
    vitals = pd.DataFrame({
        "temperature_f": clean(df["TEMPF"]) / 10,
        "heart_rate": clean(df["PULSE"], BLANK_CODES + [998]),       # 998 = Doppler, no number
        "respiratory_rate": clean(df["RESPR"]),
        "systolic_bp": clean(df["BPSYS"]),
        "diastolic_bp": clean(df["BPDIAS"], BLANK_CODES + [998]),    # 998 = palpated / Doppler
        "oxygen_saturation": clean(df["POPCT"]),
        "pain_score": clean(df["PAINSCALE"]),
    })
    # Readings no patient who walks in and describes symptoms can have are entry errors, not measurements
    return pd.DataFrame({c: vitals[c].where(vitals[c].between(*PLAUSIBLE_RANGE[c])) for c in vitals}, index=vitals.index)


def diabetes(df: pd.DataFrame) -> pd.Series:
    """Chronic-condition checklist. NOCHRON = 2 means the whole item was left blank: unknown, not "No"."""
    boxes = [c for c in ("DIABETES", "DIABTYP0", "DIABTYP1", "DIABTYP2") if c in df.columns]
    answer = np.where((df[boxes] == 1).any(axis=1), "Yes", "No")
    return pd.Series(answer, index=df.index).where(df["NOCHRON"] != 2)


def symptom_codes(df: pd.DataFrame) -> pd.Series:
    """Per visit, the symptom-module codes among the first three reasons, in order, without repeats."""
    reasons = df[REASONS].where((df[REASONS] >= SYMPTOM_MODULE[0]) & (df[REASONS] < SYMPTOM_MODULE[1]))
    return reasons.apply(lambda row: list(dict.fromkeys(int(c) for c in row.dropna())), axis=1)


def injury_reason(df: pd.DataFrame) -> pd.Series:
    """The patient named an injury, poisoning or adverse effect among the first three reasons."""
    injured = ((df[REASONS] >= INJURY_MODULE[0]) & (df[REASONS] < INJURY_MODULE[1])).any(axis=1)
    return injured.map({True: "Yes", False: "No"})


def body_system(code: int) -> str:
    return next(name for low, high, name in BODY_SYSTEMS if low <= code < high)


def other_name(code: int) -> str:
    return f"Other {body_system(code)} symptom"


def symptom_names(visit_codes: pd.Series) -> dict:
    """
    RFV code -> symptom name for every code in the data: its SYMPTOM_NAMES entry if that name is
    mentioned in at least MIN_SYMPTOM_VISITS visits, otherwise its body system's "Other" name.
    """
    codes = sorted({c for row in visit_codes for c in row})
    curated = {c: SYMPTOM_NAMES.get(c, other_name(c)) for c in codes}
    visits_per_name = pd.Series([n for row in visit_codes for n in {curated[c] for c in row}]).value_counts()
    return {c: name if visits_per_name[name] >= MIN_SYMPTOM_VISITS or name.startswith("Other ") else other_name(c)
            for c, name in curated.items()}


# ---------- diagnoses ----------

def normalise_diagnosis(code: str, icd10: bool):
    """
    NHAMCS code -> the code without padding ('4280-' -> '4280', 'J01-' -> 'J01'), or None for
    blanks and the survey's own non-diagnosis codes (V99x in ICD-9 years, ZZZx in ICD-10 years:
    left before being seen, 'none', noncodable, blank).
    """
    code = str(code).strip().rstrip("-")
    if code in ("", "nan", "None") or code.startswith("-") or len(code) < 3:
        return None
    if code.startswith("ZZZ" if icd10 else "V99"):
        return None
    return code


def organs_of(code: str, icd10: bool) -> list:
    """Organ groups a normalised diagnosis code belongs to (I13 / 404 are both heart and kidney)."""
    version = 10 if icd10 else 9
    category = code[:3]
    groups = []
    for organ, versions in ORGAN_CODES.items():
        for rule in versions[version]:
            if isinstance(rule, tuple):
                low, high = rule
                hit = len(category) == 3 and low <= category <= high and category[0] == low[0]
            else:
                hit = code.startswith(rule)
            if hit:
                groups.append(organ)
                break
    return groups


def dotted(code, icd10: bool):
    """'I509' -> 'I50.9', '42731' -> '427.31', 'V4511' -> 'V45.11' (for reading, not for matching)."""
    if not isinstance(code, str):
        return None
    if not icd10 and code.startswith("E"):
        return code[:4] + ("." + code[4:] if len(code) > 4 else "")
    return code[:3] + ("." + code[3:] if len(code) > 3 else "")


def diagnosis_labels(df: pd.DataFrame, year: int) -> pd.DataFrame:
    icd10 = year >= FIRST_ICD10_YEAR
    # Plain lists, not Series.map: pandas string columns would turn None into NaN
    codes = [[normalise_diagnosis(c, icd10) for c in row] for row in df[DIAGNOSES].itertuples(index=False)]
    groups = [[organs_of(c, icd10) if c else [] for c in row] for row in codes]

    out = pd.DataFrame(index=df.index)
    out["icd_version"] = 10 if icd10 else 9
    for i in range(len(DIAGNOSES)):
        out[f"diagnosis_{i + 1}"] = [dotted(row[i], icd10) for row in codes]
    for organ in ORGANS:
        out[organ] = [int(any(organ in g for g in row)) for row in groups]
    # Primary organ: the group of the first diagnosis that falls in one ("other" if none does)
    out["primary_organ"] = [next((g[0] for g in row if g), "other") for row in groups]
    out["has_diagnosis"] = [any(c is not None for c in row) for row in codes]
    return out


# ---------- assembly ----------

def build_year(year: int) -> tuple[pd.DataFrame, dict]:
    raw, rfv_labels = read_year(year)
    raw = raw[raw["AGE"] >= MIN_AGE].reset_index(drop=True)

    df = pd.DataFrame(index=raw.index)
    df["survey_year"] = year
    df["age"] = raw["AGE"]
    df["sex"] = raw["SEX"].map({1: "Female", 2: "Male"})
    df = df.join(vital_signs(raw))
    df["diabetes"] = diabetes(raw)
    df["injury_reason"] = injury_reason(raw)
    df["symptom_codes"] = symptom_codes(raw)
    df = df.join(diagnosis_labels(raw, year))

    keep = df["symptom_codes"].str.len().gt(0) & df["has_diagnosis"]
    df = df[keep].drop(columns="has_diagnosis")
    print(f"{year}: {len(raw):,} adult visits -> {len(df):,} with a symptom and a diagnosis | "
          + ", ".join(f"{o} {df[o].mean():.1%}" for o in ORGANS))
    return df, rfv_labels


def symptom_vocabulary(data: pd.DataFrame, names: dict, rfv_labels: dict) -> pd.DataFrame:
    """One row per symptom name: its body system, the RFV codes behind it and how many visits mention it."""
    visits = data[["symptom_1", "symptom_2", "symptom_3"]].stack().value_counts()
    rows = pd.DataFrame({"code": list(names), "symptom": list(names.values())}).sort_values("code")
    rows["body_system"] = rows["code"].map(body_system)
    rows["rfv"] = rows["code"].map(lambda c: f"{c / 10:.1f}")
    rows["rfv_label"] = rows["code"].map(lambda c: rfv_labels.get(c, ""))
    # A name whose codes sit in two body systems (flank pain 1055.2, kidney pain 1670.1) is listed under the first
    vocabulary = rows.groupby("symptom", as_index=False).agg(
        body_system=("body_system", "first"),
        rfv_codes=("rfv", " ".join),
        rfv_labels=("rfv_label", lambda s: " | ".join(label for label in s if label)))
    vocabulary.insert(2, "visits", vocabulary["symptom"].map(visits).astype(int))
    # Body systems in RFV order; named symptoms by frequency, the "Other ..." group last
    system_rank = {name: i for i, (_, _, name) in enumerate(BODY_SYSTEMS)}
    vocabulary["_system"] = vocabulary["body_system"].map(system_rank)
    vocabulary["_other"] = vocabulary["symptom"].str.startswith("Other ")
    vocabulary = vocabulary.sort_values(["_system", "_other", "visits"], ascending=[True, True, False])
    return vocabulary.drop(columns=["_system", "_other"]).reset_index(drop=True)


def diagnosis_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Every diagnosis code that put a visit in an organ group, with the number of visits it appears in."""
    columns = [f"diagnosis_{i}" for i in range(1, len(DIAGNOSES) + 1)]
    long = data.melt(id_vars="icd_version", value_vars=columns, value_name="code", ignore_index=False)
    long = long.dropna(subset=["code"]).reset_index().drop_duplicates(["index", "code"])   # once per visit
    counts = long.groupby(["icd_version", "code"]).size()

    rows = [{"organ": organ, "icd_version": int(version), "code": code, "visits": int(visits)}
            for (version, code), visits in counts.items()
            for organ in organs_of(code.replace(".", ""), icd10=version == 10)]
    groups = pd.DataFrame(rows)
    return groups.sort_values(["organ", "visits"], ascending=[True, False]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build TRIAGE_NHAMCS.csv from CDC NHAMCS emergency-department files.")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--vocabulary-out", default=str(VOCABULARY_PATH))
    parser.add_argument("--diagnosis-groups-out", default=str(DIAGNOSIS_GROUPS_PATH))
    args = parser.parse_args()

    years, labels = [], {}
    for year in FILES:
        df, rfv_labels = build_year(year)
        years.append(df)
        labels = {**labels, **rfv_labels}          # later years' labels win (newest wording)
    data = pd.concat(years, ignore_index=True)

    # Symptom codes -> names; a visit keeps its distinct names in the order given (at most three)
    names = symptom_names(data["symptom_codes"])
    per_visit = data["symptom_codes"].map(lambda codes: list(dict.fromkeys(names[c] for c in codes)))
    for i in range(3):
        data[f"symptom_{i + 1}"] = per_visit.map(lambda n: n[i] if len(n) > i else np.nan)

    vocabulary = symptom_vocabulary(data, names, labels)
    data = data.drop(columns="symptom_codes")
    data.insert(0, "visit_id", np.arange(1, len(data) + 1))
    columns = ["visit_id", "survey_year", "age", "sex", "symptom_1", "symptom_2", "symptom_3",
               "injury_reason", "diabetes", "temperature_f", "heart_rate", "respiratory_rate",
               "systolic_bp", "diastolic_bp", "oxygen_saturation", "pain_score",
               "icd_version", "diagnosis_1", "diagnosis_2", "diagnosis_3"] + ORGANS + ["primary_organ"]
    data = data[columns]

    groups = diagnosis_groups(data)
    data.to_csv(args.out, index=False)
    vocabulary.to_csv(args.vocabulary_out, index=False)
    groups.to_csv(args.diagnosis_groups_out, index=False)
    print(f"\n[SAVED] {len(data):,} rows x {data.shape[1]} columns -> {args.out}")
    print(f"[SAVED] {len(vocabulary)} symptom names -> {args.vocabulary_out}")
    print(f"[SAVED] {len(groups):,} organ-group diagnosis codes -> {args.diagnosis_groups_out}")
    print("Organ groups:", ", ".join(f"{o} {data[o].mean():.2%}" for o in ORGANS),
          f"| none (other) {(data[ORGANS].sum(axis=1) == 0).mean():.2%}")
    print("\nMissing values (%):")
    print((data.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
