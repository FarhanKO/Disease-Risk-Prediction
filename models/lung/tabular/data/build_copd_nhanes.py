"""
build_copd_nhanes.py — Builds COPD_NHANES.csv from public CDC NHANES files.

Six survey cycles (2007-2008 ... 2015-2016, plus 2017-March 2020 pre-pandemic),
examined adults aged 40+, one row per participant. Target `copd_present` is a
doctor diagnosis of COPD, emphysema, or chronic bronchitis (self-reported).

Every raw NHANES code is resolved here, so the CSV holds only real values or
genuine unknowns (NaN):
    - refused / don't-know codes (7, 9, 77, 99, 777, 999, 7777, 9999, ...) -> NaN
    - SAS-transport zeros stored as 5.4e-79 -> 0
    - survey skip patterns: a never-smoker has 0 cigarettes/day and 0 smoking
      years (a true zero, not a missing value)
    - household smokers top-coded at "2 or more" in 2017-2020, so every cycle
      is capped at 2

CLI (run from models/lung/tabular/):
    python data/build_copd_nhanes.py
"""

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "COPD_NHANES.csv"

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
MIN_AGE = 40

# (label, start year, file-name template). 2017-2020 files use a P_ prefix.
CYCLES = [
    ("2007-2008", "2007", "{}_E"),
    ("2009-2010", "2009", "{}_F"),
    ("2011-2012", "2011", "{}_G"),
    ("2013-2014", "2013", "{}_H"),
    ("2015-2016", "2015", "{}_I"),
    ("2017-2020", "2017", "P_{}"),
]
# Serum cotinine lives in COTNAL (cotinine + NNAL) until 2012, COT afterwards
COTININE_FILES = {"2007": "COTNAL", "2009": "COTNAL", "2011": "COTNAL"}
COMPONENTS = ["DEMO", "BMX", "SMQ", "SMQFAM", "MCQ", "CDQ", "CBC", "HUQ", "COT"]

ETHNICITY = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
             4: "Non-Hispanic Black", 5: "Other/Multi-Racial"}
YES_NO = {1: "Yes", 2: "No"}
UNIT_TO_YEARS = {1: 1 / 365.25, 2: 1 / 52.18, 3: 1 / 12, 4: 1.0}  # SMQ050U: days/weeks/months/years


# ---------- download / read ----------

def download(url: str, dest: Path) -> None:
    """curl first (uses the OS certificate store on Windows), urllib otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)
    if dest.read_bytes()[:6] != b"HEADER":
        dest.unlink()
        raise ValueError(f"{url} did not return a SAS transport file")


def read_component(year: str, template: str, component: str) -> pd.DataFrame:
    stem = COTININE_FILES.get(year, "COT") if component == "COT" else component
    name = template.format(stem)
    path = RAW_DIR / f"{name}.xpt"
    if not path.exists():
        print(f"  downloading {name}.xpt")
        download(BASE_URL.format(year=year, name=name), path)
    df = pd.read_sas(path, format="xport")
    # SAS transport stores 0 as 5.4e-79
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-9, 0.0)
    return df.set_index("SEQN")


def clean(series: pd.Series, missing_codes) -> pd.Series:
    return series.where(~series.isin(missing_codes))


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column or all-NaN if this cycle did not ask the question."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


# ---------- per-domain derivations ----------

def smoking_features(smq: pd.DataFrame, age: pd.Series) -> pd.DataFrame:
    ever = clean(col(smq, "SMQ020"), [7, 9])          # >=100 cigarettes in life
    now = clean(col(smq, "SMQ040"), [7, 9])           # 1 every day, 2 some days, 3 not at all
    status = pd.Series(np.select(
        [ever == 2, (ever == 1) & now.isin([1, 2]), (ever == 1) & (now == 3)],
        ["Never", "Current", "Former"], default=""), index=smq.index).replace("", np.nan)

    # Cigarettes/day. Current: cigarettes on smoking days x share of days smoked.
    per_smoking_day = clean(col(smq, "SMD650"), [777, 999])
    days_smoked = clean(col(smq, "SMD641"), [77, 99])
    current_cpd = np.where(days_smoked.notna(), per_smoking_day * days_smoked / 30,
                           np.where(now == 1, per_smoking_day, np.nan))
    former_cpd = clean(col(smq, "SMD057"), [777, 999])  # per day when quit
    cpd = np.select([status == "Never", status == "Current", status == "Former"],
                    [0.0, current_cpd, former_cpd], default=np.nan)

    # Years smoked. SMD030 = age started regularly (0 = never smoked regularly).
    age_start = clean(col(smq, "SMD030"), [777, 999])
    quit_age = clean(col(smq, "SMD055"), [777, 999])   # not asked in 2017-2020
    since_quit = clean(col(smq, "SMQ050Q"), [77777, 99999])
    since_quit = since_quit.where(since_quit != 66666, 50)  # "50 years or more"
    years_since = since_quit * col(smq, "SMQ050U").map(UNIT_TO_YEARS)
    quit_age = quit_age.fillna(age - years_since)

    regular = age_start > 0
    current_years = np.where(regular, age - age_start, np.where(age_start == 0, 0.0, np.nan))
    former_years = np.where(regular, quit_age - age_start, np.where(age_start == 0, 0.0, np.nan))
    years = np.select([status == "Never", status == "Current", status == "Former"],
                      [0.0, current_years, former_years], default=np.nan)
    years = pd.Series(years, index=smq.index).clip(lower=0).round(1)

    return pd.DataFrame({
        "smoking_status": status,
        "cigarettes_per_day": pd.Series(cpd, index=smq.index).round(1),
        "smoking_years": years,
    })


def household_smokers(fam: pd.DataFrame) -> pd.Series:
    if "SMD460" in fam.columns:                        # 2013 onwards
        count = clean(fam["SMD460"], [777, 999])
    else:                                              # 2007-2012: yes/no + count
        anyone = clean(fam["SMD410"], [7, 9])
        count = pd.Series(np.where(anyone == 2, 0.0, np.where(anyone == 1, clean(fam["SMD415"], [7, 9]), np.nan)),
                          index=fam.index)
    return count.clip(upper=2)                         # harmonise to "2 or more"


def copd_label(mcq: pd.DataFrame) -> pd.Series:
    """1 if any COPD / emphysema / chronic bronchitis diagnosis, 0 if every asked question is 'No'."""
    questions = [c for c in ("MCQ160G", "MCQ160K", "MCQ160O", "MCQ160P") if c in mcq.columns]
    answers = mcq[questions].where(~mcq[questions].isin([7, 9]))
    label = pd.Series(np.nan, index=mcq.index)
    label[(answers == 2).all(axis=1)] = 0
    label[(answers == 1).any(axis=1)] = 1
    return label


# ---------- assembly ----------

def build_cycle(label: str, year: str, template: str) -> pd.DataFrame:
    print(f"{label}:")
    t = {c: read_component(year, template, c) for c in COMPONENTS}
    demo = t["DEMO"]
    examined = demo[(demo["RIDSTATR"] == 2) & (demo["RIDAGEYR"] >= MIN_AGE)]
    idx = examined.index
    age = examined["RIDAGEYR"]

    df = pd.DataFrame(index=idx)
    df["survey_cycle"] = label
    df["age"] = age
    df["gender"] = examined["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["ethnicity"] = examined["RIDRETH1"].map(ETHNICITY)
    df["education_level"] = clean(examined["DMDEDUC2"], [7, 9])
    df["poverty_income_ratio"] = examined["INDFMPIR"]

    bmx = t["BMX"].reindex(idx)
    df["bmi"] = bmx["BMXBMI"]
    df["waist_cm"] = bmx["BMXWAIST"]
    df["height_cm"] = bmx["BMXHT"]

    df = df.join(smoking_features(t["SMQ"].reindex(idx), age))
    df["household_smokers"] = household_smokers(t["SMQFAM"].reindex(idx))
    df["serum_cotinine"] = t["COT"].reindex(idx)["LBXCOT"]

    mcq = t["MCQ"].reindex(idx)
    df["asthma_history"] = mcq["MCQ010"].map(YES_NO)
    df["heart_failure"] = mcq["MCQ160B"].map(YES_NO)
    df["coronary_heart_disease"] = mcq["MCQ160C"].map(YES_NO)
    df["sob_on_exertion"] = t["CDQ"].reindex(idx)["CDQ010"].map(YES_NO)
    df["general_health"] = clean(t["HUQ"].reindex(idx)["HUQ010"], [7, 9])

    cbc = t["CBC"].reindex(idx)
    df["eosinophils"] = cbc["LBDEONO"]
    df["neutrophils"] = cbc["LBDNENO"]
    df["lymphocytes"] = cbc["LBDLYMNO"]
    df["hemoglobin"] = cbc["LBXHGB"]

    df["copd_present"] = copd_label(mcq)
    df = df[df["copd_present"].notna()]
    print(f"  {len(df):,} adults, COPD prevalence {df['copd_present'].mean():.1%}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build COPD_NHANES.csv from CDC NHANES.")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    data = pd.concat([build_cycle(*c) for c in CYCLES])
    data.index = data.index.astype(int)
    data.index.name = "participant_id"
    data["copd_present"] = data["copd_present"].astype(int)
    data = data.reset_index()

    data.to_csv(args.out, index=False)
    print(f"\n[SAVED] {len(data):,} rows x {data.shape[1]} columns -> {args.out}")
    print(f"COPD prevalence: {data['copd_present'].mean():.2%}")
    print("\nMissing values (%):")
    print((data.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
