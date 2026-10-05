"""
build_heart_nhanes.py — Builds HEART_NHANES.csv from public CDC NHANES files.

Six survey cycles (2007-2008 ... 2015-2016, plus 2017-March 2020 pre-pandemic),
examined adults aged 40+, one row per participant. Target `heart_disease` is a
doctor diagnosis of coronary heart disease, angina, heart attack, or congestive
heart failure (self-reported).

Every raw NHANES code is resolved here, so the CSV holds only real values or
genuine unknowns (NaN):
    - refused / don't-know codes (7, 9, 77, 99, 777, 999, ...) -> NaN
    - SAS-transport zeros stored as 5.4e-79 -> 0
    - survey skip patterns: someone who never had chest pain is not asked
      whether it comes on with exertion or ever lasted 30+ minutes, so those
      answers are a true "No pain" / "No", not a missing value
    - blood pressure is the mean of the repeated readings; the auscultatory
      protocol (2007-2016) and the oscillometric one (2017-2020) are merged

The 2021-2023 cycle is left out: it dropped the chest-pain (CDQ), family-history
(MCQ300A) and recreational-activity questions, which would leave cycle-shaped
gaps in the data.

CLI (run from models/heart/tabular/):
    python data/build_heart_nhanes.py
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
OUT_PATH = DATA_DIR / "HEART_NHANES.csv"

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
MIN_AGE = 40  # the chest-pain questionnaire (CDQ) is only asked from age 40

# (label, start year, file-name template). 2017-2020 files use a P_ prefix.
CYCLES = [
    ("2007-2008", "2007", "{}_E"),
    ("2009-2010", "2009", "{}_F"),
    ("2011-2012", "2011", "{}_G"),
    ("2013-2014", "2013", "{}_H"),
    ("2015-2016", "2015", "{}_I"),
    ("2017-2020", "2017", "P_{}"),
]
COMPONENTS = ["DEMO", "BMX", "BP", "BPQ", "MCQ", "CDQ", "SMQ", "DIQ", "PAQ", "HUQ",
              "TCHOL", "HDL", "GHB", "BIOPRO", "ALB_CR", "CBC"]
# Blood pressure: auscultatory BPX until 2016, oscillometric BPXO in 2017-2020
BP_FILES = {"2017": "BPXO"}

ETHNICITY = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
             4: "Non-Hispanic Black", 5: "Other/Multi-Racial"}
YES_NO = {1: "Yes", 2: "No"}
DIABETES = {1: "Yes", 2: "No", 3: "Borderline"}


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
    stem = BP_FILES.get(year, "BPX") if component == "BP" else component
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


def yes_no(series: pd.Series) -> pd.Series:
    return series.map(YES_NO)                         # 7 / 9 fall through to NaN


# ---------- per-domain derivations ----------

def blood_pressure(bp: pd.DataFrame) -> pd.DataFrame:
    """Mean of the repeated readings. A diastolic of 0 (sounds to zero) is not a measurement."""
    sys_cols = [c for c in bp.columns if c.startswith(("BPXSY", "BPXOSY"))]
    dia_cols = [c for c in bp.columns if c.startswith(("BPXDI", "BPXODI"))]
    pulse_cols = [c for c in bp.columns if c in ("BPXPLS", "BPXOPLS1", "BPXOPLS2", "BPXOPLS3")]
    return pd.DataFrame({
        "systolic_bp": bp[sys_cols].mean(axis=1).round(1),
        "diastolic_bp": bp[dia_cols].where(bp[dia_cols] > 0).mean(axis=1).round(1),
        "resting_heart_rate": bp[pulse_cols].mean(axis=1).round(1),
    })


def smoking_status(smq: pd.DataFrame) -> pd.Series:
    ever = clean(col(smq, "SMQ020"), [7, 9])          # >=100 cigarettes in life
    now = clean(col(smq, "SMQ040"), [7, 9])           # 1 every day, 2 some days, 3 not at all
    return pd.Series(np.select(
        [ever == 2, (ever == 1) & now.isin([1, 2]), (ever == 1) & (now == 3)],
        ["Never", "Current", "Former"], default=""), index=smq.index).replace("", np.nan)


def chest_pain(cdq: pd.DataFrame) -> pd.DataFrame:
    """
    Rose-questionnaire chest pain. CDQ002 and CDQ008 are only asked after
    "yes" to CDQ001, so "no chest pain ever" means No pain / No, not unknown.
    """
    any_pain = clean(col(cdq, "CDQ001"), [7, 9])      # ever had chest pain or discomfort
    uphill = col(cdq, "CDQ002")                       # 1 yes, 2 no, 3 never walks uphill / hurries
    level = col(cdq, "CDQ003")                        # pain walking at an ordinary pace on the level
    exertional = (uphill == 1) | ((uphill == 3) & (level == 1))
    pain_type = np.select(
        [any_pain == 2, (any_pain == 1) & exertional, (any_pain == 1) & uphill.isin([2, 3])],
        ["No pain", "Exertional", "Non-exertional"], default="")   # not "None": pandas reads that as NaN

    severe = clean(col(cdq, "CDQ008"), [7, 9])        # severe chest pain lasting 30+ minutes
    severe = severe.where(any_pain != 2, 2)           # structural "No" when there was never any pain

    return pd.DataFrame({
        "chest_pain_type": pd.Series(pain_type, index=cdq.index).replace("", np.nan),
        "severe_chest_pain": yes_no(severe),
        "sob_on_exertion": yes_no(col(cdq, "CDQ010")),
    })


def high_cholesterol_history(bpq: pd.DataFrame) -> pd.Series:
    """
    BPQ080 (told cholesterol is high). Until 2016 it was only asked if the blood
    cholesterol had ever been checked (BPQ060); never checked -> never told -> "No".
    """
    told = clean(col(bpq, "BPQ080"), [7, 9])
    never_checked = col(bpq, "BPQ060") == 2
    return yes_no(told.where(told.notna() | ~never_checked, 2))


def physically_active(paq: pd.DataFrame) -> pd.Series:
    """Any moderate or vigorous recreational activity (sports, fitness) in a typical week."""
    vigorous = clean(col(paq, "PAQ650"), [7, 9])
    moderate = clean(col(paq, "PAQ665"), [7, 9])
    active = pd.Series(np.nan, index=paq.index)
    active[(vigorous == 2) & (moderate == 2)] = 2
    active[(vigorous == 1) | (moderate == 1)] = 1
    return yes_no(active)


def albumin_creatinine_ratio(alb: pd.DataFrame) -> pd.Series:
    """Urine ACR in mg/g. 2007-2008 ships the two inputs only: albumin (mg/L) / creatinine (mg/dL) x 100."""
    if "URDACT" in alb.columns:
        return alb["URDACT"]
    return (alb["URXUMA"] / alb["URXUCR"] * 100).round(2)


def heart_disease_label(mcq: pd.DataFrame) -> pd.Series:
    """1 if any heart-disease diagnosis, 0 if every question is 'No'."""
    questions = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"]  # heart failure, CHD, angina, heart attack
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

    df = pd.DataFrame(index=idx)
    df["survey_cycle"] = label
    df["age"] = examined["RIDAGEYR"]
    df["gender"] = examined["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["ethnicity"] = examined["RIDRETH1"].map(ETHNICITY)
    df["education_level"] = clean(examined["DMDEDUC2"], [7, 9])
    df["poverty_income_ratio"] = examined["INDFMPIR"]

    bmx = t["BMX"].reindex(idx)
    df["bmi"] = bmx["BMXBMI"]
    df["waist_cm"] = bmx["BMXWAIST"]
    df["height_cm"] = bmx["BMXHT"]
    df = df.join(blood_pressure(t["BP"].reindex(idx)))

    df["total_cholesterol"] = t["TCHOL"].reindex(idx)["LBXTC"]
    df["hdl_cholesterol"] = t["HDL"].reindex(idx)["LBDHDD"]
    df["hba1c"] = t["GHB"].reindex(idx)["LBXGH"]
    biopro = t["BIOPRO"].reindex(idx)
    df["serum_creatinine"] = biopro["LBXSCR"]
    df["uric_acid"] = biopro["LBXSUA"]
    df["albumin_creatinine_ratio"] = albumin_creatinine_ratio(t["ALB_CR"].reindex(idx))
    cbc = t["CBC"].reindex(idx)
    df["white_blood_cells"] = cbc["LBXWBCSI"]
    df["hemoglobin"] = cbc["LBXHGB"]
    df["rdw"] = cbc["LBXRDW"]

    df["smoking_status"] = smoking_status(t["SMQ"].reindex(idx))
    df["physically_active"] = physically_active(t["PAQ"].reindex(idx))
    bpq = t["BPQ"].reindex(idx)
    df["hypertension_history"] = yes_no(bpq["BPQ020"])
    df["high_cholesterol_history"] = high_cholesterol_history(bpq)
    df["diabetes_status"] = t["DIQ"].reindex(idx)["DIQ010"].map(DIABETES)
    mcq = t["MCQ"].reindex(idx)
    df["stroke_history"] = yes_no(mcq["MCQ160F"])
    df["family_history_chd"] = yes_no(mcq["MCQ300A"])
    df["general_health"] = clean(t["HUQ"].reindex(idx)["HUQ010"], [7, 9])
    df = df.join(chest_pain(t["CDQ"].reindex(idx)))

    df["heart_disease"] = heart_disease_label(mcq)
    df = df[df["heart_disease"].notna()]
    print(f"  {len(df):,} adults, heart disease prevalence {df['heart_disease'].mean():.1%}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build HEART_NHANES.csv from CDC NHANES.")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    data = pd.concat([build_cycle(*c) for c in CYCLES])
    data.index = data.index.astype(int)
    data.index.name = "participant_id"
    data["heart_disease"] = data["heart_disease"].astype(int)
    data = data.reset_index()

    data.to_csv(args.out, index=False)
    print(f"\n[SAVED] {len(data):,} rows x {data.shape[1]} columns -> {args.out}")
    print(f"Heart disease prevalence: {data['heart_disease'].mean():.2%}")
    print("\nMissing values (%):")
    print((data.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
