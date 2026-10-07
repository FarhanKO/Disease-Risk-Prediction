"""
data.py — Loading, target correction, cleaning, feature engineering, and the
preprocessing pipeline for the kidney disease (CKD) tabular module.

Mirrors notebooks/Kidney Disease.ipynb:

1. Target: KDIGO CKD (eGFR < 60 mL/min/1.73m2 or urine albumin-to-creatinine
   ratio >= 30 mg/g), adults 20+ whose status is determinable. The raw
   export's `ckd_present` labelled every participant without kidney labs
   (mostly children) and everyone with eGFR 60-89 as CKD, so it is rebuilt.
2. The five label-defining lab values are removed from the inputs (target
   leakage): eGFR and ACR define the label, eGFR is computed from serum
   creatinine, ACR from urine albumin / urine creatinine.
3. NHANES answer codes (7/9 = refused / don't know) and skip-patterns (the
   insulin question is only asked of people with diabetes) are resolved.
4. `weight_kg` is dropped: r = 0.89 with BMI, removed by the notebook's
   > 0.85 correlation filter. It is baked in here as a static drop so the
   feature schema stays stable across retrains.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "ckd_present"
MIN_AGE = 20

MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = MODULE_DIR / "data" / "CKD_NHANES.csv"

LABEL_DEFINING_FEATURES = ["egfr", "albumin_creatinine_ratio", "serum_creatinine", "urine_albumin", "urine_creatinine"]
CORRELATION_DROPPED_COLUMNS = ["weight_kg"]

BASE_NUMERICAL_FEATURES = [
    "age", "education_level", "poverty_income_ratio", "bmi", "height_cm",
    "bp_systolic", "bp_diastolic", "blood_urea_nitrogen", "albumin_serum",
    "phosphorus", "bicarbonate", "calcium", "uric_acid",
]
CATEGORICAL_FEATURES = ["gender", "ethnicity", "insulin_use", "diabetes_pills", "diabetes", "smoking_status"]
ENGINEERED_FEATURES = ["pulse_pressure", "map", "ca_p_product", "bmi_bp_interaction"]
NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES

# What a caller (form, API, cascade) must supply
RAW_INPUT_COLUMNS = BASE_NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def add_custom_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """The four clinical features; a missing input column raises instead of silently becoming 0."""
    X_new = X_df.copy()

    # Cardiovascular features
    X_new["pulse_pressure"] = X_new["bp_systolic"] - X_new["bp_diastolic"]
    X_new["map"] = (X_new["bp_systolic"] + 2 * X_new["bp_diastolic"]) / 3

    # CKD mineral and bone disorder (CKD-MBD) marker
    X_new["ca_p_product"] = X_new["calcium"] * X_new["phosphorus"]

    # Metabolic / hemodynamic interaction
    X_new["bmi_bp_interaction"] = X_new["bmi"] * X_new["bp_systolic"]

    return X_new


FEATURE_ENGINEERING = FunctionTransformer(add_custom_features)


def kdigo_label(df: pd.DataFrame) -> pd.Series:
    """1 = CKD, 0 = no CKD, NaN = not determinable from the available tests."""
    egfr, acr = df["egfr"], df["albumin_creatinine_ratio"]
    determinable = (egfr.notna() & acr.notna()) | (egfr < 60) | (acr >= 30)
    label = ((egfr < 60) | (acr >= 30)).astype(float)
    return label.where(determinable)


def clean_nhanes_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve NHANES answer codes and skip-patterns into readable categories."""
    df = df.copy()

    # SAS transport stores 0 as 5.4e-79
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols].abs() < 1e-9, 0.0)

    # Education 1-5; 7 = refused, 9 = don't know
    df["education_level"] = df["education_level"].where(df["education_level"].between(1, 5))

    # Diabetes status (DIQ010): 1 yes, 2 no, 3 borderline
    df["diabetes"] = df["diabetes_diagnosed"].map({1: "Yes", 2: "No", 3: "Borderline"})

    # Insulin / pills are only asked of people with (pre)diabetes: "not asked" means "No"
    diabetes_known = df["diabetes"].notna()
    for col in ["insulin_use", "diabetes_pills"]:
        coded = df[col].map({1: "Yes", 2: "No"})             # 9 = don't know -> stays missing
        coded[df[col].isna() & diabetes_known] = "No"         # skipped question -> "No"
        df[col] = coded

    # Smoking: >= 100 cigarettes in life (SMQ020) + smoking now (SMQ040: 1 every day, 2 some days, 3 not at all)
    ever, now = df["ever_smoked"], df["current_smoker"]
    df["smoking_status"] = np.select(
        [ever == 2, (ever == 1) & now.isin([1, 2]), (ever == 1) & (now == 3)],
        ["Never", "Current", "Former"], default=None)

    return df.drop(columns=["diabetes_diagnosed", "ever_smoked", "current_smoker"])


def load_raw_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Raw CKD_NHANES export -> adults 20+ with a KDIGO label, cleaned, leakage-free."""
    df = pd.read_csv(csv_path)
    df[TARGET_COLUMN] = kdigo_label(df)
    df = df[(df["age"] >= MIN_AGE) & df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df = df.drop(columns=[*LABEL_DEFINING_FEATURES, "ckd_stage", "participant_id", *CORRELATION_DROPPED_COLUMNS],
                 errors="ignore")
    return clean_nhanes_codes(df)


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into raw input columns and the binary target."""
    return df[RAW_INPUT_COLUMNS], df[TARGET_COLUMN].astype(int)


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split, matching the notebook's split exactly."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_imputation() -> ColumnTransformer:
    """Scale -> KNN-impute the numeric block (scaling first keeps KNN distances fair); mode-impute categoricals."""
    numeric = Pipeline(steps=[
        ("scaler", RobustScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num_transform", numeric, NUMERICAL_FEATURES),
            ("cat_transform", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_encoding() -> ColumnTransformer:
    """One-hot encode the categoricals; the numeric block is already scaled and imputed."""
    return ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_smote() -> SMOTENC:
    """SMOTE for mixed data: interpolates numeric features, copies categories from real neighbours."""
    return SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=RANDOM_STATE)


def build_pipeline(estimator, oversample: bool = True) -> ImbPipeline:
    """feature engineering -> scale + impute -> (SMOTENC) -> one-hot -> classifier."""
    return ImbPipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("imputation", build_imputation()),
        ("smote", build_smote() if oversample else "passthrough"),
        ("encoding", build_encoding()),
        ("classifier", estimator),
    ])
