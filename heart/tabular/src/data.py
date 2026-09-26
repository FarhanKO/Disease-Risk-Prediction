"""
data.py — Data loading, feature engineering, and preprocessing for the heart
disease tabular module (HEART_NHANES.csv, see data/README.md).

Everything here mirrors notebook/Heart_Diseases.ipynb exactly, so models trained
by src/train.py and by the notebook are interchangeable.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "HEART_NHANES.csv"

TARGET_COLUMN = "heart_disease"
METADATA_COLUMNS = ["participant_id", "survey_cycle"]    # describe the survey, not the patient

CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "smoking_status", "physically_active", "hypertension_history",
    "high_cholesterol_history", "diabetes_status", "stroke_history", "family_history_chd",
    "chest_pain_type", "severe_chest_pain", "sob_on_exertion",
]
RAW_NUMERICAL_FEATURES = [
    "age", "education_level", "poverty_income_ratio", "bmi", "waist_cm", "height_cm",
    "systolic_bp", "diastolic_bp", "resting_heart_rate", "total_cholesterol", "hdl_cholesterol",
    "hba1c", "serum_creatinine", "uric_acid", "albumin_creatinine_ratio", "white_blood_cells",
    "hemoglobin", "rdw", "general_health",
]
ENGINEERED_FEATURES = [
    "pulse_pressure", "non_hdl_cholesterol", "tc_hdl_ratio", "waist_to_height_ratio",
    "egfr", "log_acr", "risk_factor_count",
]
# The log replaces the raw albumin-creatinine ratio, which spans five orders of magnitude
NUMERICAL_FEATURES = [c for c in RAW_NUMERICAL_FEATURES if c != "albumin_creatinine_ratio"] + ENGINEERED_FEATURES

# The 31 columns a caller must supply, in the CSV's order
RAW_INPUT_COLUMNS = [
    "age", "gender", "ethnicity", "education_level", "poverty_income_ratio", "bmi", "waist_cm",
    "height_cm", "systolic_bp", "diastolic_bp", "resting_heart_rate", "total_cholesterol",
    "hdl_cholesterol", "hba1c", "serum_creatinine", "uric_acid", "albumin_creatinine_ratio",
    "white_blood_cells", "hemoglobin", "rdw", "smoking_status", "physically_active",
    "hypertension_history", "high_cholesterol_history", "diabetes_status", "stroke_history",
    "family_history_chd", "general_health", "chest_pain_type", "severe_chest_pain", "sob_on_exertion",
]


def add_custom_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's seven clinical features. Row-wise and stateless, so safe on any split."""
    X_new = X_df.copy()

    # Arterial stiffness: the gap between systolic and diastolic pressure
    X_new["pulse_pressure"] = X_new["systolic_bp"] - X_new["diastolic_bp"]

    # Lipid ratios (HDL floored at 10 mg/dL to avoid division by ~0)
    X_new["non_hdl_cholesterol"] = X_new["total_cholesterol"] - X_new["hdl_cholesterol"]
    X_new["tc_hdl_ratio"] = X_new["total_cholesterol"] / X_new["hdl_cholesterol"].clip(lower=10)

    # Central obesity
    X_new["waist_to_height_ratio"] = X_new["waist_cm"] / X_new["height_cm"]

    # Kidney function: race-free CKD-EPI 2021 creatinine equation (mL/min/1.73 m²)
    female = X_new["gender"].eq("Female")
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    creatinine_ratio = X_new["serum_creatinine"] / kappa
    X_new["egfr"] = (142 * np.minimum(creatinine_ratio, 1) ** alpha * np.maximum(creatinine_ratio, 1) ** -1.200
                     * 0.9938 ** X_new["age"] * np.where(female, 1.012, 1.0))

    # Log scale for albuminuria; clipped at 0.1 mg/g so the log is always defined
    X_new["log_acr"] = np.log10(X_new["albumin_creatinine_ratio"].clip(lower=0.1))

    # Number of classic modifiable risk factors; unknown only if all four answers are unknown
    risk_answers = X_new[["hypertension_history", "high_cholesterol_history", "diabetes_status", "smoking_status"]]
    risk_present = pd.DataFrame({
        "hypertension": X_new["hypertension_history"] == "Yes",
        "high_cholesterol": X_new["high_cholesterol_history"] == "Yes",
        "diabetes": X_new["diabetes_status"] == "Yes",
        "current_smoker": X_new["smoking_status"] == "Current",
    })
    X_new["risk_factor_count"] = risk_present.sum(axis=1).where(risk_answers.notna().any(axis=1))

    return X_new


def register_notebook_functions():
    """
    The notebook pickles its pipelines with a reference to __main__.add_custom_features.
    Exposing the same function on __main__ lets scripts and Streamlit load those files.
    """
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "add_custom_features"):
        main_module.add_custom_features = add_custom_features


def keep_workers_light():
    """
    scikit-learn copies the warning filters into every parallel worker. torch and lightning
    register filters for their own warning classes, so each worker would import torch + CUDA
    (~1.5 GB of RAM). Keep only light libraries' filters before any parallel work.
    """
    light_libraries = ("builtins", "warnings", "numpy", "scipy", "pandas", "sklearn")
    warnings.filters[:] = [f for f in warnings.filters if f[2].__module__.split(".")[0] in light_libraries]


def load_raw_data(csv_path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load HEART_NHANES.csv, drop unlabeled rows and survey metadata."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df.drop(columns=METADATA_COLUMNS, errors="ignore")


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[RAW_INPUT_COLUMNS], df[TARGET_COLUMN].astype(int)


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split — identical to the notebook's."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(classifier, oversample: bool = False) -> ImbPipeline:
    """
    feature engineering -> scale + KNN-impute (numeric) / mode-impute (categorical)
    -> optional SMOTENC -> one-hot -> classifier. Scaling comes before KNN imputation
    so no single unit dominates the neighbour search.
    """
    imputation = ColumnTransformer(
        transformers=[
            ("num_transform", Pipeline(steps=[
                ("scaler", RobustScaler()),
                ("imputer", KNNImputer(n_neighbors=5)),
            ]), NUMERICAL_FEATURES),
            ("cat_transform", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")               # keep column names so SMOTENC can find the categoricals

    encoding = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    smote = SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=RANDOM_STATE)
    return ImbPipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("imputation", imputation),
        ("smote", smote if oversample else "passthrough"),
        ("encoding", encoding),
        ("classifier", clone(classifier)),
    ])


def build_anomaly_gate(contamination: float = 0.01) -> Pipeline:
    """Stage 1 of the cascade: Isolation Forest on the numeric features (1 % contamination)."""
    return Pipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("select", ColumnTransformer([("numeric", "passthrough", NUMERICAL_FEATURES)])),
        ("scaler", RobustScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
        ("detector", IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE)),
    ])
