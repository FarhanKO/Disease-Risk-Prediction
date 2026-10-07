"""
data.py — Data loading, feature engineering, and preprocessing for the lung disease
(COPD) tabular module (COPD_NHANES.csv, see data/README.md).

Everything here mirrors notebooks/Lung_Disease.ipynb, so models trained by src/train.py
and by the notebook are interchangeable.
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
MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = MODULE_DIR / "data" / "COPD_NHANES.csv"

TARGET_COLUMN = "copd_present"
METADATA_COLUMNS = ["participant_id", "survey_cycle"]    # describe the survey, not the patient

# The notebook's > 0.85 correlation filter drops waist_cm (r = 0.89 with BMI). It is
# baked in as a static drop so the feature schema stays stable across retrains.
CORRELATION_DROPPED_COLUMNS = ["waist_cm"]

CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "smoking_status", "asthma_history", "heart_failure",
    "coronary_heart_disease", "sob_on_exertion",
]
RAW_NUMERICAL_FEATURES = [
    "age", "education_level", "poverty_income_ratio", "bmi", "height_cm", "cigarettes_per_day",
    "smoking_years", "household_smokers", "serum_cotinine", "general_health", "eosinophils",
    "neutrophils", "lymphocytes", "hemoglobin",
]
ENGINEERED_FEATURES = ["pack_years", "log_cotinine", "nlr", "comorbidity_count"]
# The log replaces raw cotinine, which spans five orders of magnitude
NUMERICAL_FEATURES = [c for c in RAW_NUMERICAL_FEATURES if c != "serum_cotinine"] + ENGINEERED_FEATURES

# The 21 columns a caller must supply, in the CSV's order
RAW_INPUT_COLUMNS = [
    "age", "gender", "ethnicity", "education_level", "poverty_income_ratio", "bmi", "height_cm",
    "smoking_status", "cigarettes_per_day", "smoking_years", "household_smokers", "serum_cotinine",
    "asthma_history", "heart_failure", "coronary_heart_disease", "sob_on_exertion", "general_health",
    "eosinophils", "neutrophils", "lymphocytes", "hemoglobin",
]


def add_custom_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's four clinical features. Row-wise and stateless, so safe on any split."""
    X_new = X_df.copy()

    # Cumulative tobacco dose: 1 pack-year = 20 cigarettes/day for one year (0 for never-smokers)
    X_new["pack_years"] = X_new["cigarettes_per_day"] * X_new["smoking_years"] / 20

    # Log scale for cotinine; clipped at the assay's reporting floor so the log is always defined
    X_new["log_cotinine"] = np.log10(X_new["serum_cotinine"].clip(lower=0.01))

    # Neutrophil-to-lymphocyte ratio (lymphocytes floored to avoid division by ~0)
    X_new["nlr"] = X_new["neutrophils"] / X_new["lymphocytes"].clip(lower=0.1)

    # Number of cardiopulmonary comorbidities; unknown only if all three answers are unknown
    comorbidities = ["asthma_history", "heart_failure", "coronary_heart_disease"]
    answered = X_new[comorbidities].notna().any(axis=1)
    X_new["comorbidity_count"] = (X_new[comorbidities] == "Yes").sum(axis=1).where(answered)

    return X_new


def register_notebook_functions():
    """
    The notebook pickles its pipelines with a reference to __main__.add_custom_features.
    Exposing the same function on __main__ lets scripts load those files.
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
    """Load COPD_NHANES.csv, drop unlabeled rows, survey metadata and the correlated column."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df.drop(columns=METADATA_COLUMNS + CORRELATION_DROPPED_COLUMNS, errors="ignore")


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[RAW_INPUT_COLUMNS], df[TARGET_COLUMN].astype(int)


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split — identical to the notebook's."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(classifier, oversample: bool = True) -> ImbPipeline:
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
