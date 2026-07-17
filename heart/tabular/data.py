from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "num"

# Raw columns kept as model inputs (id and target are dropped upstream)
BASE_NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
ENGINEERED_FEATURES = ["max_hr_ratio", "bp_hr_index", "age_st_interaction"]
NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def add_custom_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive the three clinical features used throughout the pipeline."""
    X_out = X.copy()

    expected_max_hr = 220 - X_out["age"]
    X_out["max_hr_ratio"] = X_out["thalch"] / (expected_max_hr + 1e-5)
    X_out["bp_hr_index"] = X_out["trestbps"] / (X_out["thalch"] + 1e-5)
    X_out["age_st_interaction"] = X_out["age"] * X_out["oldpeak"]

    return X_out


FEATURE_ENGINEERING = FunctionTransformer(add_custom_features)


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, drop rows with no target, drop the id column."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df.drop(columns=["id"], errors="ignore")
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into raw features and binary target."""
    X = df.drop(columns=[TARGET_COLUMN, *ENGINEERED_FEATURES], errors="ignore")
    y = (df[TARGET_COLUMN] > 0).astype(int)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split, matching the notebook's split exactly."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


