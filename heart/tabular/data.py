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


