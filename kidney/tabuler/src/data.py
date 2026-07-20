"""
data.py — Data loading, cleaning, feature engineering, and preprocessing
for the kidney disease (CKD) tabular module.
 
Ported 1:1 from Kidney_Disease.ipynb, with Colab Drive paths replaced by
local relative paths.
 
Note: the notebook drops any feature with >0.85 correlation to another
feature (a dynamic, data-dependent step). On the CKD_NHANES export this
dropped exactly one column: 'weight_kg' (highly correlated with height/BMI).
That's baked into load_raw_data() below as a static drop rather than
recomputed at runtime, so the feature schema — and therefore the fitted
preprocessor — stays stable across retrains. If you retrain on a materially
different export, re-check correlations before assuming this still holds.
"""
 
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
TARGET_COLUMN = "ckd_present"
STAGE_COLUMN = "ckd_stage"  # multi-class staging metadata, not used by the binary classifier
 
# Dropped for >0.85 correlation with height/BMI on the reference dataset (see module docstring)
CORRELATION_DROPPED_COLUMNS = ["weight_kg"]
 
CATEGORICAL_FEATURES = ["gender", "ethnicity"]
BASE_NUMERICAL_FEATURES = [
    "age", "education_level", "poverty_income_ratio", "bmi", "height_cm",
    "bp_systolic", "bp_diastolic", "serum_creatinine", "blood_urea_nitrogen",
    "albumin_serum", "phosphorus", "bicarbonate", "calcium", "uric_acid",
    "urine_creatinine", "urine_albumin", "albumin_creatinine_ratio",
    "diabetes_diagnosed", "insulin_use", "diabetes_pills", "ever_smoked",
    "current_smoker", "egfr",
]
ENGINEERED_FEATURES = ["pulse_pressure", "map", "bun_cr_ratio", "ca_p_product", "bmi_bp_interaction"]
NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
