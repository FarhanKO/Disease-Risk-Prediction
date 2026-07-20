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
