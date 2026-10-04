"""
data.py — Data loading, feature engineering and preprocessing for the symptom
triage module (TRIAGE_NHAMCS.csv, see data/README.md).

The router has one binary head per organ group (heart, lung, kidney). All three
heads share these inputs, this feature engineering and this preprocessing.
Everything here mirrors notebooks/Symptom_Triage.ipynb exactly, so models trained
by src/train.py and by the notebook are interchangeable.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = MODULE_DIR / "data" / "TRIAGE_NHAMCS.csv"
VOCABULARY_PATH = MODULE_DIR / "data" / "symptom_vocabulary.csv"

ORGANS = ["heart", "lung", "kidney"]        # one binary target (head) each; a visit in none of them is "other"
METADATA_COLUMNS = ["visit_id", "survey_year", "icd_version", "diagnosis_1", "diagnosis_2", "diagnosis_3",
                    "primary_organ"]         # describe the survey and the outcome, not the patient's inputs

SYMPTOM_COLUMNS = ["symptom_1", "symptom_2", "symptom_3"]


def symptom_feature(name: str) -> str:
    """'Flank (side) or kidney pain' -> 'symptom_flank_side_or_kidney_pain' (a safe column name for every model)."""
    return "symptom_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# The symptom names a patient can choose from, in data/symptom_vocabulary.csv order (body system, then frequency)
SYMPTOMS = pd.read_csv(VOCABULARY_PATH)["symptom"].tolist()
SYMPTOM_FEATURES = [symptom_feature(s) for s in SYMPTOMS]

ONE_HOT_FEATURES = ["sex", "injury_reason", "diabetes"]
CATEGORICAL_FEATURES = ONE_HOT_FEATURES + SYMPTOM_COLUMNS       # every text input
VITAL_SIGNS = ["temperature_f", "heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp",
               "oxygen_saturation", "pain_score"]
RAW_NUMERICAL_FEATURES = ["age"] + VITAL_SIGNS
ENGINEERED_FEATURES = ["symptom_count", "pulse_pressure", "shock_index",
                       "fever", "low_oxygen", "fast_heart_rate", "fast_breathing"]
NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_FEATURES
# The anomaly gate looks at measurements only: age, vital signs and their two combinations
GATE_FEATURES = RAW_NUMERICAL_FEATURES + ["pulse_pressure", "shock_index"]

# The 14 columns a caller supplies, in the CSV's order. Only age, sex and one symptom are needed;
# anything else may be left blank and is then scored as typical (see build_pipeline)
RAW_INPUT_COLUMNS = ["age", "sex", "symptom_1", "symptom_2", "symptom_3", "injury_reason", "diabetes"] + VITAL_SIGNS

assert len(set(SYMPTOM_FEATURES)) == len(SYMPTOMS), "two symptom names map to the same column"


def add_custom_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    One 0/1 column per symptom (order-free: "Cough, Fever" and "Fever, Cough" are the same
    visit) and seven vital-sign features. Row-wise and stateless, so safe on any split.
    A name outside the vocabulary gets no column (src.predict rejects it before this point).
    """
    X_new = X_df.copy()
    listed = X_new[SYMPTOM_COLUMNS]

    codes = pd.Categorical(listed.to_numpy(dtype=object).ravel(), categories=SYMPTOMS).codes.reshape(listed.shape)
    multi_hot = np.zeros((len(listed), len(SYMPTOMS)))
    rows, slots = np.nonzero(codes >= 0)
    multi_hot[rows, codes[rows, slots]] = 1.0

    temperature, heart_rate = X_new["temperature_f"], X_new["heart_rate"]
    breathing, oxygen = X_new["respiratory_rate"], X_new["oxygen_saturation"]
    engineered = pd.DataFrame({
        "symptom_count": (codes >= 0).sum(axis=1).astype(float),
        # Pulse pressure (systolic - diastolic) and the shock index (heart rate / systolic), a triage
        # marker of circulatory strain; systolic floored at 40 mm Hg so the ratio stays finite
        "pulse_pressure": X_new["systolic_bp"] - X_new["diastolic_bp"],
        "shock_index": heart_rate / X_new["systolic_bp"].clip(lower=40),
        # Standard triage cut-offs; unknown when the vital sign is unknown
        "fever": (temperature >= 100.4).astype(float).where(temperature.notna()),
        "low_oxygen": (oxygen < 94).astype(float).where(oxygen.notna()),
        "fast_heart_rate": (heart_rate > 100).astype(float).where(heart_rate.notna()),
        "fast_breathing": (breathing > 20).astype(float).where(breathing.notna()),
    }, index=X_new.index)
    symptoms = pd.DataFrame(multi_hot, columns=SYMPTOM_FEATURES, index=X_new.index)
    return pd.concat([X_new, engineered, symptoms], axis=1)


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
    """Load TRIAGE_NHAMCS.csv and check that every symptom is in the vocabulary."""
    df = pd.read_csv(csv_path)
    unknown = set(df[SYMPTOM_COLUMNS].stack().dropna()) - set(SYMPTOMS)
    if unknown:
        raise ValueError(f"Symptoms missing from {VOCABULARY_PATH.name}: {sorted(unknown)[:5]}. Rebuild both files together.")
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs, and the three organ targets as a 0/1 DataFrame (one column per head)."""
    return df[RAW_INPUT_COLUMNS], df[ORGANS].astype(int)


def split_data(X: pd.DataFrame, Y: pd.DataFrame):
    """Stratified 80/20 split on the combination of the three labels — identical to the notebook's."""
    strata = Y.astype(str).agg("".join, axis=1)
    return train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata)


VITAL_DROPOUT = 0.5


def drop_vital_signs(X: pd.DataFrame, fraction: float = VITAL_DROPOUT, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    A copy of the training inputs with every vital sign and the pain score blanked for a random
    `fraction` of the visits. Emergency-department records almost always have vital signs, but a
    patient using the router usually has none; trained on both kinds of visit, each head stays
    accurate and calibrated with or without them (see the notebook's "Vital-Sign Dropout" section).
    Applied to the training split only; the test set is scored as recorded and with no vital signs.
    """
    X_new = X.copy()
    blank = np.random.default_rng(random_state).random(len(X_new)) < fraction
    X_new.loc[blank, VITAL_SIGNS] = np.nan
    return X_new


CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """
    Numbers: median-impute -> robust scale. Yes/No answers and sex: most-frequent impute ->
    one-hot. Symptoms: the 0/1 columns as they are. No missing-value indicators, on purpose:
    in the emergency department a missing vital sign is rare and says something about the visit,
    but a patient using the router usually has no vital signs at all. Imputing the typical value
    scores an unknown measurement as unremarkable instead of as a learned "missing" signal.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), NUMERICAL_FEATURES),
            ("categorical", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ONE_HOT_FEATURES),
            ("symptoms", "passthrough", SYMPTOM_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier) -> Pipeline:
    """feature engineering -> impute + scale / one-hot / symptoms -> classifier (one organ head)."""
    return Pipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("preprocess", build_preprocessor()),
        ("classifier", clone(classifier)),
    ])


def build_anomaly_gate(contamination: float = 0.01) -> Pipeline:
    """Stage 1 of the cascade: Isolation Forest on age and the vital signs (1 % contamination)."""
    return Pipeline(steps=[
        ("engineering", FunctionTransformer(add_custom_features)),
        ("select", ColumnTransformer([("numeric", "passthrough", GATE_FEATURES)])),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("detector", IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE)),
    ])
