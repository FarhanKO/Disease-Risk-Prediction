"""
fields.py — How each model input is asked in the front end: its label, unit and range,
the answers it accepts, and which section of the questionnaire it belongs to.

The answers of a text input are read from the fitted pipeline's one-hot encoder, so
the choices offered are exactly the ones the model was trained on (the kidney data
spells "Other/Multiracial", the heart and lung data "Other/Multi-Racial").
"""

from functools import lru_cache

from sklearn.preprocessing import OneHotEncoder

# column -> (label, unit or help, min, max, step); the ranges are wide limits against typing errors
NUMERIC = {
    "age": ("Age", "years", 18, 100, 1),
    # Triage vital signs (emergency-department triage readings)
    "temperature_f": ("Temperature", "°F", 85.0, 110.0, 0.1),
    "heart_rate": ("Heart rate", "beats/min", 20, 250, 1),
    "respiratory_rate": ("Breathing rate", "breaths/min", 4, 80, 1),
    "systolic_bp": ("Systolic blood pressure", "mm Hg", 50, 300, 1),
    "diastolic_bp": ("Diastolic blood pressure", "mm Hg", 20, 200, 1),
    "oxygen_saturation": ("Oxygen saturation", "%", 50, 100, 1),
    "pain_score": ("Pain score", "0 = none, 10 = worst", 0, 10, 1),
    # Organ questionnaires
    "poverty_income_ratio": ("Family income ÷ poverty line", "0–5; 5 = five or more", 0.0, 5.0, 0.1),
    "bmi": ("Body-mass index", "kg/m²", 10.0, 80.0, 0.1),
    "waist_cm": ("Waist circumference", "cm", 40.0, 200.0, 0.5),
    "height_cm": ("Height", "cm", 120.0, 220.0, 0.5),
    "bp_systolic": ("Systolic blood pressure", "mm Hg", 50, 300, 1),
    "bp_diastolic": ("Diastolic blood pressure", "mm Hg", 20, 200, 1),
    "resting_heart_rate": ("Resting heart rate", "beats/min", 30, 200, 1),
    "total_cholesterol": ("Total cholesterol", "mg/dL", 50, 500, 1),
    "hdl_cholesterol": ("HDL cholesterol", "mg/dL", 10, 150, 1),
    "hba1c": ("HbA1c", "%", 3.0, 18.0, 0.1),
    "serum_creatinine": ("Serum creatinine", "mg/dL", 0.2, 15.0, 0.1),
    "uric_acid": ("Uric acid", "mg/dL", 1.0, 15.0, 0.1),
    "albumin_creatinine_ratio": ("Urine albumin ÷ creatinine", "mg/g", 0.0, 5000.0, 1.0),
    "white_blood_cells": ("White blood cells", "10³/µL", 1.0, 50.0, 0.1),
    "hemoglobin": ("Hemoglobin", "g/dL", 5.0, 20.0, 0.1),
    "rdw": ("Red cell distribution width", "%", 10.0, 30.0, 0.1),
    "blood_urea_nitrogen": ("Blood urea nitrogen", "mg/dL", 1, 150, 1),
    "albumin_serum": ("Serum albumin", "g/dL", 1.0, 6.0, 0.1),
    "phosphorus": ("Phosphorus", "mg/dL", 1.0, 10.0, 0.1),
    "bicarbonate": ("Bicarbonate", "mmol/L", 5, 45, 1),
    "calcium": ("Calcium", "mg/dL", 5.0, 15.0, 0.1),
    "cigarettes_per_day": ("Cigarettes per day", "0 if never; when quitting, if former", 0, 100, 1),
    "smoking_years": ("Years of regular smoking", "0 if never", 0, 80, 1),
    "serum_cotinine": ("Serum cotinine", "ng/mL (nicotine metabolite)", 0.0, 1000.0, 0.1),
    "eosinophils": ("Eosinophils", "10³/µL", 0.0, 5.0, 0.1),
    "neutrophils": ("Neutrophils", "10³/µL", 0.0, 30.0, 0.1),
    "lymphocytes": ("Lymphocytes", "10³/µL", 0.0, 20.0, 0.1),
}

# Numbers that are answered by choosing a description
ORDINAL = {
    "education_level": ("Education", {1: "Less than 9th grade", 2: "9th–11th grade", 3: "High school / GED",
                                      4: "Some college", 5: "College graduate"}),
    "general_health": ("General health", {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}),
    "household_smokers": ("Smokers living in the household", {0: "None", 1: "One", 2: "Two or more"}),
}

# Text inputs: the question asked; the answers come from the model
QUESTIONS = {
    "sex": "Sex", "gender": "Sex", "ethnicity": "Ethnicity", "smoking_status": "Smoking",
    "injury_reason": "Is it an injury, poisoning or a medicine's side effect?",
    "diabetes": "Diabetes (told by a doctor)", "diabetes_status": "Diabetes (told by a doctor)",
    "physically_active": "Moderate or vigorous exercise in a typical week",
    "hypertension_history": "Ever told you have high blood pressure",
    "high_cholesterol_history": "Ever told you have high cholesterol",
    "stroke_history": "Ever had a stroke",
    "family_history_chd": "Parent or sibling had a heart attack or angina before 50",
    "chest_pain_type": "Chest pain (Rose questionnaire)",
    "severe_chest_pain": "Ever had severe chest pain lasting 30 minutes or more",
    "sob_on_exertion": "Short of breath when hurrying or walking up a slight hill",
    "insulin_use": "Takes insulin", "diabetes_pills": "Takes diabetes pills",
    "asthma_history": "Ever told you have asthma", "heart_failure": "Ever told you have heart failure",
    "coronary_heart_disease": "Ever told you have coronary heart disease",
}

SECTIONS = {
    "About you": ["age", "gender", "ethnicity", "education_level", "poverty_income_ratio"],
    "Measurements": ["height_cm", "bmi", "waist_cm", "systolic_bp", "diastolic_bp", "bp_systolic", "bp_diastolic",
                     "resting_heart_rate"],
    "Lab results": ["total_cholesterol", "hdl_cholesterol", "hba1c", "serum_creatinine", "uric_acid",
                    "albumin_creatinine_ratio", "white_blood_cells", "hemoglobin", "rdw", "blood_urea_nitrogen",
                    "albumin_serum", "phosphorus", "bicarbonate", "calcium", "serum_cotinine", "eosinophils",
                    "neutrophils", "lymphocytes"],
}
OTHER_SECTION = "History and lifestyle"


def sections(columns: list) -> dict:
    """The model's input columns grouped by section, in its own column order within each."""
    placed = {c for group in SECTIONS.values() for c in group}
    grouped = {name: [c for c in columns if c in group] for name, group in SECTIONS.items()}
    grouped[OTHER_SECTION] = [c for c in columns if c not in placed]
    return {name: group for name, group in grouped.items() if group}


def label(column: str) -> str:
    if column in NUMERIC:
        return NUMERIC[column][0]
    if column in ORDINAL:
        return ORDINAL[column][0]
    return QUESTIONS.get(column, column.replace("_", " ").capitalize())


@lru_cache(maxsize=None)
def answers(organ: str) -> dict:
    """Text column -> the answers the organ's tabular model was trained on (from its fitted one-hot encoder)."""
    from common.registry import module

    model = module(organ, "tabular").MODEL.load()
    for encoder in _encoders(model._model):
        names = list(getattr(encoder, "feature_names_in_", []))
        if set(model.categorical_columns) <= set(names):
            return {c: [str(v) for v in values] for c, values in zip(names, encoder.categories_)}
    raise RuntimeError(f"No one-hot encoder over {model.categorical_columns} in the {organ} pipeline")


def _encoders(obj, seen=None, depth=0):
    """Every fitted OneHotEncoder inside a (calibrated) pipeline, depth first."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 12:
        return
    seen.add(id(obj))
    if isinstance(obj, OneHotEncoder) and hasattr(obj, "categories_"):
        yield obj
    for value in getattr(obj, "__dict__", {}).values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            for part in item if isinstance(item, tuple) else [item]:
                if hasattr(part, "__dict__"):
                    yield from _encoders(part, seen, depth + 1)
