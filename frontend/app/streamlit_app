"""
streamlit_app.py — One front end for the whole cascade (common/cascade.py).

    1. Symptoms   age, sex, one to three symptoms, optional vital signs -> the triage router
    2. Organs     for each routed organ (and any added): its questionnaire and an image upload
    3. Results    each organ's tabular risk and image finding side by side, with the next steps

Run from the repo root:
    streamlit run frontend/app/streamlit_app.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # streamlit puts only this folder on the path
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from common import ACCEPTED, REJECTED, REVIEW
from common.cascade import MIN_INPUT_FRACTION, SHARED_INPUTS, run
from common.registry import ORGANS, module, route
from frontend.components import fields

EXAMPLE_PATH = ROOT / "frontend" / "assets" / "cascade_patient.json"
ORGAN_NAMES = {"heart": "Heart", "lung": "Lung", "kidney": "Kidney"}
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
TRIAGE_OPTIONAL = ["diabetes", "injury_reason"]
VITAL_SIGNS = ["temperature_f", "heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp",
               "oxygen_saturation", "pain_score"]

st.set_page_config(page_title="Disease Risk Cascade", page_icon="🩺", layout="wide")


# ---------- models (imported once per process; the modules cache their own weights) ----------

@st.cache_resource
def organ_info() -> dict:
    """organ -> what its two models screen for and expect, without loading any weights."""
    return {o: {"disease": module(o, "tabular").MODEL.disease,
                "columns": module(o, "tabular").MODEL.input_columns,
                "image": module(o, "image").MODEL.expected_input}
            for o in ORGANS}


@st.cache_resource
def symptom_names() -> list:
    return list(module("triage", "tabular").data.SYMPTOMS)


# ---------- widgets ----------

def sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def wkey(key: str) -> str:
    """The widget key of a form input. Loading the example or starting over bumps the version, so
    every input is a new widget: the browser keeps a widget's value by key and would resend the old one."""
    return f"{st.session_state.get('form_version', 0)}|{key}"


def _default(key: str):
    """A widget's starting value: from the loaded example, else blank."""
    return st.session_state.get("defaults", {}).get(key)


def _index(key: str, options: list):
    value = _default(key)
    return options.index(value) if value in options else None


def number(column: str, key: str):
    name, unit, low, high, step = fields.NUMERIC[column]
    cast = float if isinstance(step, float) else int
    value = _default(key)
    decimals = len(str(step).split(".")[1]) if cast is float else 0
    return st.number_input(f"{name} ({unit})", min_value=cast(low), max_value=cast(high), step=cast(step),
                           value=None if value is None else cast(value), format=f"%.{decimals}f" if decimals else None,
                           placeholder="unknown", key=wkey(key))


def choice(column: str, key: str, options: list, format_func=str):
    return st.selectbox(fields.label(column), options, index=_index(key, options), format_func=format_func,
                        placeholder="unknown", key=wkey(key))


def radio(label: str, key: str, options: list):
    return st.radio(label, options, index=_index(key, options), horizontal=True, key=wkey(key))


def ask(organ: str, column: str):
    """One questionnaire input of an organ's tabular model."""
    key = f"{organ}:{column}"
    if column in fields.ORDINAL:
        labels = fields.ORDINAL[column][1]
        return choice(column, key, list(labels), format_func=lambda v: labels[v])
    if column in fields.NUMERIC:
        return number(column, key)
    return choice(column, key, fields.answers(organ)[column])


# ---------- example patient ----------

def load_example():
    """Fill every form from frontend/assets/cascade_patient.json (a callback: runs before the widgets)."""
    payload = json.loads(EXAMPLE_PATH.read_text())
    defaults = {f"t:{column}": value for column, value in payload["patient"].items()}
    for organ, answers in payload.get("tabular", {}).items():
        defaults.update({f"{organ}:{column}": value for column, value in answers.items()})
    defaults["also_check"] = payload.get("also_check", [])
    new_form(defaults)
    reset_results()


def reset_results():
    for key in ("triage", "patient", "result", "uploads"):
        st.session_state.pop(key, None)


def new_form(defaults: dict):
    """Drop every form input and start new ones (new keys) with these starting values."""
    version = st.session_state.get("form_version", 0) + 1
    for key in [k for k in st.session_state if "|" in k]:
        del st.session_state[key]
    st.session_state.form_version, st.session_state.defaults = version, defaults


def start_over():
    new_form({})
    reset_results()


# ---------- step 1: symptoms -> triage ----------

def triage_form():
    st.subheader("1 · Symptoms")
    with st.form("triage_form"):
        left, right = st.columns([1, 2])
        with left:
            age = st.number_input("Age (years)", min_value=18, max_value=100, step=1, placeholder="required",
                                  value=_default("t:age"), key=wkey("t:age"))
            sex = radio("Sex", "t:sex", ["Female", "Male"])
        with right:
            symptoms = st.multiselect("Symptoms, most important first (up to 3)", symptom_names(),
                                      default=[s for s in _default("t:symptoms") or [] if s in symptom_names()],
                                      max_selections=3, key=wkey("t:symptoms"), placeholder="Type to search 128 symptoms")
            a, b = st.columns(2)
            with a:
                diabetes = radio(fields.label("diabetes"), "t:diabetes", ["Yes", "No"])
            with b:
                injury = radio(fields.label("injury_reason"), "t:injury_reason", ["Yes", "No"])
        with st.expander("Vital signs (optional: leave blank if not measured)"):
            columns = st.columns(4)
            vitals = {}
            for i, column in enumerate(VITAL_SIGNS):
                with columns[i % 4]:
                    vitals[column] = number(column, f"t:{column}")
        submitted = st.form_submit_button("Check which organs to look at", type="primary")

    if not submitted:
        return
    missing = [name for name, value in (("age", age), ("sex", sex)) if value is None]
    if not symptoms:
        missing.append("at least one symptom")
    if missing:
        st.error("Please give " + ", ".join(missing) + ".")
        return
    patient = {"age": age, "sex": sex, "symptoms": symptoms, "diabetes": diabetes, "injury_reason": injury,
               **vitals}
    patient = {k: v for k, v in patient.items() if v is not None}
    reset_results()
    with st.spinner("Routing…"):
        try:
            st.session_state.triage = route(patient)
        except ValueError as error:
            st.error(str(error))
            return
    st.session_state.patient = patient


def show_factors(factors: list, value_header: str = "This patient"):
    if not factors:
        st.caption("No input moved the probability away from a typical patient's.")
        return
    table = pd.DataFrame({
        "Input": [fields.label(f["feature"]) for f in factors],
        value_header: [f["value"] for f in factors],
        "Typical": ["none" if f["typical"] is None else f["typical"] for f in factors],
        "Effect": [f"{f['effect'] * 100:+.1f} pts" for f in factors],
    })
    st.dataframe(table.astype(str), hide_index=True, width="stretch")


def triage_result():
    triage = st.session_state.triage
    if triage.status != ACCEPTED:
        st.error(triage.message)
        return False

    route_names = triage.details["route"]
    (st.warning if route_names else st.success)(f"**{triage.label}** · {triage.message}")
    probabilities, thresholds = triage.details["probabilities"], triage.details["thresholds"]
    for column, organ in zip(st.columns(3), ORGANS):
        with column:
            routed = organ in route_names
            st.metric(f"{ORGAN_NAMES[organ]}{' ✓ routed' if routed else ''}", f"{probabilities[organ]:.0%}",
                      f"threshold {thresholds[organ]:.0%}", delta_color="off")
            st.progress(min(probabilities[organ] / max(thresholds[organ], 1e-9) / 2, 1.0))
    st.caption("Each bar is half full at its organ's routing threshold; past half, the organ is routed.")
    with st.expander("Why this route"):
        for organ, factors in triage.explanation.get("factors", {}).items():
            st.markdown(f"**{ORGAN_NAMES[organ]}**: inputs that moved P({organ}) most")
            show_factors(factors)
        st.caption(triage.explanation.get("method", ""))
    return True


# ---------- step 2: organ questionnaires and images ----------

def organ_forms():
    st.subheader("2 · Organ checks")
    triage, patient = st.session_state.triage, st.session_state.patient
    routed = list(triage.details["route"])
    others = [o for o in ORGANS if o not in routed]
    extra = st.multiselect("Also check", others, format_func=ORGAN_NAMES.get, key=wkey(f"also_check:{'+'.join(routed)}"),
                           default=[o for o in _default("also_check") or [] if o in others],
                           help="Organs triage did not route to. Chronic kidney disease, for one, "
                                "rarely comes with symptoms the router can hear.")
    order = routed + extra
    if not order:
        st.info("Triage found no heart, lung or kidney signal. Add an organ above to check it anyway.")
        return

    info = organ_info()
    with st.form("organ_form"):
        tabs = st.tabs([ORGAN_NAMES[o] for o in order])
        shown = {}
        for organ, tab in zip(order, tabs):
            with tab:
                shown[organ] = questionnaire(organ, patient, info[organ])
        ensemble = st.toggle("Image ensemble (soft-vote several networks: slower)", key="ensemble")
        submitted = st.form_submit_button("Run the organ checks", type="primary")

    if submitted:
        tabular = {o: {c: st.session_state.get(wkey(f"{o}:{c}")) for c in columns} for o, columns in shown.items()}
        tabular = {o: {c: v for c, v in answers.items() if v is not None} for o, answers in tabular.items()}
        uploads = {o: st.session_state.get(wkey(f"{o}:image")) for o in order}
        uploads = {o: u for o, u in uploads.items() if u is not None}
        images = {o: u.getvalue() for o, u in uploads.items()}
        with st.spinner("Running the organ models… (the first image loads its network: up to a minute)"):
            st.session_state.result = run(patient, tabular=tabular, images=images, also_check=extra,
                                          ensemble=ensemble)
        st.session_state.uploads = images


def questionnaire(organ: str, patient: dict, info: dict) -> list:
    """An organ's questionnaire and image upload; returns the columns asked."""
    columns = info["columns"]
    from_triage = {column: patient[source] for source, column in SHARED_INPUTS[organ].items() if source in patient}
    st.markdown(f"**{sentence(info['disease'])} risk.** Answer what you know: the model needs at least "
                f"{math.ceil(len(columns) * MIN_INPUT_FRACTION)} of its {len(columns)} inputs and fills in the rest.")
    if from_triage:
        st.caption("From step 1: " + ", ".join(f"{fields.label(c).lower()} {v}" for c, v in from_triage.items()))

    asked = []
    for section, group in fields.sections(columns).items():
        group = [c for c in group if c not in from_triage]
        if not group:
            continue
        with st.expander(section, expanded=section == "About you"):
            grid = st.columns(3)
            for i, column in enumerate(group):
                with grid[i % 3]:
                    ask(organ, column)
            asked += group
    st.file_uploader(f"Image (optional): {info['image']}", type=IMAGE_TYPES, key=wkey(f"{organ}:image"))
    return asked


# ---------- step 3: results ----------

def results():
    result = st.session_state.result
    st.subheader("3 · Results")
    if result.status == REVIEW:
        st.warning(f"**Needs clinician review.** {result.message}")
    elif result.positive:
        st.warning(f"**{result.message}**")
    else:
        st.success(f"**{result.message}**")

    for organ, organ_result in result.organs.items():
        with st.container(border=True):
            st.markdown(f"### {ORGAN_NAMES[organ]}")
            st.caption(sentence(organ_result.reason))
            left, right = st.columns(2)
            with left:
                tabular_card(organ_result)
            with right:
                image_card(organ, organ_result)

    for organ, why in result.unchecked.items():
        st.caption(f"{ORGAN_NAMES[organ]} inputs were not used: {why}.")

    st.markdown("#### Next steps")
    st.markdown("\n".join(f"- {line}" for line in result.next_steps) or "- None")
    st.download_button("Download the full result (JSON)", mime="application/json", file_name="cascade_result.json",
                       data=json.dumps(result.to_dict(arrays=False), indent=2, default=str))


def tabular_card(organ_result):
    st.markdown("**Risk questionnaire**")
    prediction = organ_result.tabular
    if prediction is None:
        total, missing = len(organ_info()[organ_result.organ]["columns"]), organ_result.missing_inputs
        st.info(f"Not scored: {total - len(missing)} of {total} inputs known, "
                f"{math.ceil(total * MIN_INPUT_FRACTION)} needed. Add some of these in step 2 and run again: "
                + ", ".join(fields.label(c) for c in missing))
        return
    if not prediction.scored:
        st.warning(prediction.message)
        return
    st.metric(prediction.label, f"{prediction.probability:.0%}",
              f"referral threshold {prediction.details['threshold']:.0%}", delta_color="off")
    st.write(prediction.message)
    if organ_result.filled_from_triage:
        st.caption("Taken from step 1: " + ", ".join(fields.label(c).lower() for c in organ_result.filled_from_triage))
    with st.expander("What moved this risk"):
        show_factors(prediction.explanation.get("factors", []))
        st.caption(prediction.explanation.get("method", ""))


def image_card(organ: str, organ_result):
    st.markdown("**Image**")
    prediction = organ_result.image
    if prediction is None:
        line = next((s for s in organ_result.next_step if s.startswith("provide")), None)
        st.caption(sentence(line) if line else "No image given.")
        return
    if prediction.status == REJECTED:
        st.error(prediction.message)
        st.image(st.session_state.uploads[organ], width=240)
        return
    (st.warning if prediction.status == REVIEW or prediction.positive else st.success)(prediction.message)
    original, overlay = st.columns(2)
    with original:
        st.image(st.session_state.uploads[organ], caption="Uploaded", width="stretch")
    with overlay:
        if "overlay" in prediction.explanation:
            st.image(prediction.explanation["overlay"], caption="Grad-CAM: where the model looked", width="stretch")
    probabilities = pd.Series(prediction.details["probabilities"], name="probability").sort_values(ascending=False)
    st.bar_chart(probabilities, horizontal=True, height=40 + 28 * len(probabilities))


# ---------- page ----------

with st.sidebar:
    st.title("🩺 Disease Risk Cascade")
    st.markdown("Symptoms → which organs to check → each organ's **risk questionnaire** and **image model**.")
    st.button("Load example patient", on_click=load_example, width="stretch")
    st.button("Start over", on_click=start_over, width="stretch")
    st.divider()
    st.caption("Research models trained on public datasets (CDC NHAMCS, NHANES, public image sets), "
               "not a medical device. The steps were trained on different patients, so their results are "
               "shown side by side and never combined into one probability. If you feel unwell, see a doctor.")

triage_form()
if "triage" in st.session_state:
    if triage_result():
        organ_forms()
if "result" in st.session_state:
    results()
