"""
cascade.py — The whole system for one patient: the symptom triage router, then the
tabular and image models of every organ it routes to.

    symptoms, age, sex (+ vital signs)
            │
            ▼
      triage ──► route: e.g. ["heart", "lung"]        (anomaly gate "review" -> stop here)
            │
            ▼   for each routed organ (and any in also_check)
      organ tabular model   if at least half its inputs are known: risk band + referral
            │
            ▼
      organ image model     if an image was given for that organ: finding + Grad-CAM

    from common.cascade import run
    result = run({"age": 64, "sex": "Male", "symptoms": ["Chest pain"]},
                 tabular={"heart": {...heart questionnaire...}},
                 images={"heart": "ecg.png"})
    result.route, result.status, result.next_steps
    result.organs["heart"].tabular, result.organs["heart"].image      # common.Prediction each

Design choices:

- Inputs the patient already gave the router (age, sex, blood pressure, diabetes) are
  carried into each organ's questionnaire under that model's column names, unless the
  questionnaire gives its own value. They are listed in `filled_from_triage`.
- A tabular model runs only when at least MIN_INPUT_FRACTION of its inputs are known;
  below that its pipeline would mostly score imputed values. The step is then skipped
  and `missing_inputs` says what to ask.
- An image is read whenever one is given for a checked organ, also after a low tabular
  risk: the tabular models screen one chronic disease (heart disease, COPD, CKD) while the
  image models find others (pneumonia, tuberculosis, kidney stones, MI on an ECG), so a
  low tabular risk does not make the image redundant.
- Inputs for an organ the router did not reach are not scored; pass also_check=[organ]
  (e.g. kidney: chronic kidney disease rarely comes with symptoms the router can hear).
- The steps' probabilities are reported side by side, never multiplied into one: the
  router, tabular and image models were trained on different patients (NHAMCS, NHANES,
  public image sets), so no data validates them end to end.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .prediction import ACCEPTED, REVIEW, Prediction, _plain
from .registry import ORGANS, module, route as triage_route

MIN_INPUT_FRACTION = 0.5

# Triage input -> the same fact under each organ model's column name
SHARED_INPUTS = {
    "heart": {"age": "age", "sex": "gender", "systolic_bp": "systolic_bp", "diastolic_bp": "diastolic_bp",
              "diabetes": "diabetes_status"},
    "kidney": {"age": "age", "sex": "gender", "systolic_bp": "bp_systolic", "diastolic_bp": "bp_diastolic",
               "diabetes": "diabetes"},
    "lung": {"age": "age", "sex": "gender"},
}


@dataclass
class OrganResult:
    """
    organ               "heart", "lung" or "kidney"
    reason              why this organ was checked: routed by triage, or requested in also_check
    tabular             the organ's tabular Prediction; None when too few inputs were known
    image               the organ's image Prediction; None when no image was given
    filled_from_triage  inputs copied from the triage answers into the questionnaire
    missing_inputs      the tabular model's inputs nobody gave
    next_step           one line per step: its result, or what to provide next
    """

    organ: str
    reason: str
    tabular: Optional[Prediction] = None
    image: Optional[Prediction] = None
    filled_from_triage: dict = field(default_factory=dict)
    missing_inputs: list = field(default_factory=list)
    next_step: list = field(default_factory=list)

    @property
    def positive(self) -> bool:
        return any(step is not None and step.positive for step in (self.tabular, self.image))

    def to_dict(self, arrays: bool = True) -> dict:
        return {"organ": self.organ, "reason": self.reason, "positive": self.positive,
                "tabular": self.tabular.to_dict(arrays) if self.tabular else None,
                "image": self.image.to_dict(arrays) if self.image else None,
                "filled_from_triage": _plain(self.filled_from_triage, arrays),
                "missing_inputs": self.missing_inputs, "next_step": self.next_step}


@dataclass
class CascadeResult:
    """
    status      REVIEW when any step withheld or rejected its input (triage anomaly gate, a
                tabular anomaly gate, a low-confidence or rejected image); ACCEPTED otherwise
    positive    at least one organ step screened positive
    route       the organs checked, in order: triage's route, then also_check
    triage      the router's Prediction
    organs      organ -> OrganResult
    message     one line: the route and how many organs screened positive
    """

    status: str
    positive: bool
    route: list
    triage: Prediction
    organs: dict
    message: str
    unchecked: dict = field(default_factory=dict)     # organ -> why inputs given for it were not used

    @property
    def next_steps(self) -> list:
        return [f"{organ}: {line}" for organ, result in self.organs.items() for line in result.next_step]

    def to_dict(self, arrays: bool = True) -> dict:
        """Plain Python types; arrays=False drops the Grad-CAM arrays so it can go to json.dumps."""
        return {"status": self.status, "positive": self.positive, "route": self.route, "message": self.message,
                "triage": self.triage.to_dict(arrays),
                "organs": {o: r.to_dict(arrays) for o, r in self.organs.items()},
                "next_steps": self.next_steps, "unchecked": self.unchecked}


def _known(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _questionnaire(organ: str, patient: dict, answers: dict, columns: list) -> tuple[dict, dict, list]:
    """(the organ model's input row, values filled from triage, columns still missing)."""
    unknown = sorted(set(answers) - set(columns))
    if unknown:
        raise ValueError(f"Unknown {organ} tabular inputs {unknown}; the model takes {columns}")
    row = {c: answers.get(c) for c in columns}
    filled = {}
    for triage_column, column in SHARED_INPUTS[organ].items():
        if not _known(row[column]) and _known(patient.get(triage_column)):
            row[column] = filled[column] = patient[triage_column]
    missing = [c for c in columns if not _known(row[c])]
    return row, filled, missing


def _check_organs(name: str, organs) -> None:
    unknown = sorted(set(organs) - set(ORGANS))
    if unknown:
        raise ValueError(f"{name}: unknown organ(s) {unknown}; expected {list(ORGANS)}")


def _reason(organ: str, triage: Prediction, routed: bool) -> str:
    p = triage.details["probabilities"][organ]
    t = triage.details["thresholds"][organ]
    return (f"routed by triage (P = {p:.0%}, threshold {t:.0%})" if routed
            else f"requested in also_check (triage P = {p:.0%}, below its {t:.0%} threshold)")


def _check_organ(organ: str, reason: str, patient: dict, answers: Optional[dict], image,
                 min_fraction: float, explain: bool, ensemble: bool) -> OrganResult:
    tabular_module, image_module = module(organ, "tabular"), module(organ, "image")
    tabular_model, image_model = tabular_module.MODEL, image_module.MODEL
    result = OrganResult(organ=organ, reason=reason)

    row, result.filled_from_triage, result.missing_inputs = _questionnaire(
        organ, patient, answers or {}, tabular_model.input_columns)
    n_inputs, n_known = len(tabular_model.input_columns), len(tabular_model.input_columns) - len(result.missing_inputs)
    if n_known >= min_fraction * n_inputs:
        result.tabular = tabular_module.predict(row, explain=explain)
        result.next_step.append(f"{tabular_model.disease} risk: {result.tabular.message}")
    else:
        result.next_step.append(
            f"{tabular_model.disease} risk not scored: {n_known} of {n_inputs} inputs known, "
            f"{math.ceil(min_fraction * n_inputs)} needed. Missing: {', '.join(result.missing_inputs)}")

    if image is not None:
        result.image = image_module.predict(image, explain=explain, ensemble=ensemble)
        line = f"image: {result.image.message}"
        if result.image.status == ACCEPTED and result.image.positive:
            line += ": see a clinician to confirm"
        elif not result.image.scored:
            line += f". Upload {image_model.expected_input}"
        result.next_step.append(line)
    elif result.tabular is None or result.tabular.positive or result.tabular.status == REVIEW:
        result.next_step.append(f"provide {image_model.expected_input} for the image model")
    return result


def run(patient: dict, tabular: Optional[dict] = None, images: Optional[dict] = None, also_check=(),
        min_input_fraction: float = MIN_INPUT_FRACTION, explain: bool = True, ensemble: bool = False) -> CascadeResult:
    """
    One patient through the whole system.

    patient      the triage inputs: age, sex, 1-3 symptoms, optional diabetes / injury / vital signs
    tabular      organ -> that organ model's raw inputs (any subset; see each module's examples/patient.json)
    images       organ -> an image for that organ's image model (a path, bytes, an upload, a PIL image, an array)
    also_check   organs to check even if triage does not route to them
    """
    tabular, images, also_check = dict(tabular or {}), dict(images or {}), list(also_check)
    for name, organs in (("tabular", tabular), ("images", images), ("also_check", also_check)):
        _check_organs(name, organs)

    triage = triage_route(patient, explain=explain)
    given = sorted(set(tabular) | set(images))
    if triage.status != ACCEPTED:
        return CascadeResult(status=triage.status, positive=False, route=[], triage=triage, organs={},
                             message=f"Stopped at triage: {triage.message}",
                             unchecked={o: "stopped at triage" for o in given})

    routed = list(triage.details["route"])
    order = routed + [o for o in also_check if o not in routed]
    organs = {o: _check_organ(o, _reason(o, triage, o in routed), patient, tabular.get(o), images.get(o),
                              min_input_fraction, explain, ensemble)
              for o in order}

    steps = [s for r in organs.values() for s in (r.tabular, r.image) if s is not None]
    status = REVIEW if any(s.status != ACCEPTED for s in steps) else ACCEPTED
    positive = [o for o, r in organs.items() if r.positive]
    if not organs:
        message = triage.message
    else:
        message = (f"Checked {', '.join(order)}: "
                   + (f"positive screen for {', '.join(positive)}" if positive else "no positive screen")
                   + ("; some results need clinician review" if status == REVIEW else ""))
    unchecked = {o: "triage did not route here; add it to also_check to run it" for o in given if o not in organs}
    return CascadeResult(status=status, positive=bool(positive), route=order, triage=triage, organs=organs,
                         message=message, unchecked=unchecked)


def _print(result: CascadeResult) -> None:
    print(f"Triage: {result.triage.status} | {result.triage.message}")
    for organ, r in result.organs.items():
        print(f"\n{organ.upper()}: {r.reason}")
        if r.filled_from_triage:
            print(f"   from triage: {r.filled_from_triage}")
        for line in r.next_step:
            print(f"   - {line}")
    for organ, why in result.unchecked.items():
        print(f"\n{organ.upper()} not checked: {why}")
    print(f"\n{result.status.upper()}: {result.message}")


def main(argv=None):
    """`python -m common.cascade --input frontend/assets/cascade_patient.json [--image lung=xray.png] [--also-check kidney]`."""
    import argparse

    parser = argparse.ArgumentParser(description="Symptom triage, then the routed organs' tabular and image models.")
    parser.add_argument("--input", required=True,
                        help='JSON file: {"patient": {...triage inputs}, "tabular": {organ: {...}}, '
                             '"images": {organ: path}, "also_check": [organ, ...]}')
    parser.add_argument("--image", action="append", default=[], metavar="ORGAN=PATH",
                        help="An image for one organ's image model; overrides the file's.")
    parser.add_argument("--also-check", nargs="+", default=[], choices=ORGANS,
                        help="Organs to check even if triage does not route to them.")
    parser.add_argument("--heatmap-dir", default=None, help="Save the Grad-CAM overlays here.")
    parser.add_argument("--ensemble", action="store_true", help="Soft-vote each image module's ensemble.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args(argv)

    source = Path(args.input)
    payload = json.loads(source.read_text())
    images = {}
    for organ, path in payload.get("images", {}).items():      # relative to the JSON file, as written there
        images[organ] = str(path if Path(path).is_absolute() else (source.parent / path).resolve())
    for item in args.image:
        organ, _, path = item.partition("=")
        if not path:
            parser.error(f"--image takes ORGAN=PATH, got {item!r}")
        images[organ] = path

    result = run(payload["patient"], tabular=payload.get("tabular"), images=images,
                 also_check=list(dict.fromkeys(payload.get("also_check", []) + args.also_check)),
                 explain=args.heatmap_dir is not None or args.json, ensemble=args.ensemble)
    if args.json:
        print(json.dumps(result.to_dict(arrays=False), indent=2, default=str))
    else:
        _print(result)

    if args.heatmap_dir:
        from PIL import Image

        for organ, r in result.organs.items():
            if r.image is not None and "overlay" in r.image.explanation:
                out = Path(args.heatmap_dir) / f"{organ}_{Path(images[organ]).stem}_gradcam.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(r.image.explanation["overlay"]).save(out)
                print(f"Grad-CAM ({organ}) -> {out}")


if __name__ == "__main__":
    main()
