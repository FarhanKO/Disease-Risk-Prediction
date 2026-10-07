# Disease-Risk-Prediction

A combination of projects. Basically heart, lung, kidney, brain and other organ's wellbeing.

Two layers so far. A **symptom triage router** hears what a patient tells and decides
which organs to check. Each of three organs then has a **tabular** model (routine
clinical data → risk of disease) and an **image** model (a scan or printout → diagnosis):

```
symptoms ──► triage/tabular ──► Heart  ──► models/heart/tabular  ──► models/heart/image_based  (ECG)
                            ──► Lung   ──► models/lung/tabular   ──► models/lung/image_based   (chest X-ray)
                            ──► Kidney ──► models/kidney/tabular ──► models/kidney/image_based (CT)
                            ──► Other (no heart, lung or kidney signal)
```

| Layer 1 | |
|---|---|
| Triage | [`triage/tabular`](triage/tabular) — heart / lung / kidney / other from age, sex and up to three symptoms (vital signs optional); 184,629 CDC NHAMCS emergency-department visits, one calibrated head per organ |

| Organ | Tabular | Image |
|---|---|---|
| Heart | [`models/heart/tabular`](models/heart/tabular) — heart disease from NHANES vitals, labs and history | [`models/heart/image_based`](models/heart/image_based) — 12-lead ECG printout: Abnormal Heartbeat, History of MI, MI, Normal |
| Kidney | [`models/kidney/tabular`](models/kidney/tabular) — CKD (KDIGO), without the tests that define it | [`models/kidney/image_based`](models/kidney/image_based) — CT slice: Cyst, Normal, Stone, Tumor |
| Lung | [`models/lung/tabular`](models/lung/tabular) — COPD from smoking, symptoms and blood counts | [`models/lung/image_based`](models/lung/image_based) — chest X-ray: Covid-19, Emphysema, Normal, Pneumonia (bacterial / viral), Tuberculosis |

Each module has its own README (data, method, results) and the same layout:
`data/`, `notebooks/` (the full analysis), `src/` (importable code), `models/`, `results/`, `images/`,
and for the tabular modules `examples/` (a sample patient).

## Repository layout

```
common/        shared code: registry.py (one entry point), cascade.py, prediction.py,
               tabular.py / image.py (the per-module cascades), and
               config/ (where the models live), preprocessing/, utils/, evaluation/
frontend/      app/ (the Streamlit page), components/ (form fields), assets/ (example patient)
models/        the six organ models: heart/, kidney/, lung/, each with tabular/ and image_based/
triage/        tabular/: the layer-1 symptom router
tests/         pytest smoke tests: common/, heart/, kidney/, lung/, triage/
```

## One interface for all seven models

Every module's `src/predict.py` (the router and the six organ models) exposes the same
two functions, built on the shared code in [`common/`](common):

- `predict(x)` → a `Prediction`, for one patient (a dict of the module's raw input
  columns) or one image (a path, bytes, a file-like upload, a PIL image or an array)
- `predict_batch(xs)` → a DataFrame with one row per patient or image

```python
from common.registry import predict, route   # from the repo root

triage = route({"age": 64, "sex": "Male", "symptoms": ["Chest pain", "Shortness of breath"]})
triage.label, triage.details["route"]                   # e.g. 'Heart', ['heart']: organs to run next

result = predict("heart", "tabular", patient)          # patient: dict of raw inputs
result = predict("lung", "image", "xray.png")
result.status, result.label, result.probability, result.positive, result.explanation
result.to_dict(arrays=False)                            # JSON-ready (drops the heatmap arrays)
```

For the router, `label` is the route (`Heart`, `Heart + Lung`, `Other`), `probability`
the highest organ probability, `positive` whether any organ module should run, and
`details` has all three probabilities, their thresholds and the route in order. Its
anomaly gate returns `review` for impossible vital signs.

| Field | Tabular modules | Image modules |
|---|---|---|
| `status` | `accepted`, or `review` when the anomaly gate withholds an unusual profile | `accepted`; `review` below 75 % confidence; `rejected` when the out-of-distribution gate says the image is not the expected kind (e.g. an ECG given to the X-ray model) |
| `label` | risk band: Low / Medium / High Risk | predicted class |
| `probability` | calibrated P(disease) | 1 − P(Normal) |
| `positive` | probability ≥ the cost-optimal threshold: refer | predicted class is not Normal |
| `explanation` | the 5 inputs that moved the probability most, compared with a typical training patient | Grad-CAM heatmap and overlay |
| `message` | risk band and next step, e.g. "refer for spirometry" | finding and confidence, or why it was rejected |
| `details` | threshold, risk bands, model | confidence, class probabilities, top-3, model |

When `status` is `rejected`, or `review` from the tabular anomaly gate, the model gives
no `label` and no `probability`.

Every module also still runs on its own, from its folder:

```bash
python -m src.predict --input examples/patient.json      # tabular modules and the triage router
python -m src.predict --image scan.png --heatmap-dir out  # image modules
```

## The cascade: all seven models for one patient

[`common/cascade.py`](common/cascade.py) chains them. Triage routes the patient, then each
routed organ runs its tabular model and, when an image is given, its image model:

```python
from common.registry import cascade

result = cascade({"age": 64, "sex": "Male", "symptoms": ["Chest pain", "Shortness of breath"]},
                 tabular={"heart": heart_answers},       # any subset of models/heart/tabular's inputs
                 images={"heart": "ecg.png"},
                 also_check=["kidney"])                   # check an organ triage did not route to
result.route, result.status, result.positive, result.next_steps
result.organs["heart"].tabular, result.organs["heart"].image   # a Prediction each, or None
```

```bash
python -m common.cascade --input frontend/assets/cascade_patient.json                 # from the repo root
python -m common.cascade --input frontend/assets/cascade_patient.json --image heart=ecg.png --heatmap-dir out
```

How it decides:

- **Triage gate.** If triage's anomaly gate withholds the vital signs, the cascade stops there with `review`. If triage routes nowhere ("other"), no organ runs unless it is in `also_check`.
- **Shared answers.** Age, sex, blood pressure and diabetes given to triage are copied into each organ's questionnaire, unless it has its own value. They are listed in `filled_from_triage`, so nothing is asked twice.
- **Tabular step.** It runs when at least half of the organ model's inputs are known. Otherwise it is skipped and `missing_inputs` lists what to ask, so a patient is never scored mostly on imputed values.
- **Image step.** It runs whenever an image is given for a checked organ, also after a low tabular risk. The tabular models screen one chronic disease each (heart disease, COPD, CKD); the image models find others (pneumonia, tuberculosis, stones, MI on an ECG). Without an image, `next_steps` asks for one when the tabular screen is positive or could not run.
- **Status.** `review` when any step withheld, rejected or doubted its input; `positive` when any organ step screened positive. The steps' probabilities are reported side by side, never combined (see Caveats).
- **Inputs for unchecked organs.** Inputs for an organ that was not checked are not scored. `unchecked` says so; add the organ to `also_check` to run it. This matters for the kidney: chronic kidney disease rarely comes with symptoms the router can hear.

## Front end

[`frontend/app/streamlit_app.py`](frontend/app/streamlit_app.py) is one page for the whole cascade:

```bash
streamlit run frontend/app/streamlit_app.py      # from the repo root
```

1. **Symptoms.** Age, sex, up to three of the 128 symptoms, optional vital signs → the route, each organ's probability against its threshold, and why.
2. **Organ checks.** One tab per routed organ (add others under *Also check*). Each has that model's questionnaire, grouped into about you, measurements, lab results, and history. Answers given in step 1 are not asked again. Each tab also takes an optional image upload.
3. **Results.** Per organ, the risk band and what moved it, next to the image finding, its class probabilities and the Grad-CAM overlay. Then the next steps and a JSON download.

*Load example patient* fills every form from [`frontend/assets/cascade_patient.json`](frontend/assets/cascade_patient.json).
The text answers offered are read from each fitted pipeline, so they always match what the model was trained on.
The first image of a session loads its network (about 1.5 GB of memory with PyTorch).

## Setup

- Python 3.13; each module lists its packages in its own `requirements.txt`.
- The image modules run Keras 3 on the **PyTorch** backend when torch is installed,
  because TensorFlow has no GPU support on native Windows. Set `KERAS_BACKEND=tensorflow`
  to use TensorFlow instead; Grad-CAM works on both. The chest X-ray checkpoints were
  trained with TensorFlow and give the same predictions on either backend (2 of 1,737
  test images differ).
- Not in git, because of their size: the `.keras` checkpoints and the image datasets.
  Each image module's README says where to get them.

## Tests

```bash
python -m pytest tests                         # from the repo root: layout, imports, module CLIs, tabular models, triage, cascade
RUN_IMAGE_TESTS=1 python -m pytest tests       # also loads each image network (about 1.5 GB each)
```

The image tests need the `.keras` checkpoints and each module's `data/test/` images; they skip when those are missing.

## Caveats

- The tabular and image datasets come from different patients. There is no dataset
  where the same person has both, so each model is evaluated on its own. A combined
  "tabular screen → imaging" result can be reported only under an assumption that
  the two stages' errors are independent. The same holds for the router: it learns
  from emergency-department visits (NHAMCS), the organ tabular models from NHANES
  survey participants.
- These are research models trained on public datasets, not medical devices.
