# ECG Images of Cardiac Patients — Dataset Card

**491 unique 12-lead ECG printouts in 4 classes**, taken from the 928 files in the
published dataset. The rest were exact duplicates and were removed. Each image
is cropped to the ECG grid and saved as a 1054 × 617 PNG in a stratified
70 / 15 / 15 train / val / test split. Used by `notebook/Heart_ECG.ipynb` and
`src/`.

| Class | Raw files | Duplicates dropped | Unique ECGs | Train | Val | Test |
|---|---|---|---|---|---|---|
| `Abnormal_Heartbeat` | 233 | 0 | 233 | 163 | 35 | 35 |
| `History_of_MI` | 172 | 86 | 86 | 60 | 13 | 13 |
| `Myocardial_Infarction` | 239 | 209 | 30 | 21 | 5 | 4 |
| `Normal` | 284 | 142 | 142 | 99 | 21 | 22 |
| **Total** | **928** | **437** | **491** | **343** | **74** | **74** |

## Source

*ECG Images dataset of Cardiac Patients*. Ali Haider Khan and Muzammil Hussain,
Mendeley Data, version 2, 19 March 2021.
[doi:10.17632/gwbz3fsgp8.2](https://data.mendeley.com/datasets/gwbz3fsgp8/2).
Collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan.
Licensed **CC BY 4.0**: you may share and adapt the data if you credit the authors.

```bibtex
@misc{khan2021ecg,
  author    = {Khan, Ali Haider and Hussain, Muzammil},
  title     = {{ECG Images dataset of Cardiac Patients}},
  year      = {2021},
  publisher = {Mendeley Data},
  version   = {2},
  doi       = {10.17632/gwbz3fsgp8.2}
}
```

## Rebuild

The images are not in git (about 465 MB processed, 587 MB raw). Download the
zip from the Mendeley page, then run from `heart/image_based/`:

```bash
python data/build_ecg_dataset.py --source path/to/ECG.zip
```

The script also accepts the extracted folder. It recognises the original
Mendeley folder names and the short names used here, which are
`abnormal_heartbeat_ecg_images/`, `myocardial_infarction_ecg_images/`,
`normal_ecg_images/` and `post_mi_history_ecg_images/`. It uses a fixed
`random_state = 42`, so every rebuild gives the same split and matches
`manifest.csv`.

## What an image contains

Every raw file is a 2213 × 1572 JPEG scan of an ECG report page printed from a
single machine template:

- **Header:** patient ID, sex, blank demographic fields and diagnosis fields.
- **Grid:** 12 leads in a 3 × 4 layout (I, II, III · aVR, aVL, aVF · V1–V3 ·
  V4–V6), 2.5 s per lead. Below them is one 10 s rhythm strip.
- **Footer:** recording settings (0.67–25 Hz filter, 50 Hz notch, 25 mm/s,
  10 mm/mV), heart rate, device string and the recording date and time.

Classes (as labelled by the dataset authors; no finer diagnosis is given):

| Class | Source folder | File prefix | Meaning |
|---|---|---|---|
| `Normal` | Normal Person ECG Images | `Normal(n)` | No cardiac abnormality |
| `Abnormal_Heartbeat` | ECG Images of Patient that have abnormal heartbeat | `HB(n)` | Arrhythmia (type not specified) |
| `Myocardial_Infarction` | ECG Images of Myocardial Infarction Patients | `MI(n)` | Patient with myocardial infarction |
| `History_of_MI` | ECG Images of Patient that have History of MI | `PMI(n)` | Patient who had an MI in the past |

## Processing (`build_ecg_dataset.py`)

1. **Exact-duplicate removal.** Files are hashed with MD5 and only the first
   copy is kept, taking files in natural order (`MI(1)` before `MI(10)`). The
   published data repeats files heavily. Each MI ECG appears about 8 times
   (239 files, 30 distinct ECGs), and every Normal and History-of-MI ECG appears
   twice. If the duplicates were split at random, copies of the same ECG would
   land in both train and test and inflate the scores. No file appears under
   two different labels, and the script stops with an error if one does.
2. **Crop to the grid** with `GRID_BOX = (68, 283, 2177, 1518)`, which gives
   2109 × 1235 px. This removes the patient ID, date, time and heart-rate text,
   which a CNN could use as a shortcut instead of reading the waveform.
3. **Downscale by 2** with Lanczos resampling to 1054 × 617 and save as PNG.
   The models take 448 × 256 input, so no usable detail is lost.
4. **Stratified split:** 70 % train, then the remaining 30 % halved into val
   and test.

Everything else (resize to 448 × 256, augmentation, per-backbone scaling)
happens in the notebook and inside the saved models, not here.

## Layout

```
data/
├── build_ecg_dataset.py
├── manifest.csv                    # in git
├── README.md                       # in git
├── train/ val/ test/
│   └── Abnormal_Heartbeat/ History_of_MI/ Myocardial_Infarction/ Normal/   # *.png
└── <raw source folders>/           # optional local copy of the download, not in git
```

## `manifest.csv`

One row per kept image (491 rows).

| Column | Meaning |
|---|---|
| `file` | Path of the processed PNG relative to `data/`, e.g. `train/Normal/Normal(1).png` |
| `label` | One of the 4 class names |
| `split` | `train`, `val` or `test` |
| `source_file` | Original file name in the Mendeley download |
| `md5` | MD5 of the original JPEG bytes |
| `duplicates` | `;`-separated names of the identical files that were dropped |

## Caveats

- **Small and imbalanced.** There are only 30 distinct MI ECGs, with 4 in
  test, so per-class MI metrics are indicative only. Class weights are used in
  training.
- **Split is per ECG, not per patient.** The dataset does not publish patient
  IDs apart from the header text that is cropped away. If one patient has
  several different ECGs, they may be in different splits.
- **Only exact duplicates are removed.** Near-duplicates, such as the same ECG
  rescanned or re-saved, would not be caught by an MD5 check.
- **One hospital, one machine, one print template.** Other ECG machines, lead
  layouts, paper speeds or phone photos of printouts are untested. The
  cascade's OOD gate may reject them.
- **Coarse labels.** The dataset does not give the type of arrhythmia, the
  location of the infarct, or how long ago a past MI occurred.
