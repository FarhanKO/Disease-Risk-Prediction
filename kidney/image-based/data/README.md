# Kidney CT Dataset — Data Notes

The images behind `notebook/Kidney_CT.ipynb`: **11,929 unique kidney CT slices
from 189 CT scans, in 4 classes (Cyst, Normal, Stone, Tumor)**, split by scan
into train / val / test so that no patient scan appears in more than one split.

## Source

| | |
|---|---|
| Dataset | *CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone* |
| Paper | Islam, M. N., Hasan, M., et al. "Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography." *Scientific Reports* 12, 11440 (2022) |
| Original release | Kaggle — `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`(https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| Copy used here | Hugging Face mirror `ryfkn/CT-Kidney-Dataset-{Cyst,Normal,Stone,Tumor}`: one parquet per class (Normal in two parts), full-resolution originals with their original file names (`Cyst- (1)`, …), ~1.6 GB |
| Origin | Hospital PACS in Dhaka, Bangladesh; axial and coronal abdominal CT, exported as windowed JPEGs (not raw Hounsfield-unit DICOM) |
| Labels | One diagnosis per image, assigned by the dataset authors (radiologist-confirmed reports) |

Check the licence and citation terms on the Kaggle page before redistributing
the images. They are **not** committed to this repository.

## Files in this folder

| Path | In git | What it is |
|---|---|---|
| `build_kidney_dataset.py` | yes | Rebuilds everything below from `raw/` (deterministic, seed 42) |
| `manifest.csv` | yes | One row per kept slice: output file, label, split, scan series, original ID, original size, MD5, and the IDs of any duplicates dropped in its favour |
| `raw/` | no | The 5 source parquet files, named `<Class>__<file>.parquet` (~1.6 GB) |
| `train/`, `val/`, `test/` `<Class>/` | no | Built PNGs, `<Class>-<original number>.png` (~270 MB) |

To rebuild, put the parquet files in `raw/` and run, from `kidney/image_based/`:

```bash
python data/build_kidney_dataset.py
```

## Counts

| Class | Raw files | Duplicates removed | Unique slices | Scan series | Train (series) | Val (series) | Test (series) |
|---|---|---|---|---|---|---|---|
| Cyst | 3,709 | 425 | 3,284 | 64 | 2,298 (38) | 493 (14) | 493 (12) |
| Normal | 5,077 | 75 | 5,002 | 47 | 3,503 (32) | 750 (7) | 749 (8) |
| Stone | 1,377 | 17 | 1,360 | 55 | 952 (38) | 204 (5) | 204 (12) |
| Tumor | 2,283 | 0 | 2,283 | 23 | 1,600 (15) | 342 (4) | 341 (4) |
| **Total** | **12,446** | **517** | **11,929** | **189** | **8,353 (123)** | **1,789 (30)** | **1,787 (36)** |

The split is about 70 / 15 / 15 by images. Scans hold 1–262 slices each
(median 54). The classes are imbalanced (Normal is 3.7× Stone), which the
notebook handles with class weights.

## How the dataset was built (and why)

The public dataset is usually split image by image. That is misleading: the
12,446 files are consecutive slices of only ~189 scans, and neighbouring slices
are near-identical (median pixel correlation 0.998, against 0.3–0.5 for random
pairs). A memorising 1-nearest-neighbour classifier scores **100 %** on a
random image split and **45 %** on the split used here. `build_kidney_dataset.py`
therefore does four things:

1. **Removes exact duplicates.** Files with identical decoded pixels are
   dropped: 517 files, mostly Cyst. The lowest-numbered copy is kept, and the
   dropped IDs are recorded in `manifest.csv`.
2. **Recovers the scan series.** Files are sorted by their original number.
   Consecutive slices whose 48×48 thumbnails correlate above 0.7 are joined
   into one series, and any two slices of a class correlating above 0.95 are
   merged too (union-find).
3. **Splits by series, per class.** Whole series go to test, val or train. Of
   3,000 random series orders, the one closest to 15 % / 15 % / 70 % of images
   is kept, with at least 3 series per class in every split. `manifest.csv`
   confirms that no series appears in more than one split.
4. **Standardises the images.** Each slice is converted to grayscale, padded to
   a square with black (the CT background), and resized to 256×256 (Lanczos) as
   PNG. The notebook then resizes to 224×224 at load time.

## Image properties

- Originals come in **77 different sizes**, from 512×451 to 1371×1110. 61 %
  are 512×512; the rest are screen captures of other viewer windows. The
  original size of every slice is in `manifest.csv`.
- All images are 8-bit windowed grayscale, so absolute density (HU) is lost,
  and window and contrast phase vary between scans.

## Known limitations

- **Few independent scans.** 189 scans, and only **4 tumor scans** in each of
  val and test. Test metrics move a lot with a single scan, which is why the
  notebook reports scan-level bootstrap confidence intervals.
- **Acquisition shortcuts.** Average body-pixel brightness differs by class
  (Cyst darker, Normal and Tumor brighter). That reflects contrast phase and
  display window, not disease, and a model can partly learn it.
- **Single label per image.** Some slices show more than one finding (e.g. a
  stone next to a mass) but carry one label.
- **One region, one export format.** All scans come from hospitals in Dhaka,
  exported as JPEGs. Other scanners, protocols and DICOM inputs are untested.
- **Series recovery is approximate.** It is inferred from file order and image
  similarity, because the dataset has no patient or scan IDs. Two scans of the
  same patient could still land in different splits.
