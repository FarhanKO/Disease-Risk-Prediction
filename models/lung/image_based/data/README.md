# Chest X-ray Dataset — Data Notes

The images behind `notebooks/Lung_Xray.ipynb` and `src/`: **18,036 chest X-rays,
224 × 224, in 6 classes (Covid-19, Emphysema, Normal, Pneumonia-Bacterial,
Pneumonia-Viral, Tuberculosis)**, already split into train / val / test folders.
It is a Kaggle compilation of two public datasets, which are themselves built from
at least six older collections.

## Source

Covid-19, Emphysema, Normal, Pneumonia-Bacterial, Pneumonia-Viral | Kaggle [`minhnhat232/dataset-covid-bacterial-viral-normal-emphysema`](https://www.kaggle.com/datasets/minhnhat232/dataset-covid-bacterial-viral-normal-emphysema), *Dataset (Covid-Bacterial-Viral-Normal-Emphysema)*, Nguyễn Minh Nhật, 2024, CC0 |
Kaggle [`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset), *Tuberculosis (TB) Chest X-ray Database*, Rahman et al., Qatar University / University of Dhaka. Licence: "Data files © Original Authors". Credit is required. |

The combined six-class, 224 × 224, pre-split package does not appear to have its
own live Kaggle page. Three other Kaggle records confirm its contents:

- [`kashif03371733/chestx6-ssl-benchmark-splits-and-checksums`](https://www.kaggle.com/datasets/kashif03371733/chestx6-ssl-benchmark-splits-and-checksums)
  (*ChestX6*, 2026) describes **the same set**: 18,036 images, the same six class
  names, built from exactly the two datasets above. Tuberculosis makes up 17.6 %
  of it. Its MD5 audit found **48 exact duplicates** (17,988 unique). It publishes
  its own 70/10/20 split indices, not the split used here.
- [`warmouse/chest-x-ray-5-classification`](https://www.kaggle.com/datasets/warmouse/chest-x-ray-5-classification)
  is this set without Pneumonia-Viral: 15,023 = 18,036 − 3,013 images, 224 × 224,
  in train/val/test folders.
- [`roazarior/chest-x-ray-4-classification`](https://www.kaggle.com/datasets/roazarior/chest-x-ray-4-classification)
  is this set without either pneumonia class: 12,023 images.

Both subsets reuse one description, which still says each split has "six
subfolders". They were cut from the same six-class package.

### Where the images originally come from

According to the two source datasets' own descriptions:

| Class | Via | Original collections | File names here |
|---|---|---|---|
| Covid-19 | minhnhat232 | COVID-QU-Ex / COVID-19 Radiography Database (Tahir, Chowdhury, Rahman et al.) and Sait et al. (Mendeley) | `COVID-<n>.jpg` |
| Pneumonia-Viral | minhnhat232 | COVID-QU-Ex / COVID-19 Radiography Database and Sait et al. | `Viral Pneumonia-<n>.jpg` |
| Pneumonia-Bacterial | minhnhat232 | Sait et al., *Curated dataset for COVID-19 posterior-anterior chest radiography images*, Mendeley Data v4 | `Pneumonia-Bacterial (<n>).jpg` |
| Normal | minhnhat232 | Sait et al. | `Normal (<n>).jpg` |
| Emphysema | minhnhat232 | NIH ChestX-ray14 (Wang et al., 2017): 2,550 images with the Emphysema label | `Emphysema_<n>.jpg` |
| Tuberculosis | TB Chest X-ray Database | NLM Montgomery and Shenzhen sets, Belarus TB portal, NIAID TB portal, RSNA | `Tuberculosis-<n>.png` (original) and `augmented_Tuberculosis-<n>_0_<k>.jpeg` (augmented copy) |

The five minhnhat232 classes match that dataset's published counts. The only
difference is Normal, which has 3,271 images here against 3,270 there.
**Tuberculosis does not match its source.** The TB database publishes 700 TB
images, but this set has 3,185. The rest are augmented copies of those 700 (see
*Known limitations*).

### Citation

Cite the two direct sources and the collections underneath them:

- T. Rahman, A. Khandakar, M. A. Kadir, K. R. Islam, K. F. Islam, Z. B. Mahbub,
  M. A. Ayari, M. E. H. Chowdhury. "Reliable Tuberculosis Detection using Chest
  X-ray with Deep Learning, Segmentation and Visualization." *IEEE Access* 8,
  191586–191601 (2020). doi:10.1109/ACCESS.2020.3031384
- U. Sait, K. G. Lal, S. P. Prajapati, R. Bhaumik, T. Kumar, S. Shivakumar,
  K. Bhalla. *Curated dataset for COVID-19 posterior-anterior chest radiography
  images (X-rays).* Mendeley Data v4 (2022). doi:10.17632/9xkhgts2s6.4
- X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, R. M. Summers. "ChestX-ray8:
  Hospital-scale chest X-ray database and benchmarks on weakly-supervised
  classification and localization of common thorax diseases." *CVPR* 2017,
  3462–3471. doi:10.1109/CVPR.2017.369
- A. M. Tahir, M. E. H. Chowdhury, A. Khandakar, T. Rahman, Y. Qiblawey, et al.
  "COVID-19 infection localization and severity grading from chest X-ray images."
  *Computers in Biology and Medicine* 139, 105002 (2021).
  doi:10.1016/j.compbiomed.2021.105002
- M. E. H. Chowdhury, T. Rahman, A. Khandakar, et al. "Can AI help in screening
  viral and COVID-19 pneumonia?" *IEEE Access* 8, 132665–132676 (2020).
  doi:10.1109/ACCESS.2020.3010287

Each underlying collection has its own terms. The images are **not** committed to
this repository. Check the terms before redistributing them.

## Layout

One folder per class in each split. The class folder names are the labels, and
`src/data.py` (`CLASS_NAMES`) expects exactly these, in alphabetical order:

```
data/
├── train/
│   ├── Covid-19/
│   ├── Emphysema/
│   ├── Normal/
│   ├── Pneumonia-Bacterial/
│   ├── Pneumonia-Viral/
│   └── Tuberculosis/
├── val/    (same 6 folders)
└── test/   (same 6 folders)
```

To use it, download the `train/`, `val/` and `test/` folders from the Drive
folder into this directory, or pass `--data-dir <path>` to `src.train` /
`src.evaluate`. Only training and evaluation need the images. `src.predict` does
not.

## Counts

From the notebook's item-count cell (`data.count_images()` gives the same table):

| Class | Train | Val | Test | Total | Source count |
|---|---|---|---|---|---|
| Covid-19 | 2,417 | 300 | 300 | 3,017 | 3,017 |
| Emphysema | 2,050 | 250 | 250 | 2,550 | 2,550 |
| Normal | 2,671 | 300 | 300 | 3,271 | 3,270 |
| Pneumonia-Bacterial | 2,400 | 300 | 300 | 3,000 | 3,000 |
| Pneumonia-Viral | 2,413 | 300 | 300 | 3,013 | 3,013 |
| Tuberculosis | 2,600 | 298 | 287 | 3,185 | 700 originals |
| **Total** | **14,551** | **1,748** | **1,737** | **18,036** | |

The split is about 81 / 10 / 10 and was made by whoever built the compilation,
not in this repository. The classes look balanced: the largest (Normal) is 1.3×
the smallest (Emphysema). That balance is partly artificial, because
Tuberculosis was brought up to size by augmentation. Training still uses balanced
class weights (`data.class_weights()`).

## Image properties

- **All 224 × 224.** Every image the notebook sampled (200 per class) was 224 × 224,
  so the compilation was resized before it reached us. The original resolutions and
  aspect ratios are lost.
- **Mixed formats.** The five minhnhat232 classes and the augmented TB copies are
  JPEG, at about 8–14 KB each. The original TB images are PNG, at 11–32 KB.
- **Preprocessing at load time:** `RGB → 224 × 224 → ÷255`, done in
  `data.load_image()` for inference and by `image_dataset_from_directory` for
  training. Our own augmentation (rotation, zoom, flip, brightness) is applied to
  the training split only. It comes on top of the augmentation already baked into
  the TB class.

## Known limitations

- **Tuberculosis is mostly augmented copies, and they reach val and test.** About
  2,485 of the 3,185 TB images are `augmented_Tuberculosis-<n>_0_<k>.jpeg`
  variants of the 700 originals. The copies of one original seem to stay in one
  split. For example, originals 2, 3, 653 and 656 and 20+ copies of each are all in
  `val/`, and original 685 and its copies are in `test/`. So the 287 TB test images
  come from far fewer distinct patients. The **0.99 TB test F1 is measured largely
  on near-copies** and overstates how well the model would do on new TB films.
  After downloading, check it by grouping `Tuberculosis/` files by `<n>`, then
  evaluate on one image per original.
- **Duplicates.** ChestX6's MD5 audit of the same 18,036 files found 48 exact
  duplicates. This repository has not deduplicated them, unlike the ECG and CT
  datasets. Some duplicates could sit in different splits.
- **No patient IDs.** The split was made upstream. Apart from the TB file names, we
  cannot check whether one patient's films fall in more than one split. Several of
  the source collections include more than one film per patient.
- **Source shortcuts.** Each class comes from different hospitals, countries,
  scanners and age groups, and the TB originals are the only PNG files. A model can
  learn to tell the sources apart instead of the disease. The high Tuberculosis and
  Normal scores fit that pattern, though they do not prove it.
- **Hard pairs are hard for a reason.** Most errors are Pneumonia-Bacterial ↔
  Pneumonia-Viral and Covid-19 ↔ Emphysema. Bacterial and viral pneumonia are hard
  to tell apart on a radiograph even for radiologists.
- **Weak labels for Emphysema.** The NIH ChestX-ray14 labels were mined from
  radiology reports with NLP, not read from the image, and some are wrong.
- **Single label per image.** Comorbid findings, such as emphysema with pneumonia,
  get only one label.
- **Resized upstream.** At 224 × 224, fine detail such as small nodules or early
  interstitial change is gone. The model has not been tested on full-resolution
  or DICOM input.
