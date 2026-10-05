"""Build the clean ECG image dataset used by notebooks/Heart_ECG.ipynb.

Source: "ECG Images dataset of Cardiac Patients" (Khan & Hussain, Mendeley Data,
v2, DOI 10.17632/gwbz3fsgp8.2, CC BY 4.0) - 928 12-lead ECG printouts in four
folders. The zip is the one downloaded from Mendeley (ECG.zip).

What this script does:
1. Drops exact duplicate files. The published dataset repeats many images
   (Myocardial Infarction: 239 files but only 30 distinct ECGs), and a random
   split would put copies of the same ECG in both train and test.
2. Crops every printout to the ECG grid. The header holds the patient ID,
   recording date/time and heart rate, and the footer holds a timestamp - text
   a CNN could use as a shortcut instead of the waveform.
3. Halves the resolution (2109x1235 -> 1054x617) to keep the folder small; the
   models use 256x448 input, so no detail they could see is lost.
4. Writes a stratified 70/15/15 train/val/test split plus manifest.csv.

Usage (from models/heart/image_based/):
    python data/build_ecg_dataset.py --source ~/Downloads/ECG.zip
"""
import argparse
import hashlib
import io
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Source folder name -> class name used everywhere else
CLASS_MAP = {
    'abnormal_heartbeat_ecg_images': 'Abnormal_Heartbeat',
    'myocardial_infarction_ecg_images': 'Myocardial_Infarction',
    'normal_ecg_images': 'Normal',
    'post_mi_history_ecg_images': 'History_of_MI',
}
# Folder names as they appear in the original Mendeley download
CLASS_MAP.update({
    'ECG Images of Patient that have abnormal heartbeat (233x12=2796)': 'Abnormal_Heartbeat',
    'ECG Images of Myocardial Infarction Patients (240x12=2880)': 'Myocardial_Infarction',
    'Normal Person ECG Images (284x12=3408)': 'Normal',
    'ECG Images of Patient that have History of MI (172x12=2064)': 'History_of_MI',
})

PRINTOUT_SIZE = (2213, 1572)
GRID_BOX = (68, 283, 2177, 1518)   # (left, top, right, bottom) of the ECG grid
SEED = 42


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def iter_source(source):
    """Yield (source_folder, filename, bytes) from a zip or an extracted folder."""
    source = Path(source).expanduser()
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for n in sorted(names, key=natural_key):
                parts = n.split('/')
                yield parts[-2], parts[-1], zf.read(n)
    else:
        for path in sorted(source.rglob('*'), key=lambda p: natural_key(str(p))):
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                yield path.parent.name, path.name, path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='~/Downloads/ECG.zip', help='ECG.zip or its extracted folder')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent), help='output data folder')
    args = parser.parse_args()
    out = Path(args.out)

    # 1. Deduplicate by file content (first file in natural order is kept)
    records, seen = [], {}
    for folder, fname, data in iter_source(args.source):
        if folder not in CLASS_MAP:
            raise ValueError(f'Unknown class folder: {folder}')
        label = CLASS_MAP[folder]
        digest = hashlib.md5(data).hexdigest()
        if digest in seen:
            seen[digest]['duplicates'].append(fname)
            if seen[digest]['label'] != label:
                raise ValueError(f'{fname} is a duplicate with a different label')
            continue
        rec = {'label': label, 'source_file': fname, 'md5': digest, 'duplicates': [], 'data': data}
        seen[digest] = rec
        records.append(rec)

    df = pd.DataFrame(records)
    total = len(df) + df['duplicates'].str.len().sum()
    print(f'{total} files -> {len(df)} unique ECGs')
    print(df.groupby('label').agg(unique=('md5', 'size'), copies=('duplicates', lambda s: s.str.len().sum())))

    # 2. Stratified 70 / 15 / 15 split
    train_df, rest = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest['label'], random_state=SEED)
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[val_df.index, 'split'] = 'val'
    df.loc[test_df.index, 'split'] = 'test'

    # 3. Crop to the grid, downscale, write
    for split in ('train', 'val', 'test'):
        shutil.rmtree(out / split, ignore_errors=True)
    for i, row in df.iterrows():
        img = Image.open(io.BytesIO(row['data'])).convert('RGB')
        if img.size != PRINTOUT_SIZE:
            raise ValueError(f"{row['source_file']}: unexpected size {img.size}")
        img = img.crop(GRID_BOX)
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        stem = Path(row['source_file']).stem.replace(' ', '')
        dest = out / row['split'] / row['label'] / f'{stem}.png'
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, optimize=True)
        df.at[i, 'file'] = dest.relative_to(out).as_posix()

    manifest = df.drop(columns='data').assign(duplicates=df['duplicates'].str.join(';'))
    manifest = manifest[['file', 'label', 'split', 'source_file', 'md5', 'duplicates']]
    manifest.to_csv(out / 'manifest.csv', index=False)
    print(pd.crosstab(df['label'], df['split'], margins=True)[['train', 'val', 'test', 'All']])
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
