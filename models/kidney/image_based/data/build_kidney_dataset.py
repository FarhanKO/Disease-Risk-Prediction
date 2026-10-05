"""Build the leakage-free kidney CT dataset used by notebooks/Kidney_CT.ipynb.

Source: "CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone" (Islam, Hasan et al.,
Scientific Reports 2022; Kaggle nazmul0087), 12,446 CT slices collected from
hospital PACS in Dhaka, Bangladesh. Downloaded as the Hugging Face mirror
ryfkn/CT-Kidney-Dataset-{Cyst,Normal,Stone,Tumor} (one parquet file per class,
original file names kept) into data/raw/.

What this script does:
1. Drops exact duplicates (same decoded pixels): 517 files, 425 of them Cyst.
2. Groups slices into scan series. Consecutively numbered files are adjacent
   slices of one scan (median pixel correlation 0.998, random pairs 0.3-0.5),
   so a random split would put near-identical slices in train and test. A
   series is a run of consecutive IDs whose neighbours correlate > 0.7, merged
   with any other slice of the same class that correlates > 0.95.
3. Splits whole series ~70/15/15 into train/val/test, separately per class,
   choosing among 3,000 random series orders the one closest to the target
   image proportions (at least 3 series per class in every split).
4. Pads each slice to a square (black, the CT background) and resizes it to
   256x256 grayscale PNG.

Usage (from models/kidney/image_based/):
    python data/build_kidney_dataset.py            # reads data/raw/*.parquet
"""
import argparse
import hashlib
import io
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
OUT_SIZE = 256
THUMB = 48                  # thumbnail side used for similarity
CONSECUTIVE_CORR = 0.7      # neighbouring IDs above this belong to the same series
ANY_CORR = 0.95             # any two slices above this belong to the same series
TARGET = {'train': 0.70, 'val': 0.15, 'test': 0.15}
MIN_SERIES = 3              # at least 3 series per class in every split
SEED = 42


def pad_to_square(img):
    w, h = img.size
    side = max(w, h)
    canvas = Image.new('L', (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def series_groups(thumbs):
    """Union-find over (a) consecutive IDs that look alike and (b) any near-identical pair."""
    n = len(thumbs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    t = (thumbs - thumbs.mean(1, keepdims=True)) / (thumbs.std(1, keepdims=True) + 1e-6)
    corr = t @ t.T / t.shape[1]
    for i in range(n - 1):
        if corr[i, i + 1] > CONSECUTIVE_CORR:
            parent[find(i)] = find(i + 1)
    for a, b in zip(*np.where(np.triu(corr, 1) > ANY_CORR)):
        parent[find(a)] = find(b)
    roots = [find(i) for i in range(n)]
    # renumber series 0..k-1 in order of first appearance
    mapping = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([mapping[r] for r in roots])


def split_series(sizes, rng):
    """sizes: images per series. Returns split name per series, closest to TARGET."""
    n, total = len(sizes), sizes.sum()
    best, best_score = None, np.inf
    for _ in range(3000):
        order = rng.permutation(n)
        cum = np.cumsum(sizes[order]) / total
        # first series go to test, next to val, the rest to train
        n_test = int(np.argmin(np.abs(cum - TARGET['test']))) + 1
        n_val = int(np.argmin(np.abs(cum - TARGET['test'] - TARGET['val']))) + 1 - n_test
        if n_test < MIN_SERIES or n_val < MIN_SERIES or n - n_test - n_val < MIN_SERIES:
            continue
        test_frac = cum[n_test - 1]
        val_frac = cum[n_test + n_val - 1] - test_frac
        score = abs(test_frac - TARGET['test']) + abs(val_frac - TARGET['val'])
        if score < best_score:
            labels = np.empty(n, dtype=object)
            labels[order[:n_test]] = 'test'
            labels[order[n_test:n_test + n_val]] = 'val'
            labels[order[n_test + n_val:]] = 'train'
            best, best_score = labels, score
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument('--raw', default=str(here / 'raw'), help='folder with the four class parquet files')
    parser.add_argument('--out', default=str(here), help='output data folder')
    args = parser.parse_args()
    out, rng = Path(args.out), np.random.default_rng(SEED)

    for split in TARGET:
        shutil.rmtree(out / split, ignore_errors=True)

    records = []
    for cls in CLASSES:
        rows = []
        for f in sorted(Path(args.raw).glob(f'{cls}__*.parquet')):
            rows += pq.read_table(f).to_pylist()
        rows.sort(key=lambda r: int(re.search(r'\((\d+)\)', r['image_id']).group(1)))

        # 1. Exact duplicates (identical pixels); the lowest-numbered copy is kept
        kept, first_of = [], {}
        for r in rows:
            img = Image.open(io.BytesIO(r['path']['bytes'])).convert('L')
            digest = hashlib.md5(img.tobytes() + str(img.size).encode()).hexdigest()
            if digest in first_of:
                first_of[digest]['duplicates'].append(r['image_id'])
                continue
            rec = {'label': cls, 'image_id': r['image_id'],
                   'num': int(re.search(r'\((\d+)\)', r['image_id']).group(1)),
                   'md5': digest, 'orig_w': img.width, 'orig_h': img.height,
                   'duplicates': [], 'img': img}
            first_of[digest] = rec
            kept.append(rec)

        # 2. Series grouping on small thumbnails
        thumbs = np.stack([np.asarray(r['img'].resize((THUMB, THUMB), Image.BOX), np.float32).ravel() for r in kept])
        series = series_groups(thumbs)

        # 3. Series-level split
        sizes = np.bincount(series)
        split_of_series = split_series(sizes, rng)
        for r, s in zip(kept, series):
            r['series_id'] = f'{cls}_{s:03d}'
            r['split'] = split_of_series[s]

        # 4. Square pad + resize + write
        for r in kept:
            img = pad_to_square(r.pop('img')).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
            dest = out / r['split'] / cls / f"{cls}-{r['num']:04d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, optimize=True)
            r['file'] = dest.relative_to(out).as_posix()
        n_dup = sum(len(r['duplicates']) for r in kept)
        print(f'{cls}: {len(rows)} files -> {len(kept)} unique ({n_dup} duplicates) in {len(sizes)} series')
        records += kept

    df = pd.DataFrame(records)
    df['duplicates'] = df['duplicates'].str.join(';')
    df = df[['file', 'label', 'split', 'series_id', 'image_id', 'md5', 'orig_w', 'orig_h', 'duplicates']]
    df.to_csv(out / 'manifest.csv', index=False)

    print('\nImages per split:')
    print(pd.crosstab(df['label'], df['split'], margins=True)[['train', 'val', 'test', 'All']])
    print('\nSeries per split:')
    print(pd.crosstab(df.drop_duplicates('series_id')['label'], df.drop_duplicates('series_id')['split'],
                      margins=True)[['train', 'val', 'test', 'All']])


if __name__ == '__main__':
    main()
