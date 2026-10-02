"""
evaluate.py — Checks that src/ serves exactly what notebook/Heart_ECG.ipynb evaluated:
re-scores the test split through the same cascade code predict.py uses, and compares the
test metrics, the confidence coverage and the OOD gate's rejection rate with the values
the notebook saved in models/metadata.json. Writes nothing; exits 1 on a mismatch.

The models are trained by the notebook (see the README), not by a script here.

CLI (from heart/image_based/, after `python data/build_ecg_dataset.py`):
    python -m src.evaluate
    python -m src.evaluate --ensemble
"""

import argparse
import sys

from common.evaluation import compare_with_notebook

from .data import DEFAULT_DATA_DIR, load_split
from .predict import MODEL


def main():
    parser = argparse.ArgumentParser(description="Compare src/ with the notebook's test metrics.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--ensemble", action="store_true", help="Check the soft-voting ensemble instead.")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    images, labels, _ = load_split("test", args.data_dir)
    ok = compare_with_notebook(MODEL, images, labels, "Test ECGs", ensemble=args.ensemble, tolerance=args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
