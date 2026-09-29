"""
evaluate.py — Evaluation of the saved lung X-ray checkpoints: model comparison,
per-class report, confusion matrix, ROC-AUC, soft-voting ensemble, confidence
threshold coverage, OOD gate sanity check, and Grad-CAM samples.

Writes:
    results/model_comparison.csv, results/classification_report.csv
    images/*.png
    models/metadata.json   (read by predict.py)

CLI:
    python -m src.evaluate --data-dir data
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, roc_auc_score, roc_curve)

from common.image import gradcam

from .data import CLASS_NAMES, DEFAULT_DATA_DIR, IMG_SIZE, MODULE_DIR, SEED, dataset_labels, load_image, make_dataset
from .explain import overlay_heatmap
from .models import BEST_MODEL, ENSEMBLE_MEMBERS, FINAL_MODELS, MODEL_DIR, available_models, load_model
from .predict import CONFIDENCE_THRESHOLD, DEFAULT_OOD_PATH, MODEL

RESULTS_DIR = MODULE_DIR / "results"
IMAGES_DIR = MODULE_DIR / "images"
LABELS = list(range(len(CLASS_NAMES)))


def predict_probabilities(model, dataset) -> tuple[np.ndarray, float]:
    """Softmax outputs for an unshuffled dataset, plus wall-clock seconds."""
    start = time.time()
    probs = model.predict(dataset, verbose=0)
    return probs, time.time() - start


def summarize(y_true, probs) -> dict:
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_roc_auc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=LABELS),
        "log_loss": log_loss(y_true, probs, labels=LABELS),
    }


def per_class_report(y_true, probs) -> pd.DataFrame:
    report = classification_report(y_true, probs.argmax(axis=1), labels=LABELS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def confidence_coverage(y_true, probs, threshold: float) -> dict:
    """How many images clear the Stage 3 threshold, and how accurate those are."""
    confident = probs.max(axis=1) >= threshold
    y_pred = probs.argmax(axis=1)
    return {
        "threshold": threshold,
        "coverage": float(confident.mean()),
        "accuracy_accepted": float((y_pred[confident] == y_true[confident]).mean()) if confident.any() else None,
        "accuracy_flagged": float((y_pred[~confident] == y_true[~confident]).mean()) if (~confident).any() else None,
    }


def ood_sanity_check(test_ds, n_noise: int = 64) -> dict:
    """Share of real test X-rays the gate rejects vs. share of random-noise images it rejects."""
    flagged_real = np.concatenate([MODEL.score(images.numpy())[0] for images, _ in test_ds])   # tf.data batches
    rng = np.random.default_rng(SEED)
    noise = rng.integers(0, 256, size=(n_noise, *IMG_SIZE, 3)).astype(np.float32) / 255.0
    flagged_noise = MODEL.score(noise)[0]
    return {
        "test_xrays_rejected": float(flagged_real.mean()),
        "noise_images_rejected": float(flagged_noise.mean()),
    }


# ---------- plots ----------

def plot_model_comparison(comparison: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison))
    ax.bar(x - 0.2, comparison["Test Accuracy"], 0.4, label="Test accuracy")
    ax.bar(x + 0.2, comparison["Test Macro F1"], 0.4, label="Test macro F1")
    ax.set_xticks(x, comparison["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Lung X-ray model comparison (test set)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, probs, title: str, path: Path):
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=LABELS)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(LABELS, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(LABELS, CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc_curves(y_true, probs, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
        auc = roc_auc_score(y_true == i, probs[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gradcam_samples(model, file_paths, y_true, path: Path):
    """Original / heatmap / overlay for one random test image per class."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(11, 3.6 * len(CLASS_NAMES)))
    for row, name in enumerate(CLASS_NAMES):
        img_path = file_paths[rng.choice(np.flatnonzero(y_true == row))]
        img_array = load_image(img_path)
        heatmap, probs = gradcam(model, img_array)          # heatmap of the top class
        pred = int(probs.argmax())

        axes[row, 0].imshow(img_array[0])
        axes[row, 0].set_title(f"True: {name}")
        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Grad-CAM heatmap")
        axes[row, 2].imshow(overlay_heatmap(img_array[0], heatmap))
        axes[row, 2].set_title(f"Pred: {CLASS_NAMES[pred]} ({probs[pred]:.1%})",
                               color="darkgreen" if pred == row else "red")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved lung X-ray checkpoints.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models", nargs="+", default=FINAL_MODELS)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--skip-val", action="store_true", help="Only score the test split (faster on CPU).")
    parser.add_argument("--metadata-out", default=str(MODEL_DIR / "metadata.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    names = [n for n in args.models if n in available_models()]
    missing = sorted(set(args.models) - set(names))
    if missing:
        print(f"[SKIP] Checkpoints not found: {missing}")

    test_ds = make_dataset(args.data_dir, "test")
    y_test = dataset_labels(test_ds)
    val_ds = None if args.skip_val else make_dataset(args.data_dir, "val")
    y_val = None if val_ds is None else dataset_labels(val_ds)

    # --- Model comparison ---
    rows, test_probs = [], {}
    for name in names:
        print(f"Evaluating {name}...")
        model = load_model(name)
        probs, seconds = predict_probabilities(model, test_ds)
        test_probs[name] = probs
        test = summarize(y_test, probs)
        row = {
            "Model": name,
            "Test Accuracy": test["accuracy"],
            "Test Macro F1": test["macro_f1"],
            "Test Macro ROC-AUC": test["macro_roc_auc"],
            "Test Loss": test["log_loss"],
        }
        if val_ds is not None:
            val = summarize(y_val, predict_probabilities(model, val_ds)[0])
            row.update({"Val Accuracy": val["accuracy"], "Val Loss": val["log_loss"]})
        row.update({"Total Parameters": model.count_params(), "Test Eval Time (s)": round(seconds, 1)})
        rows.append(row)

    comparison = pd.DataFrame(rows)

    # --- Soft-voting ensemble ---
    ensemble = None
    if all(m in test_probs for m in ENSEMBLE_MEMBERS):
        ensemble_probs = np.mean([test_probs[m] for m in ENSEMBLE_MEMBERS], axis=0)
        ensemble = summarize(y_test, ensemble_probs)
        comparison = pd.concat([comparison, pd.DataFrame([{
            "Model": "Ensemble (soft vote)",
            "Test Accuracy": ensemble["accuracy"],
            "Test Macro F1": ensemble["macro_f1"],
            "Test Macro ROC-AUC": ensemble["macro_roc_auc"],
            "Test Loss": ensemble["log_loss"],
        }])], ignore_index=True)

    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(comparison, IMAGES_DIR / "model_comparison.png")

    if BEST_MODEL not in test_probs:
        print(f"\n[STOP] {BEST_MODEL} was not evaluated; skipping report, plots and metadata.")
        return

    # --- Best model: per-class report, confusion matrix, ROC ---
    best_probs = test_probs[BEST_MODEL]
    report = per_class_report(y_test, best_probs)
    print(f"\n=== Classification Report ({BEST_MODEL}) ===")
    print(report.to_string(float_format=lambda v: f"{v:.3f}"))
    report.to_csv(RESULTS_DIR / "classification_report.csv")

    plot_confusion_matrix(y_test, best_probs, f"Confusion Matrix - {BEST_MODEL}", IMAGES_DIR / "confusion_matrix.png")
    plot_roc_curves(y_test, best_probs, f"ROC-AUC Curves - {BEST_MODEL}", IMAGES_DIR / "roc_auc_curves.png")

    # --- Stage 3: confidence threshold ---
    coverage = confidence_coverage(y_test, best_probs, args.threshold)
    print(f"\n=== Confidence threshold {args.threshold:.2f} ===")
    print(coverage)

    # --- Stage 1: OOD gate ---
    ood_stats = None
    if Path(DEFAULT_OOD_PATH).exists():
        ood_stats = ood_sanity_check(test_ds)
        print("\n=== OOD gate ===")
        print(ood_stats)
    else:
        print(f"\n[SKIP] No OOD gate at {DEFAULT_OOD_PATH}")

    # --- Stage 4: Grad-CAM ---
    plot_gradcam_samples(load_model(BEST_MODEL), test_ds.file_paths, y_test, IMAGES_DIR / "gradcam_samples.png")

    metadata = {
        "class_names": CLASS_NAMES,
        "img_size": list(IMG_SIZE),
        "preprocessing": "RGB, resize to 224x224, divide by 255",
        "best_model": BEST_MODEL,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "confidence_threshold": args.threshold,
        "test_metrics": summarize(y_test, best_probs),
        "ensemble_test_metrics": ensemble,
        "confidence_coverage": coverage,
        "ood_gate": ood_stats,
    }
    out_path = Path(args.metadata_out)
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n[SAVED] Evaluation metadata -> {out_path}")


if __name__ == "__main__":
    main()
