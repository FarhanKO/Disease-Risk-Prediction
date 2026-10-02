"""
train.py — Two-phase training of the four architectures, then fitting of the
Stage 1 out-of-distribution (OOD) gate. Saves:

    Phase 1 (frozen backbone):  models/<Name>_best.keras
    Phase 2 (fine-tuned):       models/<Name>_FT_best.keras
    Stage 1 (OOD gate):         models/lung_ood_detector.joblib
    Training curves:            results/training_history.json

CLI:
    python -m src.train --data-dir data
    python -m src.train --data-dir data --ood-only      # refit the gate from saved checkpoints
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .data import (BATCH_SIZE, DEFAULT_DATA_DIR, MODULE_DIR, SEED,
                   compute_class_weights, make_dataset)
from .models import (BEST_MODEL, MODEL_DIR, NUM_UNFREEZE, TRANSFER_BACKBONES,
                     build_custom_cnn, build_transfer_model, extract_embeddings,
                     load_model, model_path, unfreeze_top_layers)

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

OOD_SAMPLES = 2000
OOD_CONTAMINATION = 0.01
DEFAULT_OOD_PATH = MODEL_DIR / "lung_ood_detector.joblib"
DEFAULT_HISTORY_PATH = MODULE_DIR / "results" / "training_history.json"


def get_callbacks(checkpoint_path: Path) -> list:
    """Early stopping on val_loss, LR decay on plateau, keep the best val_accuracy."""
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    return [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]


def fit_phase(model, train_ds, val_ds, class_weights, epochs, learning_rate, checkpoint_path) -> dict:
    """Compile and fit one model; returns its Keras history dict."""
    from keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(checkpoint_path),
    )
    return {k: [float(v) for v in values] for k, values in history.history.items()}


def train_all(data_dir, model_dir, architectures, initial_epochs, fine_tune_epochs, num_unfreeze) -> dict:
    """
    Phase 1: every architecture with its backbone frozen (lr 1e-4).
    Phase 2: transfer models only — last `num_unfreeze` backbone layers
    unfrozen, lr 1e-5, continuing from Phase 1's restored weights.
    """
    train_ds = make_dataset(data_dir, "train", augment=True)
    val_ds = make_dataset(data_dir, "val")
    class_weights = compute_class_weights(data_dir)

    built = {}
    for name in architectures:
        if name == "Custom_CNN":
            built[name] = (build_custom_cnn(), None)
        else:
            built[name] = build_transfer_model(name)

    histories = {}
    for name, (model, _) in built.items():
        print(f"\n{'=' * 50}\nTraining {name} - Phase 1\n{'=' * 50}")
        histories[f"{name}_phase1"] = fit_phase(
            model, train_ds, val_ds, class_weights, initial_epochs, INITIAL_LR,
            model_path(name, model_dir),
        )

    for name, (model, base) in built.items():
        if base is None or fine_tune_epochs == 0:
            continue
        print(f"\n{'=' * 50}\nFine-Tuning {name}_FT - Phase 2\n{'=' * 50}")
        unfreeze_top_layers(base, num_unfreeze)
        histories[f"{name}_phase2"] = fit_phase(
            model, train_ds, val_ds, class_weights, fine_tune_epochs, FINE_TUNE_LR,
            model_path(f"{name}_FT", model_dir),
        )

    return histories


def fit_ood_detector(model, data_dir, n_samples: int = OOD_SAMPLES) -> IsolationForest:
    """
    Stage 1 gate: IsolationForest over the classifier's pooled backbone
    embeddings of `n_samples` training X-rays. Uses un-augmented images so
    the gate learns what a real upload looks like.
    """
    train_ds = make_dataset(data_dir, "train", augment=False, shuffle=True)
    n_batches = int(np.ceil(n_samples / BATCH_SIZE))

    embeddings = [extract_embeddings(model, images) for images, _ in train_ds.take(n_batches)]
    embeddings = np.concatenate(embeddings)[:n_samples]
    print(f"Fitting IsolationForest on {len(embeddings)} embeddings of dim {embeddings.shape[1]}...")

    detector = IsolationForest(n_estimators=100, contamination=OOD_CONTAMINATION, random_state=SEED)
    detector.fit(embeddings)
    return detector


def main():
    parser = argparse.ArgumentParser(description="Train the lung X-ray classifiers and fit the OOD gate.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Folder containing train/val/test.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--architectures", nargs="+", default=["Custom_CNN", *TRANSFER_BACKBONES],
                        choices=["Custom_CNN", *TRANSFER_BACKBONES])
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--num-unfreeze", type=int, default=NUM_UNFREEZE)
    parser.add_argument("--ood-model", default=BEST_MODEL, help="Checkpoint whose embeddings the gate uses.")
    parser.add_argument("--ood-out", default=str(DEFAULT_OOD_PATH))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--ood-only", action="store_true", help="Skip training; refit the gate only.")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if not args.ood_only:
        histories = train_all(
            args.data_dir, args.model_dir, args.architectures,
            args.initial_epochs, args.fine_tune_epochs, args.num_unfreeze,
        )
        history_path = Path(args.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(histories, indent=2))
        print(f"\n[SAVED] Training history -> {history_path}")

    # Reload from disk so the gate matches the checkpoint predict.py will serve
    model = load_model(args.ood_model, args.model_dir)
    detector = fit_ood_detector(model, args.data_dir)
    # Stored with the model name so predict.py embeds with the same checkpoint
    joblib.dump({"detector": detector, "embedding_model": args.ood_model}, args.ood_out)
    print(f"[SAVED] OOD detector ({args.ood_model} embeddings) -> {args.ood_out}")


if __name__ == "__main__":
    main()
