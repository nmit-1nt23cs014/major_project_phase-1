"""Run controlled MobileNetV2 experiments from the immutable split manifest.

This module intentionally does not use the legacy training script or checkpoint.
It preserves the observed folder labels for an explicitly unverified exploratory
task, trains only on ``train``, and selects checkpoints only on ``validation``.
The ``test`` rows are not loaded by this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessor import preprocess_image
from modules.segmentor import segment_tongue


INPUT_SHAPE = (224, 224, 3)
LABEL_ORDER = ("Normal", "GI_Disease")
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv"
DEFAULT_TASK_CONFIG = PROJECT_ROOT / "configs" / "task_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "manifest_seed42"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    preprocessing: str
    freeze_backbone: bool
    augmentation: bool
    learning_rate: float
    dropout_rate: float = 0.3
    fine_tune_last_layers: int = 0


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    # Existing architecture/hyperparameters under the new manifest split.
    "segmented_frozen": ExperimentSpec("segmented_frozen", "segmented", True, True, 1e-4),
    # Controlled preprocessing comparisons with every other setting unchanged.
    "raw_resize_frozen": ExperimentSpec("raw_resize_frozen", "raw_resize", True, True, 1e-4),
    "aspect_pad_frozen": ExperimentSpec("aspect_pad_frozen", "aspect_pad", True, True, 1e-4),
    # Isolates the effect of the legacy random augmentation recipe.
    "segmented_no_augmentation": ExperimentSpec("segmented_no_augmentation", "segmented", True, False, 1e-4),
    # Requires a selected frozen experiment checkpoint as initialization.
    "finetune_last20_lr1e-5": ExperimentSpec(
        "finetune_last20_lr1e-5", "from_initialization", False, True, 1e-5, fine_tune_last_layers=20
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_config(path: Path, allow_unverified_exploratory: bool) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if config.get("task_status") != "unverified":
        raise ValueError("This runner currently expects the explicit unverified-task configuration.")
    if not allow_unverified_exploratory:
        raise ValueError(
            "The task is unverified. Re-run only with --allow-unverified-exploratory to record a "
            "non-clinical exploratory experiment."
        )
    return config


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    required = {"relative_path", "current_label", "valid", "split", "split_group_id", "sha256"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    usable = [row for row in rows if row["valid"].lower() == "true"]
    if {row["split"] for row in usable} != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test assignments.")
    unexpected = sorted({row["current_label"] for row in usable}.difference(LABEL_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected observed labels in manifest: {unexpected}")
    return usable


def load_raw_resize(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(INPUT_SHAPE[:2], Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_aspect_pad(path: Path) -> np.ndarray:
    """Resize without aspect-ratio distortion and edge-pad to the model size."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    scale = min(INPUT_SHAPE[0] / height, INPUT_SHAPE[1] / width)
    resized_height, resized_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = np.asarray(
        Image.fromarray(rgb).resize((resized_width, resized_height), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    pad_y, pad_x = INPUT_SHAPE[0] - resized_height, INPUT_SHAPE[1] - resized_width
    top, bottom = pad_y // 2, pad_y - (pad_y // 2)
    left, right = pad_x // 2, pad_x - (pad_x // 2)
    padded = np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode="edge")
    return padded.astype(np.float32) / 255.0


def prepare_image(path: Path, preprocessing: str) -> np.ndarray:
    if preprocessing == "raw_resize":
        return load_raw_resize(path)
    if preprocessing == "aspect_pad":
        return load_aspect_pad(path)
    if preprocessing == "segmented":
        return segment_tongue(preprocess_image(path)).astype(np.float32)
    raise ValueError(f"Unknown preprocessing variant: {preprocessing}")


def load_split(rows: list[dict[str, str]], split: str, preprocessing: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["relative_path"])
    images, labels, paths = [], [], []
    for row in selected:
        image_path = PROJECT_ROOT / row["relative_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest image no longer exists: {image_path}")
        images.append(prepare_image(image_path, preprocessing))
        labels.append(LABEL_ORDER.index(row["current_label"]))
        paths.append(row["relative_path"])
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32), paths


def build_model(dropout_rate: float) -> tf.keras.Model:
    base = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base._name = "mobilenetv2_backbone"
    base.trainable = False
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="tongue_image")
    x = layers.Lambda(lambda tensor: mobilenet_v2.preprocess_input(tensor * 255.0), name="mobilenetv2_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="ManifestMobileNetV2")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )


def configure_fine_tuning(model: tf.keras.Model, last_layers: int) -> None:
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-last_layers:]:
        # Keep all BatchNorm statistics frozen for this small dataset.
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


def generators(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    augmentation: bool, batch_size: int, seed: int,
) -> tuple[Any, Any]:
    train_kwargs: dict[str, Any] = {}
    if augmentation:
        train_kwargs = {
            "horizontal_flip": True,
            "rotation_range": 10,
            "width_shift_range": 0.05,
            "height_shift_range": 0.05,
            "zoom_range": 0.05,
            "fill_mode": "nearest",
        }
    train_gen = ImageDataGenerator(**train_kwargs).flow(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    val_gen = ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size, shuffle=False)
    return train_gen, val_gen


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity), "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "support": int(len(y_true)),
    }


def write_predictions(path: Path, relative_paths: list[str], y_true: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["relative_path", "observed_label", "score"])
        writer.writeheader()
        for relative_path, label, score in zip(relative_paths, y_true, scores):
            writer.writerow({"relative_path": relative_path, "observed_label": LABEL_ORDER[int(label)], "score": f"{float(score):.10f}"})


def plot_history(history: tf.keras.callbacks.History, path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(metrics: dict[str, float | int], path: Path, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
             xlabel="Predicted observed label", ylabel="True observed label", title=title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    fields = [
        "experiment", "preprocessing", "freeze_backbone", "fine_tune_last_layers", "augmentation", "learning_rate",
        "seed", "epochs_ran", "best_val_auc_history", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
        "tn", "fp", "fn", "tp", "support", "checkpoint",
    ]
    exists = results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def run_experiment(
    spec: ExperimentSpec, rows: list[dict[str, str]], output_dir: Path, seed: int, max_epochs: int,
    patience: int, batch_size: int, initial_checkpoint: Path | None, initial_preprocessing: str | None,
) -> None:
    experiment_dir = output_dir / spec.name
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {experiment_dir}")
    if spec.fine_tune_last_layers and initial_checkpoint is None:
        raise ValueError(f"{spec.name} requires --initial-checkpoint from a selected frozen experiment.")
    preprocessing = initial_preprocessing if spec.preprocessing == "from_initialization" else spec.preprocessing
    if preprocessing is None:
        raise ValueError("Fine-tuning requires an initialization preprocessing configuration.")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # The test split is intentionally not loaded here.
    x_train, y_train, _ = load_split(rows, "train", preprocessing)
    x_val, y_val, val_paths = load_split(rows, "validation", preprocessing)
    train_gen, val_gen = generators(x_train, y_train, x_val, y_val, spec.augmentation, batch_size, seed)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(np.unique(y_train), weights)}

    if initial_checkpoint is not None:
        model = tf.keras.models.load_model(initial_checkpoint)
        configure_fine_tuning(model, spec.fine_tune_last_layers)
    else:
        model = build_model(spec.dropout_rate)
    compile_model(model, spec.learning_rate)

    # TensorFlow/Keras 2.13 in the project does not support native .keras
    # checkpoint options. Keep new artifacts separate and use the existing
    # project's HDF5 compatibility format.
    checkpoint_path = experiment_dir / "best_model.h5"
    history_path = experiment_dir / "training_log.csv"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7, verbose=0),
        CSVLogger(history_path),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=max_epochs, callbacks=callbacks,
                        class_weight=class_weights, verbose=2)
    best_model = tf.keras.models.load_model(checkpoint_path)
    scores = best_model.predict(x_val, batch_size=batch_size, verbose=0).ravel()
    metrics = binary_metrics(y_val, scores)

    experiment_config = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": asdict(spec),
        "task_status": "unverified_exploratory_only",
        "observed_label_mapping": {"Normal": 0, "GI_Disease": 1},
        "manifest": str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(DEFAULT_MANIFEST),
        "split_usage": {"train": len(y_train), "validation": len(y_val), "test": "not loaded"},
        "seed": seed, "batch_size": batch_size, "max_epochs": max_epochs, "early_stopping_patience": patience,
        "class_weights": class_weights, "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "preprocessing": preprocessing,
        "inference_requirement": "Use the same preprocessing variant saved here; sigmoid outputs are uncalibrated exploratory scores.",
    }
    with (experiment_dir / "experiment_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(experiment_config, file_obj, indent=2)
    with (experiment_dir / "label_mapping.json").open("w", encoding="utf-8") as file_obj:
        json.dump(experiment_config["observed_label_mapping"], file_obj, indent=2)
    with (experiment_dir / "validation_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    write_predictions(experiment_dir / "validation_predictions.csv", val_paths, y_val, scores)
    plot_history(history, experiment_dir / "training_curves.png", spec.name)
    plot_confusion(metrics, experiment_dir / "validation_confusion_matrix.png", f"Validation: {spec.name}")
    append_result(output_dir / "experiment_results.csv", {
        "experiment": spec.name, "preprocessing": preprocessing, "freeze_backbone": spec.freeze_backbone,
        "fine_tune_last_layers": spec.fine_tune_last_layers, "augmentation": spec.augmentation,
        "learning_rate": spec.learning_rate, "seed": seed, "epochs_ran": len(history.history["loss"]),
        "best_val_auc_history": max(history.history["val_auc"]), "checkpoint": str(checkpoint_path.resolve().relative_to(PROJECT_ROOT.resolve())), **metrics,
    })
    print(f"Completed {spec.name}: validation AUC={metrics['roc_auc']:.4f}, accuracy={metrics['accuracy']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiments", default="segmented_frozen,raw_resize_frozen,aspect_pad_frozen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-preprocessing", choices=("raw_resize", "aspect_pad", "segmented"))
    parser.add_argument("--allow-unverified-exploratory", action="store_true")
    args = parser.parse_args()

    requested = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = sorted(set(requested).difference(EXPERIMENT_SPECS))
    if unknown:
        parser.error(f"Unknown experiment names: {unknown}. Choices: {sorted(EXPERIMENT_SPECS)}")
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--max-epochs, --patience, and --batch-size must be positive.")
    if not args.manifest.exists():
        parser.error(f"Manifest does not exist: {args.manifest}")
    load_task_config(args.task_config, args.allow_unverified_exploratory)
    rows = load_rows(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in requested:
        run_experiment(
            EXPERIMENT_SPECS[name], rows, args.output_dir, args.seed, args.max_epochs, args.patience,
            args.batch_size, args.initial_checkpoint, args.initial_preprocessing,
        )


if __name__ == "__main__":
    main()
