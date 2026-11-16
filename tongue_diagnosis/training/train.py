"""
Model Training Script — Tongue Diagnosis Model (Phase 1)

Runs the full training pipeline:
  1. Load segmented images from data/segmented/
  2. Apply data augmentation
  3. Split into train/validation sets (80/20)
  4. Build MobileNetV2 transfer learning model
  5. Train with callbacks
  6. Save best model to training/saved_model/best_model.h5
  7. Plot and save training curves
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessor import preprocess_batch
from modules.segmentor import segment_batch
from modules.model import build_model, get_callbacks, load_saved_model


RANDOM_SEED = 42
TEST_SPLIT = 0.20
BATCH_SIZE = 16
EPOCHS = 30

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "segmented"
MODEL_SAVE_DIR = PROJECT_ROOT / "training" / "saved_model"
MODEL_INPUT_DIR = SEGMENTED_DATA_DIR

CLASS_ORDER = ["Normal", "GI_Disease"]
LABEL_MAP = {"Normal": 0, "GI_Disease": 1}

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _directory_has_files(root_dir: Path, extension: str) -> bool:
    """Return True if any matching files exist under the class folders."""
    if not root_dir.exists():
        return False
    return any(root_dir.rglob(f"*{extension}"))


def _load_numpy_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load saved .npy arrays using the explicit Normal/GI label mapping."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    print(f"[Train] Loading model inputs from: {data_dir}")

    for class_name in CLASS_ORDER:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        npy_files = sorted(class_dir.glob("*.npy"))
        print(f"[Train] Found {len(npy_files)} samples for class '{class_name}'")

        for file_path in npy_files:
            images.append(np.load(file_path).astype(np.float32))
            labels.append(LABEL_MAP[class_name])

    if not images:
        raise ValueError("No segmented arrays were found for training.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), LABEL_MAP.copy()


def load_data() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Load model-ready segmented arrays. If missing, run preprocessing first.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    if not _directory_has_files(PROCESSED_DATA_DIR, ".npy"):
        print("[Train] Preprocessed data not found. Running preprocessing pipeline...")
        preprocess_batch(RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR)

    if not _directory_has_files(SEGMENTED_DATA_DIR, ".npy"):
        print("[Train] Segmented data not found. Generating it for downstream modules...")
        segment_batch(PROCESSED_DATA_DIR, output_dir=SEGMENTED_DATA_DIR)

    X, y, label_map = _load_numpy_arrays(MODEL_INPUT_DIR)

    print(f"\n[Train] Dataset loaded: X={X.shape}, y={y.shape}")
    print(f"[Train] Label mapping: {label_map}")
    print(f"[Train] Class distribution: {{'Normal': {int(np.sum(y == 0))}, 'GI_Disease': {int(np.sum(y == 1))}}}")
    print(f"[Train] Training input source: {MODEL_INPUT_DIR}")

    return X, y, label_map


def _compute_class_weight_map(y_train: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights to offset the dataset imbalance.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_map = {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}
    print(f"[Train] Class weights: {class_weight_map}")
    return class_weight_map


def create_data_generators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """
    Create augmented data generators for training and validation.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting Up Data Augmentation")
    print("=" * 60)

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"[Train] Training samples: {len(X_train)}")
    print(f"[Train] Validation samples: {len(X_val)}")
    print("[Train] Augmentations: flip, light rotation, light shift, light zoom")

    return train_generator, val_generator


def _plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss training curves."""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="steelblue")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="coral")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", color="steelblue")
    axes[1].plot(history.history["val_loss"], label="Validation Loss", color="coral")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MobileNetV2 Transfer Learning — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = MODEL_SAVE_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Train] Training curves saved to: {save_path}")


def _find_best_screening_threshold(
    y_true: np.ndarray,
    disease_scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Pick the sigmoid cutoff that best detects GI disease on validation data.
    """
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_key: tuple[float, float, float] | None = None

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (disease_scores >= threshold).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        key = (metrics["f1_score"], metrics["accuracy"], metrics["recall"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _find_best_balanced_threshold(
    y_true: np.ndarray,
    disease_scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Pick a stricter threshold that reduces false-positive disease labels.
    """
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_key: tuple[float, float, float] | None = None

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (disease_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        balanced_accuracy = (specificity + recall) / 2.0
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy,
        }
        key = (balanced_accuracy, metrics["accuracy"], metrics["specificity"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _save_prediction_config(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Save the calibrated disease threshold used by the Streamlit app."""
    disease_scores = model.predict(X_val, verbose=0).flatten()
    normal_threshold, screening_metrics = _find_best_screening_threshold(y_val, disease_scores)
    disease_threshold, disease_metrics = _find_best_balanced_threshold(y_val, disease_scores)

    config = {
        "normal_threshold": round(normal_threshold, 4),
        "disease_threshold": round(max(disease_threshold, normal_threshold), 4),
        "threshold_strategy": "two_zone_validation_calibration",
        "validation_screening_recall_percent_at_normal_threshold": round(
            screening_metrics["recall"] * 100.0, 2
        ),
        "validation_binary_accuracy_percent_at_disease_threshold": round(
            disease_metrics["accuracy"] * 100.0, 2
        ),
        "validation_binary_recall_percent_at_disease_threshold": round(
            disease_metrics["recall"] * 100.0, 2
        ),
        "validation_normal_specificity_percent_at_disease_threshold": round(
            disease_metrics["specificity"] * 100.0, 2
        ),
        "notes": (
            "Scores below normal_threshold are shown as Normal. Scores between normal_threshold "
            "and disease_threshold are shown as Borderline / Retake Recommended. Scores above "
            "disease_threshold are shown as Potential GI Disease Detected."
        ),
    }

    config_path = MODEL_SAVE_DIR / "prediction_config.json"
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)

    print(f"[Train] Prediction config saved to: {config_path}")
    print(f"[Train] Calibrated normal threshold: {normal_threshold:.4f}")
    print(f"[Train] Calibrated disease threshold: {disease_threshold:.4f}")
    print(
        "[Train] Threshold metrics: "
        f"screening_recall={config['validation_screening_recall_percent_at_normal_threshold']}%, "
        f"disease_threshold_accuracy={config['validation_binary_accuracy_percent_at_disease_threshold']}%, "
        f"disease_threshold_specificity={config['validation_normal_specificity_percent_at_disease_threshold']}%"
    )


def train():
    """Run the full training flow."""
    print("\n" + "=" * 60)
    print("TONGUE DIAGNOSIS MODEL — TRAINING")
    print("=" * 60)

    X, y, label_map = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val)
    class_weight_map = _compute_class_weight_map(y_train)
    print("\n" + "=" * 60)
    print("STEP 3: Building Model")
    print("=" * 60)
    model = build_model(freeze_base=True)
    model.summary()

    callbacks = get_callbacks(checkpoint_dir=MODEL_SAVE_DIR)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_SAVE_DIR / "label_map.json", "w", encoding="utf-8") as file_obj:
        json.dump(label_map, file_obj, indent=2)
    print(f"[Train] Label map saved to {MODEL_SAVE_DIR / 'label_map.json'}")

    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)
    print(f"[Train] Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("[Train] Training started...\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=1,
    )

    _plot_training_history(history)
    best_model = load_saved_model(MODEL_SAVE_DIR / "best_model.h5")
    evaluation = best_model.evaluate(val_gen, verbose=0, return_dict=True)
    _save_prediction_config(best_model, X_val, y_val)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best model saved to: {MODEL_SAVE_DIR / 'best_model.h5'}")
    print(f"Training log saved to: {MODEL_SAVE_DIR / 'training_log.csv'}")
    print(f"[Train] Best validation accuracy in history: {max(history.history['val_accuracy']):.4f}")
    print(f"[Train] Final validation accuracy (saved best model): {evaluation['accuracy']:.4f}")
    print(f"[Train] Final validation AUC (saved best model): {evaluation['auc']:.4f}")
    print("=" * 60)

    return best_model, history, X_val, y_val, label_map


if __name__ == "__main__":
    model, history, X_val, y_val, label_map = train()
    print("\nTraining script completed successfully.")
