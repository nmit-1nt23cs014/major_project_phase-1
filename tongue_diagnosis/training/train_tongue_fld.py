"""Train an image-only fatty liver classifier on the public Tongue-FLD data.

The dataset includes ultrasound-assessed FLD grades. This script maps grade 0
to No FLD and grades 1/2 to FLD, extracts frozen ImageNet MobileNetV2 features,
fits a balanced logistic head, selects its threshold on validation data, then
evaluates the locked test split once. No physiological indicators are used.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessor import preprocess_image
from modules.segmentor import segment_tongue


DATA_ROOT = PROJECT_ROOT / "data" / "external" / "tongue_fld"
IMAGE_DIR = DATA_ROOT / "images_full" / "Tongue_Images"
LABELS_CSV = DATA_ROOT / "raw_tar" / "Indicator_and_Label.csv"
SPLIT_CSV = PROJECT_ROOT / "data" / "manifests" / "tongue_fld_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "training" / "saved_model"
MODEL_PATH = OUTPUT_DIR / "fld_model.h5"
METRICS_PATH = OUTPUT_DIR / "fld_metrics.json"
INPUT_SIZE = (224, 224)
SEED = 42
BATCH_SIZE = 32


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_samples() -> tuple[list[Path], np.ndarray]:
    if not LABELS_CSV.is_file() or not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            "Tongue-FLD data is incomplete. Expected labels at "
            f"{LABELS_CSV} and images under {IMAGE_DIR}."
        )

    paths: list[Path] = []
    labels: list[int] = []
    with LABELS_CSV.open(newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            image_path = IMAGE_DIR / row["quesid"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in labels is missing: {image_path}")
            grade = row["FLD"].strip().strip('"')
            if grade not in {"_0", "_1", "_2"}:
                raise ValueError(f"Unexpected FLD grade {grade!r} for {row['quesid']}")
            paths.append(image_path)
            labels.append(int(grade != "_0"))

    return paths, np.asarray(labels, dtype=np.int32)


def create_split(paths: list[Path], labels: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(paths))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        remainder_idx, test_size=0.50, random_state=SEED, stratify=labels[remainder_idx]
    )
    split = {"train": train_idx, "validation": val_idx, "test": test_idx}

    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_CSV.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["relative_path", "label", "split"])
        for split_name, split_indices in split.items():
            for idx in split_indices:
                writer.writerow([
                    paths[int(idx)].relative_to(PROJECT_ROOT).as_posix(),
                    "FLD" if labels[int(idx)] else "No_FLD",
                    split_name,
                ])
    return split


def build_feature_extractor() -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3), name="tongue_image")
    x = layers.Lambda(
        lambda tensor: mobilenet_v2.preprocess_input(tensor * 255.0),
        name="mobilenetv2_preprocess",
    )(inputs)
    backbone = MobileNetV2(input_shape=(*INPUT_SIZE, 3), include_top=False, weights="imagenet")
    backbone.trainable = False
    feature_map = backbone(x, training=False)
    features = layers.GlobalAveragePooling2D(name="gap")(feature_map)
    return tf.keras.Model(inputs, features, name="TongueFLDFeatureExtractor"), tf.keras.Model(
        inputs, feature_map, name="TongueFLDBackbone"
    )


def preprocess_batch(paths: list[Path]) -> np.ndarray:
    batch = []
    for path in paths:
        image = preprocess_image(path)
        image = segment_tongue(image)
        if image.shape[:2] != INPUT_SIZE:
            image = tf.image.resize(image, INPUT_SIZE).numpy()
        batch.append(image.astype(np.float32))
    return np.asarray(batch, dtype=np.float32)


def extract_features(model: tf.keras.Model, paths: list[Path]) -> np.ndarray:
    features = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        images = preprocess_batch(batch_paths)
        features.append(model.predict(images, verbose=0))
        if (start // BATCH_SIZE + 1) % 20 == 0 or start + BATCH_SIZE >= len(paths):
            print(f"[FLD] Extracted features for {min(start + len(batch_paths), len(paths))}/{len(paths)} images")
    return np.concatenate(features, axis=0)


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "support": int(len(y_true)),
    }


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_metrics: dict[str, float | int] | None = None
    best_key: tuple[float, float, float] | None = None
    candidates = np.unique(np.concatenate(([0.05, 0.95], scores)))
    for threshold in candidates:
        metrics = metrics_at_threshold(y_true, scores, float(threshold))
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def main() -> None:
    set_seed()
    paths, labels = load_samples()
    split = create_split(paths, labels)
    print(f"[FLD] Samples: {len(paths)}; No FLD={(labels == 0).sum()}; FLD={(labels == 1).sum()}")
    print(
        "[FLD] Split sizes: "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split.items())
    )

    feature_model, backbone_model = build_feature_extractor()
    all_features = extract_features(feature_model, paths)
    train_idx = split["train"]
    val_idx = split["validation"]
    test_idx = split["test"]

    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"
    )
    classifier.fit(all_features[train_idx], labels[train_idx])
    val_scores = classifier.predict_proba(all_features[val_idx])[:, 1]
    threshold, val_metrics = select_threshold(labels[val_idx], val_scores)

    test_scores = classifier.predict_proba(all_features[test_idx])[:, 1]
    test_metrics = metrics_at_threshold(labels[test_idx], test_scores, threshold)

    # Export the frozen pretrained backbone and fitted logistic head as one Keras model.
    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3), name="tongue_image")
    x = layers.Lambda(
        lambda tensor: mobilenet_v2.preprocess_input(tensor * 255.0),
        name="mobilenetv2_preprocess",
    )(inputs)
    backbone = MobileNetV2(input_shape=(*INPUT_SIZE, 3), include_top=False, weights="imagenet")
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(backbone(x, training=False))
    output = layers.Dense(1, activation="sigmoid", name="fld_output")(x)
    model = tf.keras.Model(inputs, output, name="TongueFLDImageClassifier")
    model.get_layer("fld_output").set_weights(
        [classifier.coef_.astype(np.float32).T, classifier.intercept_.astype(np.float32)]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)

    output_data = {
        "task": "fatty_liver_disease_vs_no_fld",
        "source_dataset": "Tongue-FLD (MFF-TDF/HM-TDF release)",
        "source_label": "FLD grade 0 is No FLD; grades 1 and 2 are FLD",
        "input_modalities_used": ["segmented tongue image"],
        "input_modalities_not_used": ["gender", "age", "height", "waist", "hipline", "weight", "SBP", "DBP"],
        "seed": SEED,
        "split": {name: int(len(indices)) for name, indices in split.items()},
        "validation_threshold_selection": {
            "criterion": "balanced accuracy, then F1, then closest to 0.5",
            "threshold": threshold,
            "metrics": val_metrics,
        },
        "test_metrics": test_metrics,
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "split_manifest": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
        "limitations": [
            "Image-only model; the cited study also used eight physiological indicators.",
            "Image-level split; subject-level independence was not confirmed from the released labels.",
            "This is an FLD classifier, not a general gastrointestinal-disease classifier.",
            "Scores are not calibrated clinical probabilities.",
        ],
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(output_data, file_obj, indent=2)

    print(f"[FLD] Validation threshold: {threshold:.4f}; val balanced accuracy={val_metrics['balanced_accuracy']:.4f}")
    print(f"[FLD] Locked test accuracy={test_metrics['accuracy']:.4f}; AUC={test_metrics['roc_auc']:.4f}")
    print(f"[FLD] Saved model to {MODEL_PATH}")
    print(f"[FLD] Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
