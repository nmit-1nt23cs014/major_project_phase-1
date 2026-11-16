"""Pretrain MobileNetV2 on Tongue-FLD, then fine-tune the current two-class task.

The source task is fatty-liver status. Its labels are used only to adapt visual
features; the final head is freshly trained on the project's current folders.
The local folder labels are unverified and must not be presented as clinical GI
diagnoses.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.manifest_experiments import (  # noqa: E402
    DEFAULT_MANIFEST, LABEL_ORDER, build_model, load_rows, load_split, set_seed,
)

FLD_ROOT = ROOT / "data" / "external" / "tongue_fld"
FLD_IMAGES = FLD_ROOT / "images_full" / "Tongue_Images"
FLD_LABELS = FLD_ROOT / "raw_tar" / "Indicator_and_Label.csv"
FLD_SPLIT = ROOT / "data" / "manifests" / "tongue_fld_split.csv"
OUT = ROOT / "training" / "saved_model"
PRETRAINED_WEIGHTS = OUT / "fld_pretrained_backbone.weights.h5"
SOURCE_SIZE = (160, 160)
TARGET_SIZE = (224, 224)
SEED = 42


def fld_train_rows() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    split_by_name: dict[str, str] = {}
    with FLD_SPLIT.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_by_name[Path(row["relative_path"]).name] = row["split"]

    paths, labels = [], []
    with FLD_LABELS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["quesid"]
            if split_by_name.get(name) != "train":
                continue
            paths.append(str(FLD_IMAGES / name))
            labels.append(int(row["FLD"].strip().strip('"') != "_0"))
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths, np.asarray(labels, dtype=np.int32), test_size=0.10,
        random_state=SEED, stratify=np.asarray(labels, dtype=np.int32),
    )
    return train_paths, y_train, val_paths, y_val


def dataset(paths: list[str], labels: np.ndarray, size: tuple[int, int], training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, size, antialias=True) / 255.0
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def source_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input((*SOURCE_SIZE, 3), name="tongue_image")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.035)(x)
    x = layers.Lambda(lambda z: mobilenet_v2.preprocess_input(z * 255.0))(x)
    backbone = MobileNetV2(input_shape=(*SOURCE_SIZE, 3), include_top=False, weights="imagenet")
    backbone._name = "mobilenetv2_backbone"
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D()(backbone(x, training=False))
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="TongueFLDPretrainer")
    model.compile(optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model, backbone


def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    pred = (probabilities >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "support": int(len(y)),
    }


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    train_paths, y_train, val_paths, y_val = fld_train_rows()
    print(f"[Pretrain] FLD train/validation: {len(y_train)}/{len(y_val)}")
    pretrainer, backbone = source_model()
    if PRETRAINED_WEIGHTS.exists():
        print(f"[Pretrain] Reusing saved tongue-adapted backbone: {PRETRAINED_WEIGHTS}")
        backbone.load_weights(PRETRAINED_WEIGHTS)
    else:
        source_train = dataset(train_paths, y_train, SOURCE_SIZE, True)
        source_val = dataset(val_paths, y_val, SOURCE_SIZE, False)
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        class_weight = {0: float(weights[0]), 1: float(weights[1])}
        pretrainer.fit(source_train, validation_data=source_val, epochs=2, class_weight=class_weight, verbose=2)

        # Adapt the last 20 convolutional layers using only the source train partition.
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        pretrainer.compile(optimizers.Adam(1e-5), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        pretrainer.fit(source_train, validation_data=source_val, epochs=3, class_weight=class_weight, verbose=2)
        backbone.save_weights(PRETRAINED_WEIGHTS)
        print(f"[Pretrain] Saved adapted backbone: {PRETRAINED_WEIGHTS}")
    backbone_weights = backbone.get_weights()

    # Initialize the same MobileNetV2 classifier used by the current project,
    # but train its prediction head on the existing Normal/GI_Disease folders.
    rows = load_rows(DEFAULT_MANIFEST)
    target_train, y_target_train, _ = load_split(rows, "train", "segmented")
    target_val, y_target_val, target_val_paths = load_split(rows, "validation", "segmented")
    model = build_model(0.3)
    target_backbone = model.get_layer("mobilenetv2_backbone")
    target_backbone.set_weights(backbone_weights)
    target_backbone.trainable = True
    for layer in target_backbone.layers[:-20]:
        layer.trainable = False
    for layer in target_backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizers.Adam(1e-5), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    class_counts = np.bincount(y_target_train, minlength=2)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_target_train)
    target_class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(OUT / "current_pretrained_candidate.h5"), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]
    target_generator = ImageDataGenerator(
        horizontal_flip=True, rotation_range=8, width_shift_range=0.04,
        height_shift_range=0.04, zoom_range=0.04, fill_mode="nearest",
    ).flow(target_train, y_target_train, batch_size=16, shuffle=True, seed=SEED)
    model.fit(
        target_generator,
        validation_data=(target_val, y_target_val), epochs=25,
        class_weight=target_class_weight, callbacks=callbacks, verbose=2,
    )
    model = tf.keras.models.load_model(OUT / "current_pretrained_candidate.h5", custom_objects={"mobilenet_v2": mobilenet_v2})
    val_probabilities = model.predict(target_val, batch_size=16, verbose=0).ravel()
    metrics = score(y_target_val, val_probabilities)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "initialization": "MobileNetV2 pretrained on ImageNet, then adapted to Tongue-FLD train partition only",
        "fine_tuning_task": "current local folder labels: Normal vs GI_Disease (labels unverified)",
        "source_pretraining_samples": {"train": len(y_train), "validation": len(y_val)},
        "target_split": {"train": len(y_target_train), "validation": len(y_target_val)},
        "target_validation_metrics_at_0_5": metrics,
        "target_validation_predictions": [
            {"relative_path": p, "observed_label": LABEL_ORDER[int(y)], "disease_score": float(s)}
            for p, y, s in zip(target_val_paths, y_target_val, val_probabilities)
        ],
        "note": "No test split used. This is an experimental candidate and was not promoted: the backed-up app checkpoint scored better on validation after threshold calibration. Local folder labels remain unverified.",
    }
    with (OUT / "current_pretrained_candidate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[Fine-tune] Validation metrics: {json.dumps(metrics)}")
    print(f"[Fine-tune] Candidate saved to {OUT / 'current_pretrained_candidate.h5'}")


if __name__ == "__main__":
    main()
