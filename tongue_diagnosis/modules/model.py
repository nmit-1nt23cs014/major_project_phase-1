"""
Model Definition Module — Tongue Diagnosis Model (Phase 1)

Architecture:
  - Base: MobileNetV2 pretrained on ImageNet
  - Head: GAP -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.3) -> Dense(1, Sigmoid)
  - Binary classification: Normal (0) vs GI Disease (1)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 1
DROPOUT_RATE = 0.3
DENSE_UNITS = 128
LEARNING_RATE = 1e-4
DEFAULT_MODEL_PATH = Path("training") / "saved_model" / "best_model.h5"


def build_model(freeze_base: bool = True) -> tf.keras.Model:
    """
    Build and compile the MobileNetV2-based transfer learning model.
    """
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="tongue_image")

    # The preprocessing pipeline outputs normalized [0, 1] images, so we
    # convert them to MobileNetV2's expected range inside the model.
    x = layers.Lambda(
        lambda tensor: mobilenet_v2.preprocess_input(tensor * 255.0),
        name="mobilenetv2_preprocess",
    )(inputs)
    x = base_model(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TongueDiagnosisModel")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def get_callbacks(checkpoint_dir: str | os.PathLike[str] = DEFAULT_MODEL_PATH.parent) -> list[tf.keras.callbacks.Callback]:
    """
    Return the standard callback set for training.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(
            filename=str(checkpoint_dir / "training_log.csv"),
            append=False,
        ),
    ]


def load_saved_model(model_path: str | os.PathLike[str] = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load a previously trained model from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please run training/train.py first to train and save the model."
        )

    model = tf.keras.models.load_model(model_path)
    print(f"[Model] Loaded model from: {model_path}")
    return model


def predict_single(model: tf.keras.Model, img: np.ndarray) -> dict[str, float | str]:
    """
    Run inference on a single preprocessed + segmented image.
    """
    img_batch = np.expand_dims(img, axis=0).astype(np.float32)
    raw_score = float(model.predict(img_batch, verbose=0)[0][0])

    label = "GI Disease" if raw_score >= 0.5 else "Normal"
    confidence = raw_score if raw_score >= 0.5 else 1.0 - raw_score

    return {
        "label": label,
        "confidence": round(confidence * 100.0, 2),
        "raw_score": round(raw_score, 4),
    }
