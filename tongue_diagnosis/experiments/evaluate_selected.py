"""Select a validation-only threshold and evaluate one selected experiment on test.

The script refuses to overwrite an existing final test evaluation. It is for the
explicitly unverified exploratory task and must not be used for clinical claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

from manifest_experiments import (
    DEFAULT_MANIFEST, LABEL_ORDER, PROJECT_ROOT, binary_metrics, load_rows, load_split, plot_confusion,
)


def choose_validation_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float | int], list[dict[str, float | int]]]:
    """Choose balanced accuracy on validation; ties favour F1 then threshold near 0.5."""
    candidates = sorted(set([0.05, 0.95, *[float(value) for value in scores]]))
    rows = []
    best_threshold, best_metrics, best_key = 0.5, {}, None
    for threshold in candidates:
        metrics = binary_metrics(y_true, scores, threshold)
        balanced_accuracy = (float(metrics["recall"]) + float(metrics["specificity"])) / 2.0
        metrics["balanced_accuracy"] = balanced_accuracy
        rows.append(metrics)
        key = (balanced_accuracy, float(metrics["f1"]), -abs(threshold - 0.5))
        if best_key is None or key > best_key:
            best_threshold, best_metrics, best_key = threshold, metrics, key
    return best_threshold, best_metrics, rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total) / denominator
    return [float(center - margin), float(center + margin)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    experiment_dir = args.run_dir / args.experiment
    output_path = experiment_dir / "final_test_metrics.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to repeat final test evaluation: {output_path}")
    with (experiment_dir / "experiment_config.json").open(encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if config.get("task_status") != "unverified_exploratory_only":
        raise ValueError("Only explicitly unverified exploratory experiment artifacts are supported.")

    with (experiment_dir / "validation_predictions.csv").open(newline="", encoding="utf-8") as file_obj:
        validation_rows = list(csv.DictReader(file_obj))
    y_val = np.asarray([LABEL_ORDER.index(row["observed_label"]) for row in validation_rows], dtype=np.int32)
    val_scores = np.asarray([float(row["score"]) for row in validation_rows], dtype=np.float32)
    threshold, validation_metrics, threshold_rows = choose_validation_threshold(y_val, val_scores)
    with (experiment_dir / "threshold_selection_validation.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    # Test data is loaded only after validation-only selection is complete.
    rows = load_rows(args.manifest)
    x_test, y_test, test_paths = load_split(rows, "test", config["preprocessing"])
    model = tf.keras.models.load_model(
        experiment_dir / "best_model.h5",
        custom_objects={"mobilenet_v2": mobilenet_v2},
    )
    scores = model.predict(x_test, batch_size=int(config["batch_size"]), verbose=0).ravel()
    test_metrics = binary_metrics(y_test, scores, threshold)
    test_metrics["accuracy_95pct_wilson_ci"] = wilson_interval(
        int(test_metrics["tn"]) + int(test_metrics["tp"]), int(test_metrics["support"])
    )
    test_metrics["sensitivity_95pct_wilson_ci"] = wilson_interval(
        int(test_metrics["tp"]), int(test_metrics["tp"]) + int(test_metrics["fn"])
    )
    test_metrics["specificity_95pct_wilson_ci"] = wilson_interval(
        int(test_metrics["tn"]), int(test_metrics["tn"]) + int(test_metrics["fp"])
    )
    output = {
        "task_status": "unverified_exploratory_only",
        "test_split_was_not_used_for_model_or_threshold_selection": True,
        "threshold_selection": {
            "criterion": "validation balanced accuracy, ties broken by F1 then distance to 0.5",
            "threshold": threshold,
            "validation_metrics": validation_metrics,
        },
        "test_metrics": test_metrics,
        "test_support_by_observed_label": {"Normal": int((y_test == 0).sum()), "GI_Disease": int((y_test == 1).sum())},
        "uncertainty_note": "Wilson intervals treat images as independent. This is an image-level split; patient-level independence is not established.",
    }
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, indent=2)
    with (experiment_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["relative_path", "observed_label", "score"])
        writer.writeheader()
        for path, label, score in zip(test_paths, y_test, scores):
            writer.writerow({"relative_path": path, "observed_label": LABEL_ORDER[int(label)], "score": f"{float(score):.10f}"})
    plot_confusion(test_metrics, experiment_dir / "final_test_confusion_matrix.png", f"Final test: {args.experiment}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}; AUC: {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
