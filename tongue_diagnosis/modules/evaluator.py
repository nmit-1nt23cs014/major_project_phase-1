"""
Evaluation Module — Tongue Diagnosis Model (Phase 1)

Generates:
  - Classification report (precision, recall, F1, accuracy)
  - Confusion matrix (counts + normalized)
  - ROC curve with AUC score
  - Summary bar chart for the main metrics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
    save_dir: str | Path = "evaluation",
) -> dict[str, float]:
    """
    Run the full evaluation flow for a trained model on test data.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = ["Normal", "GI_Disease"]

    print("\n[Evaluator] Running predictions on test set...")
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
    }

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  Precision : {metrics['precision']}%")
    print(f"  Recall    : {metrics['recall']}%")
    print(f"  F1-Score  : {metrics['f1_score']}%")
    print("=" * 60)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\nDetailed Classification Report:")
    print(report_text)

    report_path = save_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("TONGUE DIAGNOSIS MODEL — EVALUATION REPORT\n")
        file_obj.write("=" * 60 + "\n")
        file_obj.write(f"Accuracy  : {metrics['accuracy']}%\n")
        file_obj.write(f"Precision : {metrics['precision']}%\n")
        file_obj.write(f"Recall    : {metrics['recall']}%\n")
        file_obj.write(f"F1-Score  : {metrics['f1_score']}%\n")
        file_obj.write("=" * 60 + "\n\n")
        file_obj.write(report_text)

    plot_confusion_matrix(y_test, y_pred, class_names, save_dir)
    plot_roc_curve(y_test, y_pred_proba, save_dir)
    plot_metrics_bar(metrics, save_dir)

    print(f"\n[Evaluator] All evaluation results saved to: {save_dir}")
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: str | Path,
) -> None:
    """
    Save the raw-count and normalized confusion matrix heatmaps side by side.
    """
    save_dir = Path(save_dir)
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in (
        (axes[0], cm, "Confusion Matrix (Counts)", "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalized)", ".2f"),
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_path = save_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluator] Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_dir: str | Path) -> None:
    """
    Save the ROC curve and include the AUC score in the legend.
    """
    save_dir = Path(save_dir)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Tongue Diagnosis Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    save_path = save_dir / "roc_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluator] ROC curve saved to: {save_path}")


def plot_metrics_bar(metrics: dict[str, float], save_dir: str | Path) -> None:
    """
    Save a bar chart summarizing the main evaluation metrics.
    """
    save_dir = Path(save_dir)
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["steelblue", "darkorange", "mediumseagreen", "tomato"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            bar.get_height() + 0.5,
            f"{value}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    save_path = save_dir / "metrics_bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluator] Metrics bar chart saved to: {save_path}")
