"""
Evaluation Entry Point — Tongue Diagnosis Model (Phase 1)

Loads the trained model and the segmented dataset, rebuilds the same
deterministic validation split used during training, and saves a full
evaluation report plus plots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluator import evaluate_model
from modules.model import load_saved_model
from modules.preprocessor import preprocess_batch
from modules.segmentor import segment_batch


RANDOM_SEED = 42
TEST_SPLIT = 0.20
MODEL_PATH = PROJECT_ROOT / "training" / "saved_model" / "best_model.h5"
LABEL_MAP_PATH = PROJECT_ROOT / "training" / "saved_model" / "label_map.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENTED_DIR = PROJECT_ROOT / "data" / "segmented"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def _directory_has_npy_files(root_dir: Path) -> bool:
    return root_dir.exists() and any(root_dir.rglob("*.npy"))


def _ensure_segmented_data() -> None:
    """
    Make sure the segmented arrays needed for evaluation are available.
    """
    if not _directory_has_npy_files(PROCESSED_DIR):
        print("[Evaluate] Preprocessed arrays not found. Running preprocessing first...")
        preprocess_batch(RAW_DIR, output_dir=PROCESSED_DIR)

    if not _directory_has_npy_files(SEGMENTED_DIR):
        print("[Evaluate] Segmented arrays not found. Running segmentation first...")
        segment_batch(PROCESSED_DIR, output_dir=SEGMENTED_DIR)


def _load_segmented_arrays(label_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load segmented arrays using the same class ordering stored at training time.
    """
    images: list[np.ndarray] = []
    labels: list[int] = []

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    print(f"[Evaluate] Loading segmented arrays for classes: {class_names}")

    for class_name in class_names:
        class_dir = SEGMENTED_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing segmented class directory: {class_dir}")

        npy_files = sorted(class_dir.glob("*.npy"))
        print(f"[Evaluate] Found {len(npy_files)} samples for class '{class_name}'")

        for file_path in npy_files:
            images.append(np.load(file_path).astype(np.float32))
            labels.append(int(label_map[class_name]))

    if not images:
        raise ValueError("No segmented arrays were found for evaluation.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main() -> None:
    print("\n" + "=" * 60)
    print("TONGUE DIAGNOSIS MODEL — EVALUATION")
    print("=" * 60)

    model = load_saved_model(MODEL_PATH)

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file_obj:
        label_map = json.load(file_obj)

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    print(f"[Evaluate] Classes: {class_names}")

    _ensure_segmented_data()
    X, y = _load_segmented_arrays(label_map)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"[Evaluate] Full dataset shape: {X.shape}")
    print(f"[Evaluate] Test set size: {len(X_test)} images")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        class_names=class_names,
        save_dir=EVALUATION_DIR,
    )

    print("\nEvaluation complete.")
    print(f"Results saved to: {EVALUATION_DIR}")
    print(f"Summary metrics: {metrics}")


if __name__ == "__main__":
    main()
