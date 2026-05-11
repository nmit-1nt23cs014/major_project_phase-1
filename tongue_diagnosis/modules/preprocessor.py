"""
Preprocessing Module — Tongue Diagnosis Model (Phase 1)

Responsibilities:
  - Load tongue images from file path
  - Resize, denoise, and normalize images
  - Return a standardized numpy array for downstream modules

Usage:
  from modules.preprocessor import preprocess_image, preprocess_batch
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TARGET_SIZE = (224, 224)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_image(image_path: str | os.PathLike[str]) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        NumPy array in BGR format, shape (H, W, 3).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode image (possibly corrupt): {image_path}")

    return image


def resize_image(img: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize the image using an interpolation suited to the direction of scaling.
    """
    height, width = img.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if width > target_size[0] or height > target_size[1]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(img, target_size, interpolation=interpolation)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def denoise_image(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values from [0, 255] to [0.0, 1.0]."""
    return img.astype(np.float32) / 255.0


def preprocess_image(image_path: str | os.PathLike[str]) -> np.ndarray:
    """
    Full preprocessing pipeline for a single tongue image.

    Pipeline:
        Load -> Resize -> BGR->RGB -> Denoise -> Normalize
    """
    img = load_image(image_path)
    img = resize_image(img)
    img = bgr_to_rgb(img)
    img = denoise_image(img)
    img = normalize_image(img)
    return img


def preprocess_batch(
    image_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, np.ndarray | dict[str, int]]:
    """
    Preprocess all images from a directory with class subfolders.

    Expected input structure:
        image_dir/
            Normal/
            GI_Disease/
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    classes = sorted(path.name for path in image_dir.iterdir() if path.is_dir())
    label_map = {class_name: index for index, class_name in enumerate(classes)}

    images: list[np.ndarray] = []
    labels: list[int] = []
    failed: list[tuple[str, str]] = []

    print(f"\n[Preprocessor] Processing images from: {image_dir}")
    print(f"[Preprocessor] Classes found: {classes}")

    for class_name in classes:
        class_dir = image_dir / class_name
        image_files = sorted(
            file_path for file_path in class_dir.iterdir() if file_path.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"\n  Processing class '{class_name}' ({len(image_files)} images)...")

        for file_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                processed = preprocess_image(file_path)
                images.append(processed)
                labels.append(label_map[class_name])

                if output_dir is not None:
                    output_class_dir = Path(output_dir) / class_name
                    output_class_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_class_dir / f"{file_path.stem}.npy"
                    np.save(output_path, processed)
            except Exception as exc:
                failed.append((str(file_path), str(exc)))
                print(f"\n    [WARNING] Skipping {file_path.name}: {exc}")

    if failed:
        print(f"\n[Preprocessor] WARNING: {len(failed)} images failed to process.")

    image_array = np.array(images, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)

    result = {
        "images": image_array,
        "labels": label_array,
        "label_map": label_map,
    }

    print(f"\n[Preprocessor] Done. Total processed: {len(images)} images.")
    print(f"[Preprocessor] Array shape: {image_array.shape}")
    print(f"[Preprocessor] Array dtype: {image_array.dtype}")
    if image_array.size > 0:
        print(f"[Preprocessor] Observed value range: [{image_array.min():.3f}, {image_array.max():.3f}]")
    print("[Preprocessor] Normalized output range: [0.000, 1.000]")
    print(f"[Preprocessor] Label map: {label_map}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        result = preprocess_image(test_path)
        print(f"Preprocessed image shape: {result.shape}")
        print(f"Preprocessed image dtype: {result.dtype}")
        print(f"Observed value range: [{result.min():.3f}, {result.max():.3f}]")
        print("Normalized output range: [0.000, 1.000]")
        print("Preprocessor working correctly.")
    else:
        data = preprocess_batch(DEFAULT_RAW_DIR, output_dir=DEFAULT_PROCESSED_DIR)
        print(f"\nFinal dataset shape: {data['images'].shape}")
        print(f"Labels shape: {data['labels'].shape}")
        print("Batch preprocessing complete.")
