"""
Tongue Segmentation Module — Tongue Diagnosis Model (Phase 1)

Approach:
  - Convert image to HSV color space
  - Apply HSV threshold mask targeting the pinkish-red tongue region
  - Find the largest contour (assumed to be the tongue)
  - Extract a bounding-box crop of the tongue region
  - Return the masked + cropped tongue-only image
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


HSV_LOWER_1 = np.array([0, 20, 70], dtype=np.uint8)
HSV_UPPER_1 = np.array([25, 255, 255], dtype=np.uint8)
HSV_LOWER_2 = np.array([160, 20, 70], dtype=np.uint8)
HSV_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

MORPH_KERNEL = np.ones((7, 7), np.uint8)
MIN_CONTOUR_AREA = 1000

TARGET_SIZE = (224, 224)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SEGMENTED_DIR = PROJECT_ROOT / "data" / "segmented"
DEFAULT_SEGMENTATION_TEST = PROJECT_ROOT / "segmentation_test.png"


def _to_uint8_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] or uint8 [0,255] RGB image into uint8."""
    if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
        img_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    elif img_rgb.max() <= 1.0:
        img_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img_rgb.astype(np.uint8)
    return img_uint8


def _create_tongue_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for the tongue region using HSV thresholding.
    """
    img_uint8 = _to_uint8_rgb(img_rgb)
    img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(img_hsv, HSV_LOWER_1, HSV_UPPER_1)
    mask2 = cv2.inRange(img_hsv, HSV_LOWER_2, HSV_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)

    return mask


def _find_largest_contour(mask: np.ndarray):
    """Find the largest contour in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    return largest


def segment_tongue(img: np.ndarray, return_mask: bool = False):
    """
    Segment the tongue region from a preprocessed image.

    Returns the segmented image as float32 [0,1]. If no valid tongue region is
    detected, returns the original image without crashing.
    """
    original = img.astype(np.float32, copy=True)
    mask = _create_tongue_mask(original)
    contour = _find_largest_contour(mask)

    if contour is None:
        print("  [Segmentor] WARNING: Tongue region not detected. Using full image.")
        if return_mask:
            return original, mask
        return original

    x, y, w, h = cv2.boundingRect(contour)

    pad_x = int(w * 0.10)
    pad_y = int(h * 0.10)
    image_height, image_width = original.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)

    cropped_img = original[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    cropped_mask_3ch = np.stack([cropped_mask / 255.0] * 3, axis=-1).astype(np.float32)
    masked_img = cropped_img * cropped_mask_3ch

    segmented = cv2.resize(masked_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    segmented = segmented.astype(np.float32)

    if return_mask:
        return segmented, mask
    return segmented


def segment_batch(
    preprocessed_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, np.ndarray | dict[str, int]]:
    """Run segmentation on all preprocessed .npy images in a directory."""
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    classes = sorted(path.name for path in preprocessed_dir.iterdir() if path.is_dir())
    label_map = {class_name: index for index, class_name in enumerate(classes)}

    images: list[np.ndarray] = []
    labels: list[int] = []
    failed = 0

    print(f"\n[Segmentor] Running segmentation on: {preprocessed_dir}")

    for class_name in classes:
        class_dir = preprocessed_dir / class_name
        npy_files = sorted(file_path for file_path in class_dir.iterdir() if file_path.suffix == ".npy")

        print(f"\n  Segmenting class '{class_name}' ({len(npy_files)} images)...")

        for file_path in tqdm(npy_files, desc=f"  {class_name}"):
            try:
                img = np.load(file_path)
                segmented = segment_tongue(img)
                images.append(segmented)
                labels.append(label_map[class_name])

                if output_dir is not None:
                    output_class_dir = Path(output_dir) / class_name
                    output_class_dir.mkdir(parents=True, exist_ok=True)
                    np.save(output_class_dir / file_path.name, segmented)
            except Exception as exc:
                failed += 1
                print(f"\n    [WARNING] Skipping {file_path.name}: {exc}")

    image_array = np.array(images, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)

    result = {
        "images": image_array,
        "labels": label_array,
        "label_map": label_map,
    }

    print(f"\n[Segmentor] Done. Total segmented: {len(images)} images.")
    print(f"[Segmentor] Array shape: {image_array.shape}")
    print(f"[Segmentor] Array dtype: {image_array.dtype}")
    if failed:
        print(f"[Segmentor] WARNING: {failed} files failed to segment and were skipped.")

    return result


def visualize_segmentation(image_path: str | os.PathLike[str], save_path: str | os.PathLike[str] | None = None) -> None:
    """
    Save a 3-panel visualization: Original | Mask | Segmented.
    """
    import matplotlib.pyplot as plt

    try:
        from modules.preprocessor import preprocess_image
    except ModuleNotFoundError:
        from preprocessor import preprocess_image

    image_path = Path(image_path)
    save_path = Path(save_path) if save_path is not None else DEFAULT_SEGMENTATION_TEST

    img = preprocess_image(image_path)
    segmented, mask = segment_tongue(img, return_mask=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Preprocessed Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Tongue Mask")
    axes[1].axis("off")

    axes[2].imshow(segmented)
    axes[2].set_title("Segmented Tongue")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Segmentor] Visualization saved to: {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        visualize_segmentation(test_path, save_path=DEFAULT_SEGMENTATION_TEST)
        print("Segmentation test complete. Check segmentation_test.png")
    else:
        result = segment_batch(DEFAULT_PREPROCESSED_DIR, output_dir=DEFAULT_SEGMENTED_DIR)
        print(f"\nFinal shape: {result['images'].shape}")
        print("Batch segmentation complete.")
