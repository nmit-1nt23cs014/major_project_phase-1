"""
Fallback dataset generator for the tongue diagnosis prototype.

This creates a small synthetic binary dataset for pipeline testing when a
real Kaggle dataset is not available locally. Replace these images with a
real tongue image dataset before final academic submission.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ["Normal", "GI_Disease"]
IMAGES_PER_CLASS = 100
IMG_SIZE = (224, 224)
BASE_DIR = Path(__file__).resolve().parent / "raw"


def generate_synthetic_tongue_image(disease: bool = False) -> Image.Image:
    """Generate a tongue-like image with class-specific color tendencies."""
    height, width = IMG_SIZE
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if disease:
        base_color = np.array(
            [
                np.random.randint(185, 230),
                np.random.randint(160, 200),
                np.random.randint(100, 145),
            ],
            dtype=np.uint8,
        )
    else:
        base_color = np.array(
            [
                np.random.randint(200, 240),
                np.random.randint(105, 145),
                np.random.randint(100, 135),
            ],
            dtype=np.uint8,
        )

    image[:] = base_color

    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius_y, radius_x = int(height * 0.38), int(width * 0.25)
    tongue_mask = (
        ((yy - center_y) ** 2) / (radius_y**2)
        + ((xx - center_x) ** 2) / (radius_x**2)
        <= 1.0
    )

    background = np.full_like(image, fill_value=20)
    background[:, :, 1] = 15
    background[:, :, 2] = 18
    image = np.where(tongue_mask[..., None], image, background)

    noise = np.random.randint(-18, 19, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if disease:
        coating = np.zeros_like(image)
        coating[:, :, 0] = np.random.randint(220, 245)
        coating[:, :, 1] = np.random.randint(210, 235)
        coating[:, :, 2] = np.random.randint(150, 180)
        coating_mask = tongue_mask & (yy < center_y + radius_y // 3)
        image = np.where(coating_mask[..., None], (0.6 * image + 0.4 * coating).astype(np.uint8), image)

    return Image.fromarray(image)


def main() -> None:
    for class_index, class_name in enumerate(CLASSES):
        class_dir = BASE_DIR / class_name
        os.makedirs(class_dir, exist_ok=True)

        for image_index in range(IMAGES_PER_CLASS):
            image = generate_synthetic_tongue_image(disease=(class_index == 1))
            image.save(class_dir / f"{class_name}_{image_index:04d}.jpg", format="JPEG")

        print(f"Generated {IMAGES_PER_CLASS} images for class: {class_name}")

    print("Demo dataset created in data/raw/")
    print("Replace this synthetic dataset with real tongue images before final submission.")


if __name__ == "__main__":
    main()
