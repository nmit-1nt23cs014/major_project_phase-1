"""Print basic class counts for the raw dataset."""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "raw"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    total = 0
    classes = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())

    for class_dir in classes:
        count = sum(1 for file_path in class_dir.iterdir() if file_path.suffix.lower() in VALID_EXTENSIONS)
        print(f"  Class '{class_dir.name}': {count} images")
        total += count

    print(f"  TOTAL: {total} images")
    print("=" * 50)

    if total < 200:
        print("WARNING: Dataset has fewer than 200 images.")
        print("Consider using the fallback dataset generator or augmentation.")


if __name__ == "__main__":
    main()
