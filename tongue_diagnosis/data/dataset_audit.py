"""Create a non-sensitive inventory and duplicate audit for the active raw images.

The script does not alter images or labels.  It writes a CSV manifest and JSON
report that can be used by ``generate_splits.py``.  EXIF values are deliberately
not exported: they can contain GPS coordinates, device serials, or other
identifying information.  Only the presence and number of EXIF tags are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_TASK_CONFIG = PROJECT_ROOT / "configs" / "task_config.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageRecord:
    """Non-sensitive metadata needed to audit and split one image."""

    sample_id: str
    relative_path: str
    source_folder: str
    current_label: str
    valid: bool
    width: int | None
    height: int | None
    image_format: str
    color_mode: str
    file_size_bytes: int
    sha256: str
    exif_present: bool
    exif_tag_count: int
    safe_info_keys: str
    error: str
    dhash: bytes | None = None
    exact_duplicate_group: str = ""
    near_duplicate_group: str = ""
    split_group_id: str = ""


class UnionFind:
    """Small deterministic union-find implementation for near-duplicate groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> bytes:
    """Return a grayscale difference hash without retaining image content."""
    grayscale = image.convert("L").resize((33, 32))
    pixels = np.asarray(grayscale, dtype=np.uint8)
    differences = (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8)
    return np.packbits(differences).tobytes()


def hamming_distance(left: bytes, right: bytes) -> int:
    return int(np.unpackbits(np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    )).sum())


def read_task_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def audit_image(path: Path, data_dir: Path) -> ImageRecord:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    source_folder = path.parent.relative_to(PROJECT_ROOT).as_posix()
    current_label = path.parent.name
    file_hash = sha256_file(path)
    sample_id = f"sha256:{file_hash}"

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            exif = image.getexif()
            safe_info_keys = sorted(
                key for key in image.info.keys() if key.lower() not in {"exif", "icc_profile", "xmp"}
            )
            return ImageRecord(
                sample_id=sample_id,
                relative_path=relative_path,
                source_folder=source_folder,
                current_label=current_label,
                valid=True,
                width=image.width,
                height=image.height,
                image_format=image.format or "",
                color_mode=image.mode,
                file_size_bytes=path.stat().st_size,
                sha256=file_hash,
                exif_present=bool(exif),
                exif_tag_count=len(exif),
                safe_info_keys=";".join(safe_info_keys),
                error="",
                dhash=dhash(image),
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageRecord(
            sample_id=sample_id,
            relative_path=relative_path,
            source_folder=source_folder,
            current_label=current_label,
            valid=False,
            width=None,
            height=None,
            image_format="",
            color_mode="",
            file_size_bytes=path.stat().st_size,
            sha256=file_hash,
            exif_present=False,
            exif_tag_count=0,
            safe_info_keys="",
            error=f"{type(exc).__name__}: {exc}",
        )


def assign_duplicate_groups(records: list[ImageRecord], near_duplicate_distance: int) -> dict[str, Any]:
    """Assign exact/near duplicate groups and return summary counts."""
    hash_groups: dict[str, list[int]] = defaultdict(list)
    valid_indices: list[int] = []
    for index, record in enumerate(records):
        hash_groups[record.sha256].append(index)
        if record.valid and record.dhash is not None:
            valid_indices.append(index)

    exact_group_number = 0
    for indices in hash_groups.values():
        if len(indices) > 1:
            exact_group_number += 1
            group_id = f"exact_{exact_group_number:04d}"
            for index in indices:
                records[index].exact_duplicate_group = group_id

    union_find = UnionFind(len(valid_indices))
    near_pairs: list[dict[str, Any]] = []
    for local_left, record_index_left in enumerate(valid_indices):
        left = records[record_index_left]
        for local_right in range(local_left + 1, len(valid_indices)):
            record_index_right = valid_indices[local_right]
            right = records[record_index_right]
            distance = hamming_distance(left.dhash or b"", right.dhash or b"")
            if distance <= near_duplicate_distance:
                union_find.union(local_left, local_right)
                near_pairs.append({
                    "left": left.relative_path,
                    "right": right.relative_path,
                    "hamming_distance": distance,
                })

    component_members: dict[int, list[int]] = defaultdict(list)
    for local_index, record_index in enumerate(valid_indices):
        component_members[union_find.find(local_index)].append(record_index)

    near_group_number = 0
    for indices in component_members.values():
        if len(indices) > 1:
            near_group_number += 1
            group_id = f"near_{near_group_number:04d}"
            for index in indices:
                records[index].near_duplicate_group = group_id

    for record in records:
        if record.near_duplicate_group:
            record.split_group_id = record.near_duplicate_group
        elif record.exact_duplicate_group:
            record.split_group_id = record.exact_duplicate_group
        else:
            record.split_group_id = record.sample_id

    return {
        "exact_duplicate_groups": exact_group_number,
        "exact_duplicate_extra_images": sum(max(0, len(indices) - 1) for indices in hash_groups.values()),
        "near_duplicate_groups": near_group_number,
        "near_duplicate_pairs_at_or_below_hamming_distance": near_pairs,
    }


def write_manifest(records: list[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "relative_path", "source_folder", "current_label", "valid", "width", "height",
        "image_format", "color_mode", "file_size_bytes", "sha256", "exif_present", "exif_tag_count",
        "safe_info_keys", "exact_duplicate_group", "near_duplicate_group", "split_group_id", "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.relative_path):
            row = {field: getattr(record, field) for field in fields}
            writer.writerow(row)


def build_report(
    records: list[ImageRecord],
    duplicate_summary: dict[str, Any],
    task_config: dict[str, Any],
    near_duplicate_distance: int,
) -> dict[str, Any]:
    valid_records = [record for record in records if record.valid]
    return {
        "schema_version": 1,
        "task_id": task_config.get("task_id"),
        "task_status": task_config.get("task_status"),
        "clinical_claims_permitted": task_config.get("clinical_claims_permitted"),
        "dataset_root": "data/raw",
        "sample_count": len(records),
        "valid_image_count": len(valid_records),
        "invalid_image_count": len(records) - len(valid_records),
        "class_counts": dict(sorted(Counter(record.current_label for record in records).items())),
        "format_counts": dict(sorted(Counter(record.image_format for record in valid_records).items())),
        "color_mode_counts": dict(sorted(Counter(record.color_mode for record in valid_records).items())),
        "source_folder_counts": dict(sorted(Counter(record.source_folder for record in records).items())),
        "images_with_exif": sum(record.exif_present for record in valid_records),
        "metadata_privacy": "EXIF values and other potentially identifying metadata are not exported; only EXIF presence/count and safe image-info key names are retained.",
        "near_duplicate_hamming_distance": near_duplicate_distance,
        "duplicate_summary": duplicate_summary,
        "task_limitations": {
            "provenance": task_config.get("status_reason"),
            "evidence_required_for_gi_task": task_config.get("evidence_required_for_gi_task", []),
            "patient_independence": "No patient identifiers are available. Image-level duplicate grouping cannot establish patient-disjoint splits."
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    args = parser.parse_args()

    if args.near_duplicate_distance < 0:
        parser.error("--near-duplicate-distance must be non-negative")
    if not args.data_dir.exists():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if not args.task_config.exists():
        parser.error(f"Task config does not exist: {args.task_config}")

    image_paths = sorted(
        path for path in args.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        parser.error(f"No supported images found under: {args.data_dir}")

    records = [audit_image(path, args.data_dir) for path in image_paths]
    duplicate_summary = assign_duplicate_groups(records, args.near_duplicate_distance)
    task_config = read_task_config(args.task_config)

    manifest_path = args.output_dir / "dataset_manifest.csv"
    report_path = args.output_dir / "dataset_audit_report.json"
    write_manifest(records, manifest_path)
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            build_report(records, duplicate_summary, task_config, args.near_duplicate_distance),
            file_obj,
            indent=2,
        )

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote audit report: {report_path}")
    print(f"Images: {len(records)} | valid: {sum(record.valid for record in records)}")
    print(f"Class counts: {dict(sorted(Counter(record.current_label for record in records).items()))}")
    print(
        "Duplicate groups: "
        f"exact={duplicate_summary['exact_duplicate_groups']}, "
        f"near={duplicate_summary['near_duplicate_groups']}"
    )


if __name__ == "__main__":
    main()
