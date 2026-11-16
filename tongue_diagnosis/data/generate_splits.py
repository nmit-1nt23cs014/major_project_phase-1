"""Generate deterministic duplicate-aware train/validation/test assignments.

Input rows are never copied, deleted, or relabelled.  Every member of a
duplicate/near-duplicate group is assigned to exactly one split.  This protects
against detectable image-level leakage, but cannot guarantee patient-level
independence because the local dataset has no patient identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "manifests" / "split_summary.json"
SPLIT_NAMES = ("train", "validation", "test")


def parse_fraction(value: str) -> float:
    fraction = float(value)
    if not 0.0 < fraction < 1.0:
        raise argparse.ArgumentTypeError("Fractions must be strictly between zero and one.")
    return fraction


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    required = {"sample_id", "relative_path", "current_label", "valid", "split_group_id", "sha256"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    valid_rows = [row for row in rows if row["valid"].strip().lower() == "true"]
    if not valid_rows:
        raise ValueError("Manifest contains no valid images.")
    return valid_rows


def integer_targets(total: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    raw = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def make_assignments(
    rows: list[dict[str, str]],
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Allocate label-pure image groups while staying close to desired counts."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)

    labels_by_group = {
        group_id: {row["current_label"] for row in members}
        for group_id, members in groups.items()
    }
    mixed_groups = {group_id: labels for group_id, labels in labels_by_group.items() if len(labels) != 1}
    if mixed_groups:
        example = next(iter(mixed_groups.items()))
        raise ValueError(
            "A duplicate-aware split group contains conflicting labels. "
            f"Resolve label conflict before splitting; example={example}."
        )

    groups_by_label: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_id, members in groups.items():
        groups_by_label[next(iter(labels_by_group[group_id]))].append((group_id, members))

    assignments: dict[str, str] = {}
    target_by_label: dict[str, dict[str, int]] = {}
    count_by_label: dict[str, dict[str, int]] = {}
    for label in sorted(groups_by_label):
        label_groups = groups_by_label[label]
        target_by_label[label] = integer_targets(sum(len(members) for _, members in label_groups), fractions)
        count_by_label[label] = {name: 0 for name in SPLIT_NAMES}

        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(label_groups)
        # Larger duplicate groups go first; the seeded shuffle breaks equal-size ties.
        label_groups.sort(key=lambda item: len(item[1]), reverse=True)
        for group_id, members in label_groups:
            group_size = len(members)
            split = max(
                SPLIT_NAMES,
                key=lambda name: (
                    target_by_label[label][name] - count_by_label[label][name],
                    -SPLIT_NAMES.index(name),
                ),
            )
            assignments[group_id] = split
            count_by_label[label][split] += group_size

    return assignments, {"targets": target_by_label, "actual": count_by_label}


def check_leakage(rows: list[dict[str, str]], assignments: dict[str, str]) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    seen_paths: set[str] = set()
    duplicate_paths: list[str] = []
    for row in rows:
        split = assignments[row["split_group_id"]]
        group_splits[row["split_group_id"]].add(split)
        hash_splits[row["sha256"]].add(split)
        if row["relative_path"] in seen_paths:
            duplicate_paths.append(row["relative_path"])
        seen_paths.add(row["relative_path"])

    leaked_groups = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
    leaked_hashes = sorted(image_hash for image_hash, splits in hash_splits.items() if len(splits) > 1)
    return {
        "each_path_assigned_once": not duplicate_paths and len(seen_paths) == len(rows),
        "duplicate_paths": duplicate_paths,
        "split_groups_crossing_splits": leaked_groups,
        "exact_hashes_crossing_splits": leaked_hashes,
        "leakage_check_passed": not leaked_groups and not leaked_hashes and not duplicate_paths,
    }


def write_split_manifest(rows: list[dict[str, str]], assignments: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["split"]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["relative_path"]):
            writer.writerow({**row, "split": assignments[row["split_group_id"]]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=parse_fraction, default=0.70)
    parser.add_argument("--validation-fraction", type=parse_fraction, default=0.15)
    parser.add_argument("--test-fraction", type=parse_fraction, default=0.15)
    args = parser.parse_args()

    fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
    if abs(sum(fractions) - 1.0) > 1e-9:
        parser.error("Train, validation, and test fractions must sum to exactly 1.0.")
    if not args.manifest.exists():
        parser.error(f"Manifest does not exist: {args.manifest}. Run dataset_audit.py first.")

    rows = load_manifest(args.manifest)
    assignments, allocation = make_assignments(rows, fractions, args.seed)
    leakage = check_leakage(rows, assignments)
    if not leakage["leakage_check_passed"]:
        raise RuntimeError(f"Leakage checks failed: {leakage}")

    write_split_manifest(rows, assignments, args.output)
    split_counts = Counter(assignments[row["split_group_id"]] for row in rows)
    split_class_counts: dict[str, dict[str, int]] = {
        split: dict(sorted(Counter(
            row["current_label"] for row in rows if assignments[row["split_group_id"]] == split
        ).items()))
        for split in SPLIT_NAMES
    }
    summary = {
        "schema_version": 1,
        "source_manifest": args.manifest.as_posix(),
        "seed": args.seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions)),
        "sample_counts": dict(sorted(split_counts.items())),
        "class_counts_by_split": split_class_counts,
        "allocation": allocation,
        "leakage_checks": leakage,
        "patient_independence_limitation": "Patient IDs are unavailable. This duplicate-aware image-level split does not guarantee patient-level independence.",
        "test_set_policy": "Keep test samples isolated from training, checkpoint selection, hyperparameter tuning, threshold selection, and preprocessing fitting."
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print(f"Wrote split manifest: {args.output}")
    print(f"Wrote split summary: {args.summary}")
    print(f"Split counts: {dict(sorted(split_counts.items()))}")
    print(f"Class counts by split: {split_class_counts}")
    print("Leakage checks: PASSED")
    print("Patient-level independence: NOT ESTABLISHED (patient IDs unavailable)")


if __name__ == "__main__":
    main()
