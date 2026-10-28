"""Materialize the audited full AugN/Mi20 train and validation cache."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.augnet_mi20.data.augmentednet_cache import (
    _file_record,
    _load_vocabularies,
    _materialize_piece,
    _output_npz_path,
    _prepare_output_directory,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    _read_csv,
    _validate_transposition_pair,
    manifest_sha256,
    read_split_paths,
    validate_augmented_manifest,
)

EXPECTED_AUGMENTATION_SETTINGS = {
    "aug_by_key": True,
    "aug_by_key_n_keys": 12,
    "aug_within_range": False,
    "aug_rhythms": False,
}
REFERENCE_TOTAL_KEYS = (
    "files",
    "frames",
    "sequences",
    "padded_frames",
    "padding_frames",
    "valid_frames",
    "label_frames",
    "mi20_label_frames",
)
FAILURE_DIAGNOSTICS = (
    "unresolved_special_events",
    "unmapped_quality_events",
    "augmented_six_roundtrip_mismatches",
    "empty_score_frames_after_fill",
    "unsupported_spelling_rows",
    "spelling_pitch_class_mismatches",
)


def validate_reference_totals(
    actual: Mapping[str, int], reference: Mapping[str, int]
) -> None:
    mismatches = {
        key: {"expected": int(reference[key]), "actual": int(actual.get(key, -1))}
        for key in REFERENCE_TOTAL_KEYS
        if int(actual.get(key, -1)) != int(reference[key])
    }
    if mismatches:
        raise ValueError(f"Materialized train totals differ from audit: {mismatches}")


def _accumulate_record(
    totals: Counter[str],
    head_frames: Counter[str],
    mi20_head_frames: Counter[str],
    diagnostics: Counter[str],
    record: Mapping[str, Any],
) -> None:
    totals["files"] += 1
    for key in REFERENCE_TOTAL_KEYS[1:]:
        totals[key] += int(record[key])
    for component in COMPONENTS:
        head_frames[component] += int(record["head_label_frames"][component])
        mi20_head_frames[component] += int(record["mi20_head_label_frames"][component])
    for key, value in record["diagnostics"].items():
        if isinstance(value, int):
            diagnostics[key] += value


def _concise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": record["relative_path"],
        "source": record["source"],
        "output_npz": record["output_npz"],
        "frames": record["frames"],
        "sequences": record["sequences"],
        "padding_frames": record["padding_frames"],
        "label_frames": record["label_frames"],
        "mi20_label_frames": record["mi20_label_frames"],
        "empty_score_frames_before_fill": record["diagnostics"][
            "empty_score_frames_before_fill"
        ],
        "enharmonic_normalizations": record["enharmonic_normalizations"],
    }


def materialize_full_cache(
    *,
    base_root: Path,
    augmented_root: Path,
    base_split_dir: Path,
    augmented_split_dir: Path,
    augmentation_settings: Path,
    ontology_audit: Path,
    reference_conversion_audit: Path,
    output_dir: Path,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Persist the controlled 4,260-path train and 44-piece validation cache."""
    started = time.monotonic()
    source_roots = (base_root, augmented_root)
    _prepare_output_directory(output_dir, source_roots)
    vocabularies = _load_vocabularies(ontology_audit)

    base_train_manifest = base_split_dir / "train_paths.txt"
    base_validation_manifest = base_split_dir / "valid_paths.txt"
    base_test_manifest = base_split_dir / "test_paths.txt"
    augmented_train_manifest = augmented_split_dir / "train_paths.txt"
    base_train_paths = read_split_paths(base_train_manifest)
    validation_paths = read_split_paths(base_validation_manifest)
    test_paths = read_split_paths(base_test_manifest)
    groups = validate_augmented_manifest(
        read_split_paths(augmented_train_manifest), base_train_paths
    )
    if set(base_train_paths) & (set(validation_paths) | set(test_paths)):
        raise ValueError("Train overlaps validation or test")
    if set(validation_paths) & set(test_paths):
        raise ValueError("Validation overlaps test")

    settings = json.loads(augmentation_settings.read_text(encoding="utf-8"))
    settings_mismatches = {
        key: {"expected": expected, "actual": settings.get(key)}
        for key, expected in EXPECTED_AUGMENTATION_SETTINGS.items()
        if settings.get(key) != expected
    }
    if settings_mismatches:
        raise ValueError(f"Unexpected augmentation settings: {settings_mismatches}")

    split_totals = {split: Counter() for split in ("train", "validation")}
    split_head_frames = {split: Counter() for split in split_totals}
    split_mi20_head_frames = {split: Counter() for split in split_totals}
    split_diagnostics = {split: Counter() for split in split_totals}
    split_subtypes = {split: Counter() for split in split_totals}
    normalizations: Counter[str] = Counter()
    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    npz_paths: dict[str, list[str]] = {"train": [], "validation": []}

    completed = 0
    for base_relative_path in base_train_paths:
        base_df = _read_csv(base_root / base_relative_path)
        shifts = []
        for augmentation_index, relative_path in groups[base_relative_path]:
            source_path = augmented_root / relative_path
            shift = _validate_transposition_pair(
                base_df,
                _read_csv(source_path),
                base_path=base_relative_path,
                augmented_path=relative_path,
            )
            shifts.append(shift)
            output_path = _output_npz_path(output_dir, "train", relative_path)
            record, piece_normalizations = _materialize_piece(
                source_path,
                relative_path,
                output_path,
                vocabularies,
                canonicalize_extreme_input_spellings=True,
            )
            _accumulate_record(
                split_totals["train"],
                split_head_frames["train"],
                split_mi20_head_frames["train"],
                split_diagnostics["train"],
                record,
            )
            split_subtypes["train"].update(
                record["diagnostics"]["augmented_six_subtype_counts"]
            )
            normalizations.update(piece_normalizations)
            concise = _concise_record(record)
            concise.update(
                {
                    "base_relative_path": base_relative_path,
                    "augmentation_index": augmentation_index,
                    "midi_shift": shift,
                }
            )
            records["train"].append(concise)
            npz_paths["train"].append(str(output_path.resolve()))
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"train={completed}/4260 "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
        if len(set(shifts)) != 12 or 0 not in shifts:
            raise ValueError(
                f"Expected 12 distinct shifts including zero for "
                f"{base_relative_path}, got {sorted(shifts)}"
            )

    for index, relative_path in enumerate(validation_paths, start=1):
        source_path = base_root / relative_path
        output_path = _output_npz_path(output_dir, "validation", relative_path)
        record, piece_normalizations = _materialize_piece(
            source_path,
            relative_path,
            output_path,
            vocabularies,
            canonicalize_extreme_input_spellings=False,
        )
        if piece_normalizations:
            raise ValueError(
                f"Untransposed validation unexpectedly needs normalization: "
                f"{relative_path}: {dict(piece_normalizations)}"
            )
        _accumulate_record(
            split_totals["validation"],
            split_head_frames["validation"],
            split_mi20_head_frames["validation"],
            split_diagnostics["validation"],
            record,
        )
        split_subtypes["validation"].update(
            record["diagnostics"]["augmented_six_subtype_counts"]
        )
        records["validation"].append(_concise_record(record))
        npz_paths["validation"].append(str(output_path.resolve()))
        if progress_every and index % progress_every == 0:
            print(
                f"validation={index}/{len(validation_paths)} "
                f"elapsed_seconds={time.monotonic() - started:.1f}",
                flush=True,
            )

    reference_audit = json.loads(reference_conversion_audit.read_text(encoding="utf-8"))
    validate_reference_totals(split_totals["train"], reference_audit["totals"])
    nonzero_failures = {
        split: {
            key: int(split_diagnostics[split][key])
            for key in FAILURE_DIAGNOSTICS
            if split_diagnostics[split][key]
        }
        for split in split_totals
    }
    nonzero_failures = {
        split: failures for split, failures in nonzero_failures.items() if failures
    }
    if nonzero_failures:
        raise ValueError(f"Full cache conversion failures: {nonzero_failures}")

    manifest_records = {}
    for split, paths in npz_paths.items():
        manifest = output_dir / f"{split}_npz_paths.txt"
        manifest.write_text("\n".join(paths) + "\n", encoding="utf-8")
        manifest_records[split] = _file_record(manifest)

    report = {
        "status": "pass",
        "purpose": "controlled full AugN/Mi20 train and validation cache",
        "source_policy": (
            "CSV roots opened read-only; every converted source SHA-256 checked "
            "before and after; only derived NPZ, manifest, and report files written"
        ),
        "dataset": {
            "base_root": str(base_root.resolve()),
            "augmented_root": str(augmented_root.resolve()),
            "base_train_manifest": {
                "path": str(base_train_manifest.resolve()),
                "sha256": manifest_sha256(base_train_manifest),
            },
            "base_validation_manifest": {
                "path": str(base_validation_manifest.resolve()),
                "sha256": manifest_sha256(base_validation_manifest),
            },
            "base_test_manifest": {
                "path": str(base_test_manifest.resolve()),
                "sha256": manifest_sha256(base_test_manifest),
            },
            "augmented_train_manifest": {
                "path": str(augmented_train_manifest.resolve()),
                "sha256": manifest_sha256(augmented_train_manifest),
            },
            "augmentation_settings": _file_record(augmentation_settings),
            "augmentation_settings_verified": EXPECTED_AUGMENTATION_SETTINGS,
            "ontology_audit": _file_record(ontology_audit),
            "reference_conversion_audit": _file_record(reference_conversion_audit),
        },
        "structure": {
            "base_train_pieces": len(base_train_paths),
            "transposition_multiplier": 12,
            "train_files": len(npz_paths["train"]),
            "validation_files": len(npz_paths["validation"]),
            "test_files_not_materialized": len(test_paths),
            "split_disjointness_verified": True,
        },
        "totals": {split: dict(values) for split, values in split_totals.items()},
        "head_label_frames": {
            split: dict(values) for split, values in split_head_frames.items()
        },
        "mi20_head_label_frames": {
            split: dict(values) for split, values in split_mi20_head_frames.items()
        },
        "diagnostics": {
            split: dict(values) for split, values in split_diagnostics.items()
        },
        "augmented_six_subtype_counts": {
            split: dict(sorted(values.items()))
            for split, values in split_subtypes.items()
        },
        "enharmonic_normalizations": dict(sorted(normalizations.items())),
        "npz_manifests": manifest_records,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = output_dir / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=PROCESSED_SCORES_DIR,
    )
    parser.add_argument(
        "--augmented-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base-split-dir",
        type=Path,
        default=Path("data_splits/p_dcml"),
    )
    parser.add_argument(
        "--augmented-split-dir",
        type=Path,
        default=Path("data_splits/p_dcml/augmented_splits"),
    )
    parser.add_argument(
        "--augmentation-settings",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ontology-audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference-conversion-audit",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize_full_cache(
        base_root=args.base_root,
        augmented_root=args.augmented_root,
        base_split_dir=args.base_split_dir,
        augmented_split_dir=args.augmented_split_dir,
        augmentation_settings=args.augmentation_settings,
        ontology_audit=args.ontology_audit,
        reference_conversion_audit=args.reference_conversion_audit,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "structure": report["structure"],
                "totals": report["totals"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
