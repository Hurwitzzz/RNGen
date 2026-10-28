"""Independently audit a materialized full AugN/Mi20 NPZ cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    CORE_COMPONENTS,
    MICCHI_DOWNSAMPLE,
    SEQUENCE_LENGTH,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_keys() -> set[str]:
    keys = {
        "X_Bass19",
        "X_Chromagram19",
        "X_BassChromagram70",
        "valid_mask",
        "label_mask",
        "frame_offsets",
        "mi20_label_mask",
    }
    for component in COMPONENTS:
        keys.update(
            {
                f"y_{component}",
                f"label_mask_{component}",
                f"mi20_y_{component}",
                f"mi20_label_mask_{component}",
            }
        )
    return keys


def validate_cached_arrays(
    arrays: Mapping[str, np.ndarray], relative_path: str
) -> dict[str, Any]:
    actual_keys = set(arrays)
    expected_keys = _expected_keys()
    if actual_keys != expected_keys:
        raise ValueError(
            f"Cached key mismatch for {relative_path}: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    sequences = arrays["valid_mask"].shape[0]
    expected_full_shape = (sequences, SEQUENCE_LENGTH)
    expected_mi20_shape = (
        sequences,
        SEQUENCE_LENGTH // MICCHI_DOWNSAMPLE,
    )
    expected_input_shapes = {
        "X_Bass19": expected_full_shape + (19,),
        "X_Chromagram19": expected_full_shape + (19,),
        "X_BassChromagram70": expected_full_shape + (70,),
        "frame_offsets": expected_full_shape,
    }
    for name, expected_shape in expected_input_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"Invalid {name} shape for {relative_path}")
    if arrays["valid_mask"].shape != expected_full_shape:
        raise ValueError(f"Invalid valid_mask shape for {relative_path}")
    if arrays["label_mask"].shape != expected_full_shape:
        raise ValueError(f"Invalid label_mask shape for {relative_path}")
    if arrays["mi20_label_mask"].shape != expected_mi20_shape:
        raise ValueError(f"Invalid Mi20 mask shape for {relative_path}")
    if not np.isin(arrays["valid_mask"], (0, 1)).all():
        raise ValueError(f"Non-binary valid mask for {relative_path}")
    if not np.isin(arrays["label_mask"], (0, 1)).all():
        raise ValueError(f"Non-binary label mask for {relative_path}")
    if np.any(arrays["label_mask"] > arrays["valid_mask"]):
        raise ValueError(f"Label mask enters padding for {relative_path}")
    if not np.array_equal(
        arrays["mi20_label_mask"],
        arrays["label_mask"][:, ::MICCHI_DOWNSAMPLE],
    ):
        raise ValueError(f"Shared Mi20 mask misalignment for {relative_path}")

    flattened_valid = arrays["valid_mask"].reshape(-1)
    valid_frames = int(flattened_valid.sum())
    expected_valid = np.zeros_like(flattened_valid)
    expected_valid[:valid_frames] = 1
    if not np.array_equal(flattened_valid, expected_valid):
        raise ValueError(f"Non-contiguous padding mask for {relative_path}")

    head_frames = {}
    mi20_head_frames = {}
    for component in COMPONENTS:
        y = arrays[f"y_{component}"]
        mask = arrays[f"label_mask_{component}"]
        mi20_y = arrays[f"mi20_y_{component}"]
        mi20_mask = arrays[f"mi20_label_mask_{component}"]
        if y.shape != expected_full_shape + (1,):
            raise ValueError(f"Invalid {component} target shape for {relative_path}")
        if mask.shape != expected_full_shape:
            raise ValueError(f"Invalid {component} mask shape for {relative_path}")
        if mi20_y.shape != expected_mi20_shape + (1,):
            raise ValueError(
                f"Invalid Mi20 {component} target shape for {relative_path}"
            )
        if mi20_mask.shape != expected_mi20_shape:
            raise ValueError(f"Invalid Mi20 {component} mask shape for {relative_path}")
        if not np.isin(mask, (0, 1)).all() or not np.isin(mi20_mask, (0, 1)).all():
            raise ValueError(f"Non-binary {component} mask for {relative_path}")
        if np.any(mask > arrays["label_mask"]):
            raise ValueError(
                f"{component} mask exceeds shared mask for {relative_path}"
            )
        if component in CORE_COMPONENTS and not np.array_equal(
            mask, arrays["label_mask"]
        ):
            raise ValueError(f"Core {component} mask differs for {relative_path}")
        if not np.array_equal(mi20_y, y[:, ::MICCHI_DOWNSAMPLE]):
            raise ValueError(
                f"Mi20 {component} target misalignment for {relative_path}"
            )
        if not np.array_equal(mi20_mask, mask[:, ::MICCHI_DOWNSAMPLE]):
            raise ValueError(f"Mi20 {component} mask misalignment for {relative_path}")
        head_frames[component] = int(mask.sum())
        mi20_head_frames[component] = int(mi20_mask.sum())

    return {
        "files": 1,
        "frames": valid_frames,
        "sequences": sequences,
        "padded_frames": int(arrays["valid_mask"].size),
        "padding_frames": int(arrays["valid_mask"].size - valid_frames),
        "valid_frames": valid_frames,
        "label_frames": int(arrays["label_mask"].sum()),
        "mi20_label_frames": int(arrays["mi20_label_mask"].sum()),
        "head_label_frames": head_frames,
        "mi20_head_label_frames": mi20_head_frames,
    }


def _audit_split(
    cache_root: Path,
    split: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = cache_root / f"{split}_npz_paths.txt"
    paths = [Path(line) for line in manifest.read_text().splitlines() if line]
    records = report["records"][split]
    expected_paths = [Path(record["output_npz"]["path"]) for record in records]
    if paths != expected_paths:
        raise ValueError(f"{split} manifest order/content differs from report")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{split} NPZ manifest contains duplicates")

    totals: Counter[str] = Counter()
    head_frames: Counter[str] = Counter()
    mi20_head_frames: Counter[str] = Counter()
    source_hash_mismatches = []
    output_hash_mismatches = []
    for index, (path, record) in enumerate(zip(paths, records), start=1):
        try:
            path.resolve().relative_to((cache_root / split).resolve())
        except ValueError as exc:
            raise ValueError(f"{split} NPZ escapes cache root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        output_hash = _sha256(path)
        if output_hash != record["output_npz"]["sha256"]:
            output_hash_mismatches.append(str(path))
        source_path = Path(record["source"]["path"])
        source_hash = _sha256(source_path)
        if (
            not record["source"]["unchanged"]
            or source_hash != record["source"]["sha256_before"]
            or source_hash != record["source"]["sha256_after"]
        ):
            source_hash_mismatches.append(str(source_path))
        with np.load(path, allow_pickle=False) as loaded:
            item = validate_cached_arrays(
                {key: loaded[key] for key in loaded.files},
                str(path),
            )
        for key in (
            "files",
            "frames",
            "sequences",
            "padded_frames",
            "padding_frames",
            "valid_frames",
            "label_frames",
            "mi20_label_frames",
        ):
            totals[key] += item[key]
        head_frames.update(item["head_label_frames"])
        mi20_head_frames.update(item["mi20_head_label_frames"])
        if index % 250 == 0:
            print(f"audited_{split}={index}/{len(paths)}", flush=True)

    if source_hash_mismatches or output_hash_mismatches:
        raise ValueError(
            f"{split} hash mismatches: sources={source_hash_mismatches[:5]}, "
            f"outputs={output_hash_mismatches[:5]}"
        )
    if dict(totals) != report["totals"][split]:
        raise ValueError(f"{split} totals differ from materialization report")
    if dict(head_frames) != report["head_label_frames"][split]:
        raise ValueError(f"{split} head-frame totals differ")
    if dict(mi20_head_frames) != report["mi20_head_label_frames"][split]:
        raise ValueError(f"{split} Mi20 head-frame totals differ")
    return {
        "status": "pass",
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": _sha256(manifest),
        },
        "totals": dict(totals),
        "head_label_frames": dict(head_frames),
        "mi20_head_label_frames": dict(mi20_head_frames),
        "source_hashes_rechecked": len(paths),
        "output_hashes_rechecked": len(paths),
        "npz_schema_mask_downsampling_checks": len(paths),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def audit_full_cache(
    cache_root: Path, slurm_job_id: Optional[str] = None
) -> dict[str, Any]:
    report_path = cache_root / "materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError("Materialization report status is not pass")
    if report["structure"] != {
        "base_train_pieces": 355,
        "transposition_multiplier": 12,
        "train_files": 4260,
        "validation_files": 44,
        "test_files_not_materialized": 47,
        "split_disjointness_verified": True,
    }:
        raise ValueError(f"Unexpected cache structure: {report['structure']}")
    splits = {
        split: _audit_split(cache_root, split, report)
        for split in ("train", "validation")
    }
    extra_npzs = sorted(
        str(path)
        for path in cache_root.rglob("*.npz")
        if not any(
            _is_relative_to(path, cache_root / split)
            for split in ("train", "validation")
        )
    )
    if extra_npzs:
        raise ValueError(f"Unexpected NPZs outside train/validation: {extra_npzs[:5]}")
    return {
        "status": "pass",
        "cache_root": str(cache_root.resolve()),
        "materialization_report": {
            "path": str(report_path.resolve()),
            "sha256": _sha256(report_path),
        },
        "materializer_elapsed_seconds": report["elapsed_seconds"],
        "slurm_job_id": slurm_job_id,
        "source_policy": (
            "all 4,304 converted source hashes independently rechecked against "
            "both before/after hashes; source CSVs remain read-only"
        ),
        "test_split_materialized": False,
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_full_cache(args.cache_root, args.slurm_job_id)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
