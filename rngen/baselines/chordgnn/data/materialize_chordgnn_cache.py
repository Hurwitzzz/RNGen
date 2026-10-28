"""Materialize the controlled train/validation ChordGNN graph cache."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    assert_output_outside_source_roots,
    manifest_sha256,
    read_split_paths,
    validate_augmented_manifest,
)
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import (
    EDGE_TYPES,
    FEATURE_NAMES,
    UPSTREAM_COMMIT,
    convert_piece,
    file_sha256,
    ontology_payload,
)


EXPECTED_TRAIN_GRAPHS = 4260
EXPECTED_VALIDATION_GRAPHS = 44


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _prepare_output(output_dir: Path, source_roots: Sequence[Path]) -> None:
    assert_output_outside_source_roots(output_dir, source_roots)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Cache path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Cache directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _cache_path(output_dir: Path, split: str, relative_path: str) -> Path:
    relative = Path(relative_path)
    return output_dir / split / relative.with_suffix(".npz")


def _materialize_split(
    *,
    split: str,
    paths: Sequence[str],
    dataset_root: Path,
    ontology_root: Path,
    ontology_split_dir: Path,
    output_dir: Path,
    vocabularies: Mapping[str, Sequence[str]],
    progress_every: int,
) -> tuple[list[Path], dict[str, Any]]:
    outputs: list[Path] = []
    edge_counts: Counter[str] = Counter()
    head_supervision: Counter[str] = Counter()
    label_normalizations: Counter[str] = Counter()
    auxiliary_oov: dict[str, Counter[str]] = {
        component: Counter() for component in COMPONENTS
    }
    total_nodes = 0
    total_onsets = 0
    total_edges = 0
    source_records: list[dict[str, Any]] = []
    started = time.time()
    for index, relative_path in enumerate(paths, start=1):
        source_path = dataset_root / relative_path
        before = file_sha256(source_path)
        example = convert_piece(
            source_path,
            dataset_root=ontology_root,
            split_dir=ontology_split_dir,
            relative_path=relative_path,
            vocabularies=vocabularies,
        )
        output_path = _cache_path(output_dir, split, relative_path)
        example.save_npz(output_path)
        after = file_sha256(source_path)
        if before != after:
            raise RuntimeError(f"Source CSV changed during conversion: {source_path}")
        outputs.append(output_path.resolve())
        total_nodes += int(example.diagnostics["nodes"])
        total_onsets += int(example.diagnostics["graph_onsets"])
        total_edges += int(example.diagnostics["edges"])
        edge_counts.update(example.diagnostics["edge_type_counts"])
        head_supervision.update(example.diagnostics["head_supervision"])
        label_normalizations.update(example.diagnostics["label_normalizations"])
        for component, values in example.diagnostics["auxiliary_oov_masks"].items():
            auxiliary_oov[component].update(values)
        source_records.append(
            {
                "relative_path": relative_path,
                "source_path": str(source_path.resolve()),
                "source_sha256": before,
                "cache_path": str(output_path.resolve()),
                "cache_sha256": file_sha256(output_path),
                "nodes": int(example.diagnostics["nodes"]),
                "onsets": int(example.diagnostics["graph_onsets"]),
                "edges": int(example.diagnostics["edges"]),
            }
        )
        if progress_every and (index % progress_every == 0 or index == len(paths)):
            elapsed = time.time() - started
            print(
                f"[{split}] {index}/{len(paths)} graphs, "
                f"{total_nodes} nodes, {elapsed:.1f}s",
                flush=True,
            )
    return outputs, {
        "graphs": len(outputs),
        "nodes": total_nodes,
        "graph_onsets": total_onsets,
        "forward_edges": total_edges,
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "label_normalizations": dict(sorted(label_normalizations.items())),
        "auxiliary_oov_masks": {
            component: {
                "total": int(sum(auxiliary_oov[component].values())),
                "labels": dict(sorted(auxiliary_oov[component].items())),
            }
            for component in COMPONENTS
        },
        "head_supervision": {
            component: int(head_supervision[component]) for component in COMPONENTS
        },
        "source_records": source_records,
        "elapsed_seconds": time.time() - started,
    }


def materialize(
    *,
    base_root: Path,
    augmented_root: Path,
    split_dir: Path,
    augmented_split_dir: Path,
    output_dir: Path,
    progress_every: int = 100,
) -> dict[str, Any]:
    _prepare_output(output_dir, (base_root, augmented_root))
    base_train = read_split_paths(split_dir / "train_paths.txt")
    augmented_train = read_split_paths(augmented_split_dir / "train_paths.txt")
    validation = read_split_paths(split_dir / "valid_paths.txt")
    test = read_split_paths(split_dir / "test_paths.txt")
    validate_augmented_manifest(augmented_train, base_train)
    if set(base_train) & set(validation) or set(base_train) & set(test):
        raise ValueError("Base train/validation/test manifests are not disjoint")
    if set(validation) & set(test):
        raise ValueError("Validation/test manifests are not disjoint")
    if len(augmented_train) != EXPECTED_TRAIN_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_GRAPHS} augmented train files, "
            f"got {len(augmented_train)}"
        )
    if len(validation) != EXPECTED_VALIDATION_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_GRAPHS} validation files, "
            f"got {len(validation)}"
        )

    vocabularies, ontology = ontology_payload(base_root, split_dir)
    ontology_path = output_dir / "ontology.json"
    _write_json(ontology_path, ontology)
    train_outputs, train_report = _materialize_split(
        split="train",
        paths=augmented_train,
        dataset_root=augmented_root,
        ontology_root=base_root,
        ontology_split_dir=split_dir,
        output_dir=output_dir,
        vocabularies=vocabularies,
        progress_every=progress_every,
    )
    validation_outputs, validation_report = _materialize_split(
        split="validation",
        paths=validation,
        dataset_root=base_root,
        ontology_root=base_root,
        ontology_split_dir=split_dir,
        output_dir=output_dir,
        vocabularies=vocabularies,
        progress_every=progress_every,
    )
    (output_dir / "train_npz_paths.txt").write_text(
        "".join(f"{path}\n" for path in train_outputs)
    )
    (output_dir / "validation_npz_paths.txt").write_text(
        "".join(f"{path}\n" for path in validation_outputs)
    )
    report = {
        "audit": "controlled ChordGNN train/validation cache",
        "status": "completed",
        "schema_version": "chordgnn-pdcml-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "provenance": full_provenance(require_git=False),
        "source_roots": {
            "base": str(base_root.resolve()),
            "augmented": str(augmented_root.resolve()),
        },
        "manifests": {
            "base_train": {
                "path": str((split_dir / "train_paths.txt").resolve()),
                "sha256": manifest_sha256(split_dir / "train_paths.txt"),
                "files": len(base_train),
            },
            "augmented_train": {
                "path": str((augmented_split_dir / "train_paths.txt").resolve()),
                "sha256": manifest_sha256(augmented_split_dir / "train_paths.txt"),
                "files": len(augmented_train),
            },
            "validation": {
                "path": str((split_dir / "valid_paths.txt").resolve()),
                "sha256": manifest_sha256(split_dir / "valid_paths.txt"),
                "files": len(validation),
            },
            "test_locked": {
                "path": str((split_dir / "test_paths.txt").resolve()),
                "sha256": manifest_sha256(split_dir / "test_paths.txt"),
                "files": len(test),
                "cached": False,
            },
        },
        "feature_names": list(FEATURE_NAMES),
        "edge_types": EDGE_TYPES,
        "ontology_path": str(ontology_path.resolve()),
        "ontology_sha256": file_sha256(ontology_path),
        "splits": {
            "train": train_report,
            "validation": validation_report,
        },
        "graph_manifests": {
            "train": {
                "path": str((output_dir / "train_npz_paths.txt").resolve()),
                "sha256": file_sha256(output_dir / "train_npz_paths.txt"),
            },
            "validation": {
                "path": str((output_dir / "validation_npz_paths.txt").resolve()),
                "sha256": file_sha256(output_dir / "validation_npz_paths.txt"),
            },
        },
        "gates": {
            "exact_train_graphs": len(train_outputs) == EXPECTED_TRAIN_GRAPHS,
            "exact_validation_graphs": len(validation_outputs)
            == EXPECTED_VALIDATION_GRAPHS,
            "test_not_cached": True,
            "source_csvs_unchanged": True,
            "split_disjointness_verified": True,
        },
    }
    _write_json(output_dir / "materialization_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--split-dir",
        type=Path,
        default=Path("data_splits/p_dcml"),
    )
    parser.add_argument(
        "--augmented-split-dir",
        type=Path,
        default=Path("data_splits/p_dcml/augmented_splits"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize(
        base_root=args.base_root,
        augmented_root=args.augmented_root,
        split_dir=args.split_dir,
        augmented_split_dir=args.augmented_split_dir,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_graphs": report["splits"]["train"]["graphs"],
                "validation_graphs": report["splits"]["validation"]["graphs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
