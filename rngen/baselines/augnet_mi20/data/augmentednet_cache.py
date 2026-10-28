"""Shared helpers for materializing AugmentedNet/Micchi2020 NPZ caches."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    _validate_encoded_arrays,
    assert_output_outside_source_roots,
    canonicalize_piece_enharmonic_labels,
    convert_piece,
    encode_and_segment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_vocabularies(path: Path) -> dict[str, Sequence[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocabularies = payload.get("vocabularies")
    if not isinstance(vocabularies, dict):
        raise ValueError(f"Ontology audit lacks vocabularies: {path}")
    missing = sorted(set(COMPONENTS) - set(vocabularies))
    if missing:
        raise ValueError(f"Ontology audit lacks components: {missing}")
    return vocabularies


def _prepare_output_directory(output_dir: Path, source_roots: Sequence[Path]) -> None:
    assert_output_outside_source_roots(output_dir, source_roots)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Cache output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _output_npz_path(output_dir: Path, split: str, relative_path: str) -> Path:
    return output_dir / split / Path(relative_path).with_suffix(".npz")


def _special_events(piece: Any) -> list[dict[str, Any]]:
    result = []
    for event in piece.harmony_events:
        primary = event.canonical_rn.split("/", 1)[0]
        if not primary.startswith(("Fr", "Ger", "It")):
            continue
        result.append(
            {
                "onset": event.onset,
                "raw_rn": event.gt_rnbert,
                "canonical_rn": event.canonical_rn,
                "primary_degree": event.labels["primary_degree"],
                "quality": event.labels["quality"],
                "inversion": event.labels["inversion"],
            }
        )
    return result


def _materialize_piece(
    source_path: Path,
    relative_path: str,
    output_path: Path,
    vocabularies: Mapping[str, Sequence[str]],
    *,
    canonicalize_extreme_input_spellings: bool,
) -> tuple[dict[str, Any], Counter[str]]:
    source_before = _sha256(source_path)
    piece = convert_piece(
        source_path,
        relative_path=relative_path,
        canonicalize_extreme_input_spellings=canonicalize_extreme_input_spellings,
    )
    normalizations = canonicalize_piece_enharmonic_labels(piece, vocabularies)
    arrays = encode_and_segment(piece, vocabularies)
    encoded = _validate_encoded_arrays(piece, arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    source_after = _sha256(source_path)
    if source_before != source_after:
        raise RuntimeError(f"Source CSV changed during conversion: {source_path}")
    return (
        {
            "relative_path": relative_path,
            "source": {
                "path": str(source_path.resolve()),
                "sha256_before": source_before,
                "sha256_after": source_after,
                "unchanged": True,
            },
            "output_npz": _file_record(output_path),
            **encoded,
            "diagnostics": piece.diagnostics,
            "enharmonic_normalizations": dict(sorted(normalizations.items())),
            "special_events": _special_events(piece),
        },
        normalizations,
    )
