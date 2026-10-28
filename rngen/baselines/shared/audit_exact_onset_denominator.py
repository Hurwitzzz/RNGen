"""Build and audit the canonical exact-reference-onset denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    convert_piece,
    manifest_sha256,
    read_split_paths,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--report must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError("--report must use non-empty NAME=PATH")
    return name, Path(raw_path)


def build_denominator(
    *,
    dataset_root: Path,
    manifest: Path,
    split: str,
    output_dir: Path,
    reports: Sequence[tuple[str, Path]] = (),
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Denominator split must be validation or test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Denominator output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    piece_counts = {}
    for relative_path in read_split_paths(manifest):
        piece = convert_piece(dataset_root / relative_path, relative_path=relative_path)
        event_onsets = [event.onset for event in piece.harmony_events]
        rational_onsets = set()
        for onset in piece.reference_note_onsets:
            if bisect_right(event_onsets, float(onset)) - 1 < 0:
                continue
            rational_onsets.add(Fraction(float(onset)).limit_denominator(256))
        ordered = sorted(rational_onsets)
        piece_counts[relative_path] = len(ordered)
        entries.extend(
            {
                "relative_path": relative_path,
                "onset_numerator": onset.numerator,
                "onset_denominator": onset.denominator,
            }
            for onset in ordered
        )
    denominator_path = output_dir / "canonical_denominator.jsonl"
    with denominator_path.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    denominator_hash = _sha256(denominator_path)
    comparisons = {}
    for name, path in reports:
        report = json.loads(path.read_text())
        observed = int(report["annotated_reference_note_onsets"])
        if observed != len(entries):
            raise ValueError(
                f"{name} denominator {observed} differs from canonical "
                f"{len(entries)}"
            )
        comparisons[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "annotated_reference_note_onsets": observed,
            "matches": True,
        }
    payload = {
        "audit": "canonical exact-reference-onset denominator",
        "status": "passed",
        "split": split,
        "dataset_root": str(dataset_root.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256(manifest),
        "pieces": len(piece_counts),
        "annotated_reference_note_onsets": len(entries),
        "canonical_denominator": str(denominator_path.resolve()),
        "canonical_denominator_sha256": denominator_hash,
        "maximum_fraction_denominator": max(
            (entry["onset_denominator"] for entry in entries), default=0
        ),
        "piece_counts": piece_counts,
        "reports": comparisons,
        "provenance": full_provenance(require_git=False),
    }
    _write_json(output_dir / "denominator_audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROCESSED_SCORES_DIR,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_denominator(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        reports=[_parse_report(value) for value in args.report],
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pieces": payload["pieces"],
                "onsets": payload["annotated_reference_note_onsets"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
