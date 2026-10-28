"""Evaluate controlled AugN/Mi20 checkpoints at exact reference note onsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    AUGMENTED_SIX_PREFIX_BY_QUALITY,
    COMPONENTS,
    FIXED_OFFSET,
    MICCHI_DOWNSAMPLE,
    NONE_LABEL,
    PieceTensors,
    canonicalize_special_rn,
    convert_piece,
    decode_augmented_six_rn,
    encode_and_segment,
    read_split_paths,
)
from rngen.baselines.augnet_mi20.model import (
    OFFICIAL_HEAD_ORDER,
    OFFICIAL_MONITORED_HEADS,
)
from rngen.baselines.augnet_mi20.trainer import (
    COMPONENT_TO_HEAD,
    MaskedTrainingConfig,
    build_controlled_baseline,
    build_masked_training_model,
    prepare_baseline_arrays,
    prepare_output_directory,
)

ROMAN_BY_DEGREE = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
}
TRIAD_FIGURES = {"0": "", "1": "6", "2": "64"}
SEVENTH_FIGURES = {"0": "7", "1": "65", "2": "43", "3": "2"}
QUALITY_DECODING = {
    "major triad": (True, "", TRIAD_FIGURES),
    "minor triad": (False, "", TRIAD_FIGURES),
    "diminished triad": (False, "o", TRIAD_FIGURES),
    "augmented triad": (True, "+", TRIAD_FIGURES),
    "minor seventh chord": (False, "", SEVENTH_FIGURES),
    "major seventh chord": (True, "M", SEVENTH_FIGURES),
    "dominant seventh chord": (True, "", SEVENTH_FIGURES),
    "incomplete dominant-seventh chord": (True, "", SEVENTH_FIGURES),
    "diminished seventh chord": (False, "o", SEVENTH_FIGURES),
    "half-diminished seventh chord": (False, "ø", SEVENTH_FIGURES),
}
CORE_COMPONENTS = (
    "local_key",
    "primary_degree",
    "secondary_degree",
    "inversion",
    "quality",
    "chord_root",
)
CLASSWISE_FAMILIES = ("K", "D", "Q", "I")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _load_vocabularies(path: Path) -> Dict[str, Sequence[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "vocabularies" not in payload:
        raise ValueError(f"Ontology audit lacks vocabularies: {path}")
    return payload["vocabularies"]


def _degree_to_roman(degree: str, *, uppercase: bool) -> str:
    match = re.fullmatch(r"([#-]*)([1-7])", str(degree))
    if match is None:
        raise ValueError(f"Unsupported altered numeric degree: {degree!r}")
    alteration, number = match.groups()
    numeral = ROMAN_BY_DEGREE[number]
    if not uppercase:
        numeral = numeral.lower()
    return alteration.replace("-", "b") + numeral


def canonical_rn_from_components(labels: Mapping[str, str]) -> str:
    """Decode the representable canonical RNA figure from baseline components."""
    quality = str(labels.get("quality", ""))
    inversion = str(labels.get("inversion", ""))
    if quality in AUGMENTED_SIX_PREFIX_BY_QUALITY:
        primary = decode_augmented_six_rn(quality, inversion)
    else:
        try:
            uppercase, quality_marker, figures = QUALITY_DECODING[quality]
            figure = figures[inversion]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported quality/inversion combination: {quality!r}/{inversion!r}"
            ) from exc
        primary = (
            _degree_to_roman(
                str(labels.get("primary_degree", "")),
                uppercase=uppercase,
            )
            + quality_marker
            + figure
        )

    secondary_degree = str(labels.get("secondary_degree", ""))
    if secondary_degree == NONE_LABEL:
        return canonicalize_special_rn(primary)
    tonicized_key = str(labels.get("tonicized_key", ""))
    if not tonicized_key or tonicized_key == NONE_LABEL:
        raise ValueError(
            "A secondary degree requires a predicted tonicized key to recover mode"
        )
    secondary = _degree_to_roman(
        secondary_degree,
        uppercase=tonicized_key[0].isupper(),
    )
    return canonicalize_special_rn(f"{primary}/{secondary}")


def audit_ground_truth_decoder(piece: PieceTensors) -> Dict[str, Any]:
    """Audit raw versus representable canonical roundtrip for one piece."""
    mismatch_examples = []
    decode_failures = []
    for event in piece.harmony_events:
        try:
            decoded = canonical_rn_from_components(event.labels)
        except ValueError as exc:
            decode_failures.append(
                {
                    "offset": event.onset,
                    "raw_canonical_rn": event.canonical_rn,
                    "error": str(exc),
                }
            )
            continue
        if decoded != event.canonical_rn:
            mismatch_examples.append(
                {
                    "offset": event.onset,
                    "raw_canonical_rn": event.canonical_rn,
                    "representable_canonical_rn": decoded,
                }
            )
    return {
        "relative_path": piece.relative_path,
        "harmony_events": len(piece.harmony_events),
        "decode_failures": len(decode_failures),
        "raw_canonical_roundtrip_mismatches": len(mismatch_examples),
        "representational_collapses": mismatch_examples,
        "decode_failure_examples": decode_failures,
    }


def _reference_event_indices(piece: PieceTensors) -> Tuple[np.ndarray, np.ndarray]:
    event_offsets = [event.onset for event in piece.harmony_events]
    onsets = []
    indices = []
    for onset in piece.reference_note_onsets:
        event_index = bisect_right(event_offsets, float(onset)) - 1
        if event_index >= 0:
            onsets.append(float(onset))
            indices.append(event_index)
    return np.asarray(onsets, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def _input_frame(onset: float, n_frames: int) -> int:
    return min(n_frames - 1, max(0, int(math.ceil(onset / FIXED_OFFSET - 1e-9))))


def _prediction_labels_at_onset(
    predicted_indices_by_head: Mapping[str, np.ndarray],
    vocabularies: Mapping[str, Sequence[str]],
    architecture: str,
    input_frame: int,
) -> Dict[str, str]:
    output_frame = (
        input_frame
        if architecture == "AugmentedNet"
        else input_frame // MICCHI_DOWNSAMPLE
    )
    labels = {}
    for component, head in COMPONENT_TO_HEAD.items():
        indices = np.asarray(predicted_indices_by_head[head]).reshape(-1)
        if output_frame >= len(indices):
            raise ValueError(
                f"Prediction timeline for {head} ends before frame {output_frame}"
            )
        index = int(indices[output_frame])
        classes = vocabularies[component]
        if index < 0 or index >= len(classes):
            raise ValueError(f"Invalid predicted class {index} for {head}")
        labels[component] = str(classes[index])
    return labels


def _metric_values(
    predicted: Mapping[str, str],
    reference: Mapping[str, str],
) -> Dict[str, bool]:
    values = {
        "K": predicted["local_key"] == reference["local_key"],
        "D1": predicted["primary_degree"] == reference["primary_degree"],
        "D2": predicted["secondary_degree"] == reference["secondary_degree"],
        "Q": predicted["quality"] == reference["quality"],
        "I": predicted["inversion"] == reference["inversion"],
        "Root": predicted["chord_root"] == reference["chord_root"],
    }
    values["D"] = values["D1"] and values["D2"]
    values["RNA"] = all(values[name] for name in ("K", "D1", "D2", "Q", "I", "Root"))
    return values


def _class_label(family: str, labels: Mapping[str, str]) -> str:
    if family == "K":
        return labels["local_key"]
    if family == "D":
        return f"{labels['primary_degree']}/{labels['secondary_degree']}"
    if family == "Q":
        return labels["quality"]
    if family == "I":
        return labels["inversion"]
    raise ValueError(f"Unsupported classwise family: {family}")


def _classwise_rows(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = {}
    for family in CLASSWISE_FAMILIES:
        support: Counter[str] = Counter()
        correct: Counter[str] = Counter()
        confusions: Dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            reference = _class_label(family, record["reference_labels"])
            predicted = _class_label(family, record["predicted_labels"])
            support[reference] += 1
            confusions[reference][predicted] += 1
            correct[reference] += int(reference == predicted)
        result[family] = {
            "total_onsets": int(sum(support.values())),
            "rows": [
                {
                    "label": label,
                    "support": int(support[label]),
                    "correct": int(correct[label]),
                    "accuracy": correct[label] / support[label],
                    "predicted_counts": dict(sorted(confusions[label].items())),
                }
                for label in sorted(support)
            ],
        }
    return result


def score_piece_onsets(
    piece: PieceTensors,
    predicted_indices_by_head: Mapping[str, np.ndarray],
    vocabularies: Mapping[str, Sequence[str]],
    architecture: str,
    *,
    max_failure_cases: int = 32,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Score one piece at every exact annotated reference note onset."""
    if architecture not in {"AugmentedNet", "Micchi2020"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    onsets, event_indices = _reference_event_indices(piece)
    records: list[Dict[str, Any]] = []
    failures = []
    input_frames = []
    output_frames = []
    decoder_failures = 0
    raw_reference_collapses = 0
    counts = {
        name: {"correct": 0, "total": 0}
        for name in (
            "RNA",
            "K",
            "D",
            "D1",
            "D2",
            "Q",
            "I",
            "Root",
            "canonical_RNA",
            "canonical_figure",
            "raw_canonical_figure",
        )
    }
    auxiliary_counts = {
        component: {"correct": 0, "total": 0}
        for component in COMPONENTS
        if component not in CORE_COMPONENTS
    }

    for onset, event_index in zip(onsets, event_indices):
        event = piece.harmony_events[int(event_index)]
        input_frame = _input_frame(float(onset), len(piece.frame_offsets))
        output_frame = (
            input_frame
            if architecture == "AugmentedNet"
            else input_frame // MICCHI_DOWNSAMPLE
        )
        predicted = _prediction_labels_at_onset(
            predicted_indices_by_head,
            vocabularies,
            architecture,
            input_frame,
        )
        reference = dict(event.labels)
        reference["harmonic_rhythm"] = (
            "True" if math.isclose(float(onset), event.onset, abs_tol=1e-7) else "False"
        )
        metric_values = _metric_values(predicted, reference)
        for name, correct in metric_values.items():
            counts[name]["total"] += 1
            counts[name]["correct"] += int(correct)

        reference_figure = canonical_rn_from_components(reference)
        raw_reference_collapses += int(reference_figure != event.canonical_rn)
        predicted_figure = None
        decoder_error = None
        try:
            predicted_figure = canonical_rn_from_components(predicted)
        except ValueError as exc:
            decoder_failures += 1
            decoder_error = str(exc)
        canonical_figure_correct = predicted_figure == reference_figure
        raw_figure_correct = predicted_figure == event.canonical_rn
        canonical_rna_correct = metric_values["K"] and canonical_figure_correct
        for name, correct in (
            ("canonical_figure", canonical_figure_correct),
            ("raw_canonical_figure", raw_figure_correct),
            ("canonical_RNA", canonical_rna_correct),
        ):
            counts[name]["total"] += 1
            counts[name]["correct"] += int(correct)

        for component in auxiliary_counts:
            reference_label = reference[component]
            if reference_label in vocabularies[component]:
                auxiliary_counts[component]["total"] += 1
                auxiliary_counts[component]["correct"] += int(
                    predicted[component] == reference_label
                )

        record = {
            "relative_path": piece.relative_path,
            "onset_quarter_notes": float(onset),
            "input_frame": input_frame,
            "output_frame": output_frame,
            "reference_event_offset": event.onset,
            "reference_labels": reference,
            "predicted_labels": predicted,
            "raw_reference_rn": event.gt_rnbert,
            "canonical_reference_rn": event.canonical_rn,
            "representable_reference_rn": reference_figure,
            "predicted_canonical_rn": predicted_figure,
            "decoder_error": decoder_error,
            "correct": metric_values,
            "canonical_figure_correct": canonical_figure_correct,
        }
        records.append(record)
        input_frames.append(input_frame)
        output_frames.append(output_frame)
        if not metric_values["RNA"] and len(failures) < max_failure_cases:
            failures.append(record)

    metrics = {
        f"{name}_acc": value["correct"] / value["total"] if value["total"] else 0.0
        for name, value in counts.items()
    }
    auxiliary_metrics = {
        f"{component}_acc": value["correct"] / value["total"] if value["total"] else 0.0
        for component, value in auxiliary_counts.items()
    }
    report = {
        "relative_path": piece.relative_path,
        "architecture": architecture,
        "metrics": metrics,
        "counts": counts,
        "auxiliary_metrics": auxiliary_metrics,
        "auxiliary_counts": auxiliary_counts,
        "sampling": {
            "exact_reference_note_onsets_total": int(len(piece.reference_note_onsets)),
            "annotated_reference_note_onsets": len(records),
            "unannotated_leading_note_onsets": int(
                len(piece.reference_note_onsets) - len(records)
            ),
            "unique_input_frames_sampled": len(set(input_frames)),
            "input_frame_collisions": len(input_frames) - len(set(input_frames)),
            "unique_output_frames_sampled": len(set(output_frames)),
            "output_frame_collisions": len(output_frames) - len(set(output_frames)),
            "off_grid_reference_note_onsets": int(
                sum(
                    not math.isclose(
                        onset / FIXED_OFFSET,
                        round(onset / FIXED_OFFSET),
                        abs_tol=1e-7,
                    )
                    for onset in onsets
                )
            ),
            "mi20_sampling": (
                "zero-order hold from output frame floor(input_frame / 4)"
                if architecture == "Micchi2020"
                else None
            ),
        },
        "predicted_decoder_failures": decoder_failures,
        "raw_reference_representational_collapses": raw_reference_collapses,
        "classwise": _classwise_rows(records),
        "failure_cases": failures,
        "ground_truth_decoder_audit": audit_ground_truth_decoder(piece),
    }
    return report, records


def native_fixed_frame_metrics(
    prepared: Any,
    predictions_by_head: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute native fixed-frame diagnostics; never use these as primary metrics."""
    correct_by_head = {}
    report = {}
    for head in OFFICIAL_HEAD_ORDER:
        predicted = np.argmax(predictions_by_head[head], axis=-1).reshape(-1)
        reference = prepared.targets[head][..., 0].reshape(-1)
        mask = prepared.masks[head].reshape(-1).astype(bool)
        correct = predicted == reference
        correct_by_head[head] = correct
        report[head] = {
            "correct": int((correct & mask).sum()),
            "total": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
        }
    core_mask = np.logical_and.reduce(
        [
            prepared.masks[head].reshape(-1).astype(bool)
            for head in OFFICIAL_MONITORED_HEADS
        ]
    )
    degree = correct_by_head["PrimaryDegree22"] & correct_by_head["SecondaryDegree22"]
    rna = degree.copy()
    for head in ("LocalKey35", "ChordQuality15", "ChordRoot35", "Inversion4"):
        rna &= correct_by_head[head]
    for name, correct in (("Degree", degree), ("RomanNumeral", rna)):
        report[name] = {
            "correct": int((correct & core_mask).sum()),
            "total": int(core_mask.sum()),
            "accuracy": float(correct[core_mask].mean()) if core_mask.any() else 0.0,
        }
    return report


def _merge_piece_reports(piece_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    auxiliary: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for report in piece_reports:
        for name, value in report["counts"].items():
            counts[name]["correct"] += int(value["correct"])
            counts[name]["total"] += int(value["total"])
        for name, value in report["auxiliary_counts"].items():
            auxiliary[name]["correct"] += int(value["correct"])
            auxiliary[name]["total"] += int(value["total"])
    native: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for report in piece_reports:
        for name, value in report["native_fixed_frame_diagnostics"].items():
            native[name]["correct"] += int(value["correct"])
            native[name]["total"] += int(value["total"])
    return {
        "pieces": len(piece_reports),
        "metrics": {
            f"{name}_acc": value["correct"] / value["total"] if value["total"] else 0.0
            for name, value in counts.items()
        },
        "counts": dict(counts),
        "auxiliary_metrics": {
            f"{name}_acc": value["correct"] / value["total"] if value["total"] else 0.0
            for name, value in auxiliary.items()
        },
        "auxiliary_counts": dict(auxiliary),
        "annotated_reference_note_onsets": sum(
            int(report["sampling"]["annotated_reference_note_onsets"])
            for report in piece_reports
        ),
        "input_frame_collisions": sum(
            int(report["sampling"]["input_frame_collisions"])
            for report in piece_reports
        ),
        "output_frame_collisions": sum(
            int(report["sampling"]["output_frame_collisions"])
            for report in piece_reports
        ),
        "predicted_decoder_failures": sum(
            int(report["predicted_decoder_failures"]) for report in piece_reports
        ),
        "raw_reference_representational_collapses": sum(
            int(report["raw_reference_representational_collapses"])
            for report in piece_reports
        ),
        "native_fixed_frame_diagnostics": {
            name: {
                **value,
                "accuracy": (
                    value["correct"] / value["total"] if value["total"] else 0.0
                ),
            }
            for name, value in native.items()
        },
    }


def _resolve_piece_paths(
    dataset_root: Path,
    pieces: Sequence[str],
    manifests: Sequence[Path],
) -> list[Tuple[str, Path]]:
    relative_paths = list(pieces)
    for manifest in manifests:
        relative_paths.extend(read_split_paths(manifest))
    if not relative_paths:
        raise ValueError("At least one --piece or --manifest is required")
    result = []
    seen = set()
    root = dataset_root.resolve()
    for relative_path in relative_paths:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Piece escapes dataset root: {relative_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((relative_path, path))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("AugmentedNet", "Micchi2020"), required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--ground-truth-audit-only",
        action="store_true",
        help="Audit canonical GT reconstruction without loading TensorFlow.",
    )
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_SCORES_DIR)
    parser.add_argument("--piece", action="append", default=[])
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--ontology-audit",
        type=Path,
        help="Ontology JSON required unless --ground-truth-audit-only is set.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-failure-cases", type=int, default=32)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if not args.ground_truth_audit_only and args.ontology_audit is None:
        parser.error("--ontology-audit is required unless --ground-truth-audit-only is set")
    return args


def main() -> None:
    args = parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_failure_cases < 0:
        raise ValueError("max-failure-cases cannot be negative")
    if not args.ground_truth_audit_only:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required unless --ground-truth-audit-only"
            )
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    pieces = _resolve_piece_paths(args.dataset_root, args.piece, args.manifest)
    configured_source_roots = MaskedTrainingConfig(
        architecture=args.architecture,
        output_dir=args.output_dir,
    ).read_only_source_roots
    prepare_output_directory(
        args.output_dir,
        configured_source_roots + (args.dataset_root,),
    )
    if args.ground_truth_audit_only:
        piece_audits = []
        total_note_onsets = 0
        total_grid_collisions = 0
        for relative_path, csv_path in pieces:
            piece = convert_piece(csv_path, relative_path=relative_path)
            audit = audit_ground_truth_decoder(piece)
            for collapse in audit["representational_collapses"]:
                collapse["relative_path"] = relative_path
            for failure in audit["decode_failure_examples"]:
                failure["relative_path"] = relative_path
            piece_audits.append(audit)
            total_note_onsets += int(piece.diagnostics["reference_note_onsets"])
            total_grid_collisions += int(
                piece.diagnostics["reference_note_onset_grid_collisions"]
            )
        report = {
            "audit": "controlled corpus canonical decoder and exact-onset contract",
            "dataset_root": str(args.dataset_root.resolve()),
            "pieces": len(piece_audits),
            "harmony_events": sum(item["harmony_events"] for item in piece_audits),
            "decode_failures": sum(item["decode_failures"] for item in piece_audits),
            "raw_canonical_roundtrip_mismatches": sum(
                item["raw_canonical_roundtrip_mismatches"] for item in piece_audits
            ),
            "representational_collapses": [
                collapse
                for item in piece_audits
                for collapse in item["representational_collapses"]
            ],
            "decode_failure_examples": [
                failure
                for item in piece_audits
                for failure in item["decode_failure_examples"]
            ],
            "exact_reference_note_onsets": total_note_onsets,
            "distinct_onsets_sharing_a_fixed_grid_frame": total_grid_collisions,
            "manifests": [_file_record(path) for path in args.manifest],
            "raw_roundtrip_policy": (
                "Report, never guess, RN distinctions beyond the official "
                "single SecondaryDegree22 level"
            ),
        }
        _write_json(args.output_dir / "ground_truth_decoder_audit.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    evaluation_started = time.perf_counter()
    vocabularies = _load_vocabularies(args.ontology_audit)
    assert args.checkpoint is not None
    model = None
    piece_reports = []
    all_records = []
    source_records = []
    for relative_path, csv_path in pieces:
        piece = convert_piece(csv_path, relative_path=relative_path)
        arrays = encode_and_segment(piece, vocabularies)
        prepared = prepare_baseline_arrays(arrays, vocabularies, args.architecture)
        if model is None:
            base_model = build_controlled_baseline(prepared, vocabularies)
            model = build_masked_training_model(base_model)
            model.load_weights(str(args.checkpoint))
        raw_predictions = model.predict(
            prepared.inputs,
            batch_size=args.batch_size,
            verbose=0,
        )
        if not isinstance(raw_predictions, list):
            raw_predictions = [raw_predictions]
        predictions_by_head = dict(zip(model.output_names, raw_predictions))
        predicted_indices = {
            head: np.argmax(predictions_by_head[head], axis=-1)
            for head in OFFICIAL_HEAD_ORDER
        }
        report, records = score_piece_onsets(
            piece,
            predicted_indices,
            vocabularies,
            args.architecture,
            max_failure_cases=args.max_failure_cases,
        )
        report["native_fixed_frame_diagnostics"] = native_fixed_frame_metrics(
            prepared,
            predictions_by_head,
        )
        piece_reports.append(report)
        all_records.extend(records)
        source_records.append(_file_record(csv_path))

    aggregate = _merge_piece_reports(piece_reports)
    aggregate["architecture"] = args.architecture
    aggregate["primary_metric_protocol"] = (
        "exact annotated reference note onsets; RNA is the conjunction of "
        "LocalKey35, PrimaryDegree22, SecondaryDegree22, ChordQuality15, "
        "ChordRoot35, and Inversion4"
    )
    aggregate["mi20_sampling"] = (
        "zero-order hold from floor(input frame / 4)"
        if args.architecture == "Micchi2020"
        else None
    )
    aggregate["evaluation_elapsed_seconds"] = time.perf_counter() - evaluation_started
    config = {
        "architecture": args.architecture,
        "checkpoint": _file_record(args.checkpoint),
        "ontology_audit": _file_record(args.ontology_audit),
        "dataset_root": str(args.dataset_root.resolve()),
        "source_csvs": source_records,
        "manifests": [_file_record(path) for path in args.manifest],
        "batch_size": args.batch_size,
        "output_dir": str(args.output_dir.resolve()),
        "source_policy": "read-only conversion in memory; no source CSV writes",
    }
    _write_json(args.output_dir / "evaluation_config.json", config)
    _write_json(args.output_dir / "onset_metrics.json", aggregate)
    _write_json(args.output_dir / "per_piece_metrics.json", piece_reports)
    _write_json(
        args.output_dir / "classwise_metrics.json",
        _classwise_rows(all_records),
    )
    _write_json(
        args.output_dir / "failure_cases.json",
        [case for report in piece_reports for case in report["failure_cases"]],
    )
    with (args.output_dir / "onset_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
