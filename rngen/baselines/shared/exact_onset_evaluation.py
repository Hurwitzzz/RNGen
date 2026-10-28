"""Model-independent exact-reference-onset evaluation for RNA baselines."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rngen.baselines.augnet_mi20.evaluation.evaluate_augmentednet_baseline import (
    canonical_rn_from_components,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    CORE_COMPONENTS,
    PieceTensors,
)


@dataclass(frozen=True)
class PredictionTimeline:
    """Decoded model outputs on a strictly increasing native timeline."""

    offsets: np.ndarray
    labels: Mapping[str, np.ndarray]
    source_resolution: str

    def validate(self) -> None:
        if self.offsets.ndim != 1 or not len(self.offsets):
            raise ValueError("Prediction timeline offsets must be non-empty 1-D")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Prediction timeline offsets must be strictly increasing")
        for component in COMPONENTS:
            if component not in self.labels:
                raise ValueError(f"Prediction timeline is missing {component}")
            if np.asarray(self.labels[component]).shape != self.offsets.shape:
                raise ValueError(
                    f"Prediction timeline head {component} has wrong shape"
                )

    def index_at_or_before(self, offset: float) -> int | None:
        index = int(np.searchsorted(self.offsets, offset, side="right") - 1)
        return index if index >= 0 else None


def _reference_event_index(piece: PieceTensors, onset: float) -> int | None:
    offsets = [event.onset for event in piece.harmony_events]
    index = int(np.searchsorted(offsets, onset, side="right") - 1)
    return index if index >= 0 else None


def _component_correct(
    predicted: Mapping[str, str], reference: Mapping[str, str]
) -> dict[str, bool]:
    correct = {
        component: predicted.get(component) == reference.get(component)
        for component in COMPONENTS
    }
    correct["degree"] = correct["primary_degree"] and correct["secondary_degree"]
    correct["rna"] = all(correct[component] for component in CORE_COMPONENTS)
    return correct


def score_exact_onsets(
    piece: PieceTensors,
    timeline: PredictionTimeline,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every distinct annotated source-note onset using zero-order hold."""

    timeline.validate()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    selected_native_indices: list[int] = []
    for onset in piece.reference_note_onsets:
        event_index = _reference_event_index(piece, float(onset))
        if event_index is None:
            counts["unannotated_leading_note_onsets"] += 1
            continue
        native_index = timeline.index_at_or_before(float(onset))
        reference = dict(piece.harmony_events[event_index].labels)
        if native_index is None:
            predicted = {component: "<NO_PREDICTION>" for component in COMPONENTS}
            prediction_offset: float | None = None
            counts["missing_native_predictions"] += 1
        else:
            selected_native_indices.append(native_index)
            predicted = {
                component: str(timeline.labels[component][native_index])
                for component in COMPONENTS
            }
            prediction_offset = float(timeline.offsets[native_index])
        correctness = _component_correct(predicted, reference)
        prediction_decode_error = None
        if native_index is None:
            prediction_rn = "<NO_PREDICTION>"
        else:
            try:
                prediction_rn = canonical_rn_from_components(predicted)
            except ValueError as exc:
                prediction_rn = "<INVALID_COMPONENT_COMBINATION>"
                prediction_decode_error = str(exc)
                counts["prediction_decode_failures"] += 1
        counts["annotated_reference_note_onsets"] += 1
        counts["rna_correct"] += int(correctness["rna"])
        counts["key_correct"] += int(correctness["local_key"])
        counts["degree_correct"] += int(correctness["degree"])
        for component in COMPONENTS:
            counts[f"{component}_correct"] += int(correctness[component])
        records.append(
            {
                "relative_path": piece.relative_path,
                "onset_quarter_notes": float(onset),
                "prediction_offset": prediction_offset,
                "prediction_native_index": native_index,
                "reference_event_offset": float(
                    piece.harmony_events[event_index].onset
                ),
                "reference": reference,
                "prediction": predicted,
                "correct": correctness,
                "reference_rn": canonical_rn_from_components(reference),
                "prediction_rn": prediction_rn,
                "prediction_decode_error": prediction_decode_error,
            }
        )
    total = int(counts["annotated_reference_note_onsets"])
    if not total:
        raise ValueError(f"No annotated reference note onsets in {piece.relative_path}")
    selected = np.asarray(selected_native_indices, dtype=np.int64)
    collisions = int(len(selected) - len(np.unique(selected)))
    report = {
        "relative_path": piece.relative_path,
        "source_resolution": timeline.source_resolution,
        "metrics": {
            "RNA_acc": counts["rna_correct"] / total,
            "K_acc": counts["key_correct"] / total,
            "D_acc": counts["degree_correct"] / total,
            "Q_acc": counts["quality_correct"] / total,
            "I_acc": counts["inversion_correct"] / total,
            **{
                f"{component}_acc": counts[f"{component}_correct"] / total
                for component in COMPONENTS
            },
        },
        "counts": {
            "annotated_reference_note_onsets": total,
            "unannotated_leading_note_onsets": int(
                counts["unannotated_leading_note_onsets"]
            ),
            "missing_native_predictions": int(counts["missing_native_predictions"]),
            "native_timeline_positions": int(len(timeline.offsets)),
            "selected_native_positions": int(len(np.unique(selected))),
            "zero_order_hold_collisions": collisions,
            "rna_correct": int(counts["rna_correct"]),
            "key_correct": int(counts["key_correct"]),
            "degree_correct": int(counts["degree_correct"]),
            "quality_correct": int(counts["quality_correct"]),
            "inversion_correct": int(counts["inversion_correct"]),
            "prediction_decode_failures": int(counts["prediction_decode_failures"]),
            **{
                f"{component}_correct": int(counts[f"{component}_correct"])
                for component in COMPONENTS
            },
        },
    }
    return report, records


def merge_piece_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("Cannot merge an empty exact-onset evaluation")
    total = sum(
        int(report["counts"]["annotated_reference_note_onsets"]) for report in reports
    )
    count_keys = (
        "rna_correct",
        "key_correct",
        "degree_correct",
        "quality_correct",
        "inversion_correct",
    )
    counts = {
        key: sum(int(report["counts"][key]) for report in reports) for key in count_keys
    }
    component_correct = {
        component: sum(
            int(report["counts"][f"{component}_correct"]) for report in reports
        )
        for component in COMPONENTS
    }
    return {
        "pieces": len(reports),
        "annotated_reference_note_onsets": total,
        "metrics": {
            "RNA_acc": counts["rna_correct"] / total,
            "K_acc": counts["key_correct"] / total,
            "D_acc": counts["degree_correct"] / total,
            "Q_acc": counts["quality_correct"] / total,
            "I_acc": counts["inversion_correct"] / total,
            **{
                f"{component}_acc": component_correct[component] / total
                for component in COMPONENTS
            },
        },
        "counts": {
            **counts,
            **{
                f"{component}_correct": value
                for component, value in component_correct.items()
            },
            "unannotated_leading_note_onsets": sum(
                int(report["counts"]["unannotated_leading_note_onsets"])
                for report in reports
            ),
            "prediction_decode_failures": sum(
                int(report["counts"]["prediction_decode_failures"])
                for report in reports
            ),
            "missing_native_predictions": sum(
                int(report["counts"]["missing_native_predictions"])
                for report in reports
            ),
            "native_timeline_positions": sum(
                int(report["counts"]["native_timeline_positions"]) for report in reports
            ),
            "zero_order_hold_collisions": sum(
                int(report["counts"]["zero_order_hold_collisions"])
                for report in reports
            ),
        },
    }
