"""Diagnostic-only native and paper-grid measures for ChordGNN."""

from __future__ import annotations

from typing import Any

from rngen.baselines.shared.exact_onset_evaluation import (
    PredictionTimeline,
    _component_correct,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    PieceTensors,
)


def score_paper_fixed_grid(
    piece: PieceTensors, timeline: PredictionTimeline
) -> dict[str, Any]:
    """Score six-head CSR on the existing 1/32-note P.DCML grid.

    This is retained only as a paper-style diagnostic. The primary comparison
    remains the exact distinct annotated source-note-onset denominator.
    """

    timeline.validate()
    counts = {
        "support": 0,
        "rna_correct": 0,
        "key_correct": 0,
        "degree_correct": 0,
        "quality_correct": 0,
        "inversion_correct": 0,
        "missing_predictions": 0,
    }
    for frame, onset in enumerate(piece.frame_offsets):
        if not bool(piece.label_mask[frame]):
            continue
        counts["support"] += 1
        native_index = timeline.index_at_or_before(float(onset))
        reference = {
            component: str(piece.label_strings[component][frame])
            for component in COMPONENTS
        }
        if native_index is None:
            counts["missing_predictions"] += 1
            predicted = {component: "<NO_PREDICTION>" for component in COMPONENTS}
        else:
            predicted = {
                component: str(timeline.labels[component][native_index])
                for component in COMPONENTS
            }
        correct = _component_correct(predicted, reference)
        counts["rna_correct"] += int(correct["rna"])
        counts["key_correct"] += int(correct["local_key"])
        counts["degree_correct"] += int(correct["degree"])
        counts["quality_correct"] += int(correct["quality"])
        counts["inversion_correct"] += int(correct["inversion"])
    support = counts["support"]
    return {
        "definition": (
            "six-head conjunction sampled on the existing 1/32-quarter-note "
            "P.DCML fixed grid; diagnostic only"
        ),
        "counts": counts,
        "metrics": {
            "CSR": counts["rna_correct"] / support if support else 0.0,
            "K_acc": counts["key_correct"] / support if support else 0.0,
            "D_acc": counts["degree_correct"] / support if support else 0.0,
            "Q_acc": counts["quality_correct"] / support if support else 0.0,
            "I_acc": counts["inversion_correct"] / support if support else 0.0,
        },
    }
