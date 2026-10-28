"""Classwise K/D/Q/I error tables for exact-onset ChordGNN evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def _label(family: str, values: Mapping[str, str]) -> str:
    if family == "K":
        return values["local_key"]
    if family == "D":
        return f"{values['primary_degree']}/{values['secondary_degree']}"
    if family == "Q":
        return values["quality"]
    if family == "I":
        return values["inversion"]
    raise ValueError(f"Unsupported classwise family {family}")


def classwise_exact_onsets(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = {}
    for family in ("K", "D", "Q", "I"):
        support: Counter[str] = Counter()
        correct: Counter[str] = Counter()
        predicted: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            reference_label = _label(family, record["reference"])
            predicted_label = _label(family, record["prediction"])
            support[reference_label] += 1
            correct[reference_label] += int(reference_label == predicted_label)
            predicted[reference_label][predicted_label] += 1
        output[family] = {
            "support": int(sum(support.values())),
            "rows": [
                {
                    "label": label,
                    "support": int(support[label]),
                    "correct": int(correct[label]),
                    "accuracy": correct[label] / support[label],
                    "predicted_counts": dict(sorted(predicted[label].items())),
                }
                for label in sorted(support)
            ],
        }
    return output
