#!/usr/bin/env python3
"""Summarize RNA class-wise performance together with annotation support counts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rngen.training.metrics_rna_components import (
    canonicalize_degree,
    canonicalize_inversion,
    canonicalize_local_key,
    canonicalize_primary_function,
    canonicalize_quality,
    canonicalize_secondary_function,
    serialize_classwise_label,
)
from rngen.training.metrics_rna_shared import (
    _expand_harmony_assignments,
    _get_sorted_harmonies,
)

FAMILY_EXTRACTORS = {
    "K": canonicalize_local_key,
    "P": canonicalize_primary_function,
    "S": canonicalize_secondary_function,
    "Q": canonicalize_quality,
    "D": canonicalize_degree,
    "I": canonicalize_inversion,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize RNA class-wise performance together with optional train-set "
            "support counts computed from annotation files."
        )
    )
    parser.add_argument(
        "--classwise-report-file",
        type=Path,
        required=True,
        help="Path to rna_posthoc_classwise_metrics.json.",
    )
    parser.add_argument(
        "--train-annos-file",
        type=Path,
        default=None,
        help=(
            "Optional train annotation JSON. If provided, onset-level train support "
            "counts will be computed with the same semantics as the scorer."
        ),
    )
    parser.add_argument(
        "--compound-ratio",
        type=int,
        default=9,
        help="Compound ratio used by the encoder and annotation events.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save merged JSON and per-family CSV reports.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _label_key(family: str, label: Any) -> str:
    return serialize_classwise_label(family, label)["label"]


def compute_annotation_family_supports(
    annotation_payload: Mapping[str, Any],
    compound_ratio: int,
    families: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    family_supports: Dict[str, Dict[str, int]] = {family: {} for family in families}

    for sample_id, sample in annotation_payload.items():
        if sample_id == "template":
            continue

        output = sample.get("output")
        if not isinstance(output, Mapping):
            continue

        events = sample.get("events")
        last_target_bar = sample.get("metadata", {}).get("last_target_bar")
        if not isinstance(events, str) or last_target_bar is None:
            continue

        sorted_refs = _get_sorted_harmonies(output)
        _, ref_onset_map = _expand_harmony_assignments(
            sorted_refs,
            events,
            compound_ratio,
            last_target_bar,
        )

        for ref_harmony in ref_onset_map.values():
            for family in families:
                extractor = FAMILY_EXTRACTORS[family]
                label = extractor(ref_harmony)
                if label is None:
                    continue
                label_key = _label_key(family, label)
                family_supports[family][label_key] = (
                    family_supports[family].get(label_key, 0) + 1
                )

    return family_supports


def merge_classwise_with_train_support(
    classwise_payload: Mapping[str, Any],
    train_supports: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    classwise_report = classwise_payload["classwise"]
    merged: Dict[str, Any] = {
        "task": classwise_payload.get("task"),
        "prediction_file": classwise_payload.get("prediction_file"),
        "annos_file": classwise_payload.get("annos_file"),
        "compound_ratio": classwise_payload.get("compound_ratio"),
        "train_annos_file": None,
        "families": {},
    }

    for family, report in classwise_report.items():
        rows = []
        total_test_onsets = report["total_onsets"]
        total_train_onsets = (
            sum(train_supports.get(family, {}).values()) if train_supports else None
        )

        for row in report["rows"]:
            merged_row = dict(row)
            if train_supports is not None:
                train_count = train_supports.get(family, {}).get(row["label"], 0)
                merged_row["train_support_count"] = train_count
                merged_row["train_support_ratio"] = (
                    train_count / total_train_onsets if total_train_onsets else 0.0
                )
                merged_row["support_count_delta"] = (
                    train_count - merged_row["support_count"]
                )
            rows.append(merged_row)

        merged["families"][family] = {
            "total_test_onsets": total_test_onsets,
            "total_train_onsets": total_train_onsets,
            "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"],
            "rows": rows,
        }

    return merged


def save_distribution_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "rna_classwise_distribution_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    for family, family_report in report["families"].items():
        csv_path = output_dir / f"rna_classwise_distribution_{family.lower()}.csv"
        rows = family_report["rows"]
        if not rows:
            continue

        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    classwise_payload = load_json(args.classwise_report_file)

    train_supports = None
    if args.train_annos_file is not None:
        train_annos = load_json(args.train_annos_file)
        train_supports = compute_annotation_family_supports(
            train_annos,
            compound_ratio=args.compound_ratio,
            families=classwise_payload["classwise"].keys(),
        )

    merged_report = merge_classwise_with_train_support(
        classwise_payload,
        train_supports=train_supports,
    )
    if args.train_annos_file is not None:
        merged_report["train_annos_file"] = str(args.train_annos_file)

    if args.output_dir is not None:
        save_distribution_report(args.output_dir, merged_report)

    for family, family_report in merged_report["families"].items():
        print(
            f"{family}: test_onsets={family_report['total_test_onsets']}, "
            f"train_onsets={family_report['total_train_onsets']}, "
            f"macro_f1={family_report['macro_f1']:.4f}, "
            f"weighted_f1={family_report['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
