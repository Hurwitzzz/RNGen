#!/usr/bin/env python3
"""Compute key or RNA metrics from saved prediction files.

This script is intended for manual post-hoc analysis of saved predictions.
It is not part of the training runtime.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rngen.training.metrics_key_generation import (
    compute_key_detection_metrics,
    compute_note_and_onset_level_accuracy_from_events as compute_key_acc,
)
from rngen.training.metrics_rna_shared import (
    compute_rna_note_and_onset_level_accuracy as compute_rna_acc,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute key or RNA metrics from saved prediction JSON files."
    )
    parser.add_argument(
        "--task",
        choices=["key_sft", "rna_sft"],
        required=True,
        help="Task type used to select the metric family.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        required=True,
        help="Path to a saved predictions JSON file.",
    )
    parser.add_argument(
        "--annos-file",
        type=Path,
        required=True,
        help=(
            "Path to the annotation JSON file. Each sample must provide "
            "metadata.last_target_bar as an integer."
        ),
    )
    parser.add_argument(
        "--compound-ratio",
        type=int,
        default=9,
        help="Compound ratio used by the encoder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory used to save post-hoc metric artifacts. For RNA "
            "this includes both summary metrics and class-wise reports."
        ),
    )
    return parser.parse_args()


def build_metadata_list(test_annos: dict, num_samples: int) -> list:
    metadata_list = []
    for i in range(num_samples):
        sample = test_annos[str(i)]
        metadata = sample.get("metadata")
        last_target_bar = (
            metadata.get("last_target_bar") if isinstance(metadata, dict) else None
        )
        if not isinstance(last_target_bar, int) or isinstance(last_target_bar, bool):
            raise ValueError(
                f"Annotation sample {i} requires an integer metadata.last_target_bar."
            )
        metadata_list.append(
            {
                "events": sample["events"],
                "note_count": sample["note_count"],
                "sample_id": str(i),
                "last_target_bar": last_target_bar,
            }
        )
    return metadata_list


def load_prediction_payload(prediction_file: Path) -> Dict[str, Any]:
    with prediction_file.open("r") as f:
        return json.load(f)


def load_annotation_payload(annos_file: Path) -> Dict[str, Any]:
    with annos_file.open("r") as f:
        return json.load(f)


def compute_posthoc_metrics(
    task: str,
    prediction_payload: Dict[str, Any],
    annotation_payload: Dict[str, Any],
    compound_ratio: int,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    pred_list = prediction_payload["predictions"]
    ref_list = prediction_payload["references"]
    assert len(pred_list) == len(ref_list) == len(annotation_payload) - 1

    metadata_list = build_metadata_list(annotation_payload, len(pred_list))

    if task == "key_sft":
        metrics = compute_key_acc(
            predictions=pred_list,
            references=ref_list,
            metadata_list=metadata_list,
            compound_ratio=compound_ratio,
        )
        metrics.update(compute_key_detection_metrics(pred_list, ref_list))
        return metrics, None
    else:
        metrics, classwise_report = compute_rna_acc(
            predictions=pred_list,
            references=ref_list,
            metadata_list=metadata_list,
            compound_ratio=compound_ratio,
            return_classwise=True,
        )
        return metrics, classwise_report


def save_posthoc_report(
    output_dir: Optional[Path],
    task: str,
    metrics: Dict[str, float],
    *,
    classwise_report: Optional[Dict[str, Any]] = None,
    prediction_file: Optional[Path] = None,
    annos_file: Optional[Path] = None,
    compound_ratio: Optional[int] = None,
) -> None:
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "task": task,
        "prediction_file": str(prediction_file) if prediction_file else None,
        "annos_file": str(annos_file) if annos_file else None,
        "compound_ratio": compound_ratio,
        "metrics": metrics,
    }

    summary_path = output_dir / f"{task}_posthoc_metrics.json"
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)

    if task == "rna_sft" and classwise_report is not None:
        classwise_payload = {
            "task": task,
            "prediction_file": str(prediction_file) if prediction_file else None,
            "annos_file": str(annos_file) if annos_file else None,
            "compound_ratio": compound_ratio,
            "classwise": classwise_report,
        }
        classwise_path = output_dir / "rna_posthoc_classwise_metrics.json"
        with classwise_path.open("w") as f:
            json.dump(classwise_payload, f, indent=2)


def main():
    args = parse_args()

    prediction_payload = load_prediction_payload(args.prediction_file)
    annotation_payload = load_annotation_payload(args.annos_file)
    metrics, classwise_report = compute_posthoc_metrics(
        task=args.task,
        prediction_payload=prediction_payload,
        annotation_payload=annotation_payload,
        compound_ratio=args.compound_ratio,
    )
    save_posthoc_report(
        args.output_dir,
        args.task,
        metrics,
        classwise_report=classwise_report,
        prediction_file=args.prediction_file,
        annos_file=args.annos_file,
        compound_ratio=args.compound_ratio,
    )

    if args.task == "key_sft":
        print("Key metrics:")
    else:
        print("RNA metrics:")

    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
