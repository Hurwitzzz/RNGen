"""Audit completed three-seed AugmentedNet or Micchi2020 evaluations.

The audit is independent of TensorFlow. It checks checkpoint availability and
aggregates saved evaluation reports without requiring training histories, Git
commits, or matching training configurations.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SEEDS = (42, 43, 44)
SUBSETS = ("beethoven_test", "full_test")
METRICS = ("RNA", "K", "Q", "D", "I")
REPORT_FILES = (
    "classwise_metrics.json",
    "evaluation_config.json",
    "failure_cases.json",
    "onset_metrics.json",
    "onset_predictions.jsonl",
    "per_piece_metrics.json",
)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sum_counts(per_piece: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        field: sum(piece["counts"][key][field] for piece in per_piece)
        for field in ("correct", "total")
    }


def _decoder_error_counts(failures: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                failure["decoder_error"]
                for failure in failures
                if failure["decoder_error"] is not None
            ).items()
        )
    )


def _audit_training(seed_root: Path) -> dict[str, Any]:
    checkpoint = seed_root / "training" / "checkpoints" / "best.weights.h5"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    return {
        "best_checkpoint": {"path": str(checkpoint.resolve()), "exists": True},
        "checks": ["checkpoint_file_exists"],
    }


def _audit_evaluation(
    seed_root: Path, architecture: str, subset: str, checkpoint: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = seed_root / "evaluation" / subset
    for name in REPORT_FILES:
        assert (output_dir / name).is_file(), output_dir / name

    metrics = _load(output_dir / "onset_metrics.json")
    config = _load(output_dir / "evaluation_config.json")
    per_piece = _load(output_dir / "per_piece_metrics.json")
    failures = _load(output_dir / "failure_cases.json")

    assert metrics["architecture"] == architecture
    assert len(per_piece) == metrics["pieces"]
    assert sum(piece["counts"]["RNA"]["total"] for piece in per_piece) == (
        metrics["annotated_reference_note_onsets"]
    )
    for metric in METRICS:
        assert _sum_counts(per_piece, metric) == metrics["counts"][metric]
    assert sum(piece["predicted_decoder_failures"] for piece in per_piece) == (
        metrics["predicted_decoder_failures"]
    )
    assert (
        sum(piece["raw_reference_representational_collapses"] for piece in per_piece)
        == metrics["raw_reference_representational_collapses"]
    )

    manifest = config["manifests"][0]

    native = metrics["native_fixed_frame_diagnostics"]["RomanNumeral"]
    result = {
        "annotated_reference_note_onsets": metrics["annotated_reference_note_onsets"],
        "checkpoint": checkpoint,
        "decoder_failure_reasons_observed_in_saved_failure_cases": _decoder_error_counts(
            failures
        ),
        "evaluator_elapsed_seconds": metrics["evaluation_elapsed_seconds"],
        "input_frame_collisions": metrics["input_frame_collisions"],
        "manifest": manifest,
        "native_fixed_frame_roman_numeral_conjunction": {
            "accuracy": native["accuracy"],
            "correct": native["correct"],
            "total": native["total"],
        },
        "output_dir": str(output_dir),
        "output_frame_collisions": metrics["output_frame_collisions"],
        "pieces": metrics["pieces"],
        "predicted_decoder_failures": metrics["predicted_decoder_failures"],
        "raw_reference_representational_collapses": metrics[
            "raw_reference_representational_collapses"
        ],
        "report_files": list(REPORT_FILES),
        "unified_exact_onset_counts": {
            metric: metrics["counts"][metric] for metric in METRICS
        },
        "unified_exact_onset_metrics": {
            metric: metrics["metrics"][f"{metric}_acc"] for metric in METRICS
        },
    }
    return result, per_piece


def audit(
    run_root: Path,
    architecture: str,
    training_job_id: int | None = None,
    evaluation_job_id: int | None = None,
) -> dict[str, Any]:
    seed_records: dict[str, Any] = {}
    denominators: dict[str, set[int]] = {subset: set() for subset in SUBSETS}
    manifests: dict[str, set[str]] = {subset: set() for subset in SUBSETS}

    for seed in SEEDS:
        seed_root = run_root / f"seed_{seed}"
        training = _audit_training(seed_root)

        evaluations: dict[str, Any] = {}
        per_piece_by_subset: dict[str, list[dict[str, Any]]] = {}
        for subset in SUBSETS:
            evaluation, per_piece = _audit_evaluation(
                seed_root, architecture, subset, training["best_checkpoint"]
            )
            evaluations[subset] = evaluation
            per_piece_by_subset[subset] = per_piece
            denominators[subset].add(evaluation["annotated_reference_note_onsets"])
            manifests[subset].add(evaluation["manifest"]["sha256"])

        beethoven_by_path = {
            piece["relative_path"]: piece
            for piece in per_piece_by_subset["beethoven_test"]
        }
        full_by_path = {
            piece["relative_path"]: piece for piece in per_piece_by_subset["full_test"]
        }
        assert all(
            full_by_path[path] == piece for path, piece in beethoven_by_path.items()
        )
        seed_records[str(seed)] = {
            "evaluations": evaluations,
            "training": training,
        }

    assert all(len(values) == 1 for values in denominators.values())
    assert all(len(values) == 1 for values in manifests.values())

    summary: dict[str, Any] = {}
    for subset in SUBSETS:
        summary[subset] = {}
        for metric in (*METRICS, "native_RN_conjunction"):
            if metric == "native_RN_conjunction":
                values = {
                    str(seed): seed_records[str(seed)]["evaluations"][subset][
                        "native_fixed_frame_roman_numeral_conjunction"
                    ]["accuracy"]
                    for seed in SEEDS
                }
            else:
                values = {
                    str(seed): seed_records[str(seed)]["evaluations"][subset][
                        "unified_exact_onset_metrics"
                    ][metric]
                    for seed in SEEDS
                }
            summary[subset][metric] = {
                "mean": statistics.mean(values.values()),
                "per_seed": values,
                "sample_standard_deviation": statistics.stdev(values.values()),
            }

    result = {
        "cross_checks": {
            "checkpoint_files_exist": True,
            "full_reports_reproduce_beethoven_subset_record_for_record": True,
            "per_piece_counts_sum_to_aggregate": True,
            "same_manifest_and_exact_onset_denominator_within_each_subset": True,
        },
        "jobs": {
            "evaluation_slurm_job_id": evaluation_job_id,
            "training_slurm_job_id": training_job_id,
        },
        "primary_metric_protocol": (
            "Exact annotated reference note onsets; RNA is the conjunction of "
            "LocalKey35, PrimaryDegree22, SecondaryDegree22, ChordQuality15, "
            "ChordRoot35, and Inversion4."
        ),
        "purpose": (
            f"Audit {architecture} seeds 42/43/44 and their Beethoven-8 and "
            "complete 47-piece P.DCML test evaluations."
        ),
        "seeds": seed_records,
        "standard_deviation_convention": (
            "sample standard deviation across the three supplied reports "
            "(n=3, ddof=1)"
        ),
        "status": f"completed_{architecture.lower()}_three_seed_test_audit",
        "summary": summary,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("AugmentedNet", "Micchi2020"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-job-id", type=int)
    parser.add_argument("--evaluation-job-id", type=int)
    args = parser.parse_args()

    result = audit(
        args.run_root,
        args.architecture,
        training_job_id=args.training_job_id,
        evaluation_job_id=args.evaluation_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
