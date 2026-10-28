"""Aggregate three-seed ChordGNN +Post test evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


SEEDS = (42, 43, 44)
HEADLINE_METRICS = ("RNA_acc", "K_acc", "D_acc", "Q_acc", "I_acc")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(
    *,
    evaluation_dirs: Mapping[int, Path],
    subset: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Post aggregate {output_path}")
    reports: dict[int, dict[str, Any]] = {}
    denominators: set[int] = set()
    manifest_hashes: set[str] = set()
    for seed in SEEDS:
        path = evaluation_dirs[seed] / "onset_metrics.json"
        report = json.loads(path.read_text())
        if (
            int(report["seed"]) != seed
            or report["split"] != "test"
            or report.get("test_subset") != subset
        ):
            raise ValueError(f"Invalid Post test report for seed {seed}")
        if (
            int(report.get("class_index_decoder_failures", -1)) != 0
            or int(report["counts"].get("prediction_decode_failures", -1)) != 0
            or int(report["counts"].get("missing_native_predictions", -1)) != 0
        ):
            raise ValueError(f"Post seed {seed} has test prediction failures")
        denominators.add(int(report["annotated_reference_note_onsets"]))
        manifest_hashes.add(report["manifest_sha256"])
        reports[seed] = report
    if len(denominators) != 1 or len(manifest_hashes) != 1:
        raise ValueError("Post seeds do not share the exact test denominator")

    metrics = {}
    for metric in HEADLINE_METRICS:
        values = [float(reports[seed]["metrics"][metric]) for seed in SEEDS]
        metrics[metric] = {
            "values": {str(seed): value for seed, value in zip(SEEDS, values)},
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
    payload = {
        "audit": "controlled ChordGNN+Post three-seed test aggregate",
        "status": "completed",
        "subset": subset,
        "pairs": [f"{reports[seed]['base_seed']}->{seed}" for seed in SEEDS],
        "annotated_reference_note_onsets": next(iter(denominators)),
        "test_manifest_sha256": next(iter(manifest_hashes)),
        "metrics": metrics,
        "reports": {
            str(seed): {
                "path": str((evaluation_dirs[seed] / "onset_metrics.json").resolve()),
                "sha256": _sha256(evaluation_dirs[seed] / "onset_metrics.json"),
            }
            for seed in SEEDS
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed-{seed}-evaluation", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate(
        evaluation_dirs={
            42: args.seed_42_evaluation,
            43: args.seed_43_evaluation,
            44: args.seed_44_evaluation,
        },
        subset=args.subset,
        output_path=args.output,
    )
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
