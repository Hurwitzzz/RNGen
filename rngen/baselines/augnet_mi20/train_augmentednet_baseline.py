"""Train a controlled masked AugN or Mi20 run from derived converter NPZs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from rngen.paths import baseline_source_roots

from rngen.baselines.augnet_mi20.trainer import (
    MaskedTrainingConfig,
    concatenate_converted_npz,
    fit_masked_baseline,
)


def _parse_csv_numbers(value: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("AugmentedNet", "Micchi2020"),
        required=True,
    )
    parser.add_argument("--train-npz", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-npz", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--ontology-audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-report", type=Path,
        help="Materialization report whose source CSV roots must remain read-only.",
    )
    parser.add_argument(
        "--source-root", action="extend", nargs="+", type=Path,
        help="Additional read-only CSV roots; repeatable or space-separated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-boundaries", default="40")
    parser.add_argument("--lr-values", default="0.001,0.001")
    parser.add_argument("--max-validation-examples-per-head", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run-eagerly", action="store_true")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Clear CUDA visibility before TensorFlow is imported lazily.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    vocabularies = _load_vocabularies(args.ontology_audit)
    train_arrays = concatenate_converted_npz(args.train_npz)
    validation_arrays = concatenate_converted_npz(args.validation_npz)
    config = MaskedTrainingConfig(
        architecture=args.architecture,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate_boundaries=_parse_csv_numbers(args.lr_boundaries, int),
        learning_rate_values=_parse_csv_numbers(args.lr_values, float),
        max_validation_examples_per_head=args.max_validation_examples_per_head,
        verbose=0 if args.quiet else 1,
        run_eagerly=args.run_eagerly,
        read_only_source_roots=baseline_source_roots(args.source_root, args.cache_report),
        data_provenance={
            "code": {
                "git_commit": os.environ.get("RNGEN_EXPECTED_GIT_COMMIT"),
                "repository": str(Path.cwd().resolve()),
            },
            "ontology_audit": _file_record(args.ontology_audit),
            "train_npz": [_file_record(path) for path in args.train_npz],
            "validation_npz": [_file_record(path) for path in args.validation_npz],
        },
    )
    result = fit_masked_baseline(
        train_arrays,
        validation_arrays,
        vocabularies,
        config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
