"""CLI for controlled ChordGNN training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from rngen.paths import baseline_source_roots

from rngen.baselines.chordgnn.trainer import (
    ChordGNNTrainingConfig,
    fit_chordgnn,
)


def _read_paths(path: Path) -> list[Path]:
    values = [
        Path(line.strip()) for line in path.read_text().splitlines() if line.strip()
    ]
    if not values:
        raise ValueError(f"Graph manifest is empty: {path}")
    return values


def _provenance(
    train_manifest: Path, validation_manifest: Path, cache_report: Path | None
) -> dict[str, Any]:
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "train_graph_manifest": str(train_manifest.resolve()),
        "train_graph_manifest_sha256": digest(train_manifest),
        "validation_graph_manifest": str(validation_manifest.resolve()),
        "validation_graph_manifest_sha256": digest(validation_manifest),
    }
    if cache_report is not None:
        report = json.loads(cache_report.read_text())
        gates = report.get("gates", {})
        if report.get("status") != "completed" or not gates or not all(gates.values()):
            raise ValueError("ChordGNN cache report did not pass every gate")
        if int(report["splits"]["train"]["graphs"]) != 4260:
            raise ValueError(
                "ChordGNN cache report does not contain 4,260 train graphs"
            )
        if int(report["splits"]["validation"]["graphs"]) != 44:
            raise ValueError(
                "ChordGNN cache report does not contain 44 validation graphs"
            )
        cache_git = report.get("provenance", {}).get("git", {})
        payload["cache_report"] = str(cache_report.resolve())
        payload["cache_report_sha256"] = digest(cache_report)
        payload["cache_git_commit"] = cache_git.get("commit")
        payload["cache_git_dirty"] = cache_git.get("dirty")
    return payload


def run(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    ontology_path: Path,
    cache_report: Path | None,
    protocol: str,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    num_workers: int,
    device: str,
    resume: bool,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    source_roots: Sequence[Path],
) -> dict:
    if protocol == "formal" and cache_report is None:
        raise ValueError("Formal ChordGNN training requires --cache-report")
    config = ChordGNNTrainingConfig(
        output_dir=output_dir,
        ontology_path=ontology_path,
        protocol=protocol,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        num_workers=num_workers,
        device=device,
        resume=resume,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        read_only_source_roots=baseline_source_roots(source_roots, cache_report),
        data_provenance=_provenance(train_manifest, validation_manifest, cache_report),
    )
    return fit_chordgnn(
        _read_paths(train_manifest),
        _read_paths(validation_manifest),
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path)
    parser.add_argument("--protocol", choices=("formal", "gate"), default="formal")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-root",
        action="extend",
        nargs="+",
        type=Path,
        help="Additional read-only CSV roots; repeatable or space-separated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        ontology_path=args.ontology,
        cache_report=args.cache_report,
        protocol=args.protocol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        source_roots=args.source_root or (),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
