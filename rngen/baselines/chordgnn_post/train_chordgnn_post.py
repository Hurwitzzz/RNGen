"""CLI for controlled frozen-base ChordGNN +Post training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rngen.paths import baseline_source_roots

from rngen.baselines.chordgnn.train_chordgnn_baseline import _provenance, _read_paths
from rngen.baselines.chordgnn_post.trainer import (
    CONTROLLED_POST_PATIENCE,
    OFFICIAL_POST_MAX_EPOCHS,
    ChordGNNPostTrainingConfig,
    fit_chordgnn_post,
)


def run(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    cache_report: Path | None,
    output_dir: Path,
    ontology_path: Path,
    base_checkpoint: Path,
    base_run_config: Path,
    protocol: str,
    epochs: int,
    batch_size: int,
    base_seed: int,
    seed: int,
    hidden_size: int,
    n_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    device: str,
    resume: bool,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    source_roots: Sequence[Path],
) -> dict:
    if protocol == "formal" and cache_report is None:
        raise ValueError("Formal ChordGNN +Post training requires --cache-report")
    config = ChordGNNPostTrainingConfig(
        output_dir=output_dir,
        ontology_path=ontology_path,
        base_checkpoint=base_checkpoint,
        base_run_config=base_run_config,
        protocol=protocol,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        base_seed=base_seed,
        seed=seed,
        num_workers=num_workers,
        device=device,
        resume=resume,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        read_only_source_roots=baseline_source_roots(source_roots, cache_report),
        data_provenance=_provenance(train_manifest, validation_manifest, cache_report),
    )
    return fit_chordgnn_post(
        _read_paths(train_manifest),
        _read_paths(validation_manifest),
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-run-config", type=Path, required=True)
    parser.add_argument("--protocol", choices=("formal", "gate"), default="formal")
    parser.add_argument("--epochs", type=int, default=OFFICIAL_POST_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=CONTROLLED_POST_PATIENCE,
    )
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
    summary = run(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        cache_report=args.cache_report,
        output_dir=args.output_dir,
        ontology_path=args.ontology,
        base_checkpoint=args.base_checkpoint,
        base_run_config=args.base_run_config,
        protocol=args.protocol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_seed=args.base_seed,
        seed=args.seed,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        source_roots=args.source_root or (),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
