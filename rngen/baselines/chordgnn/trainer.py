"""Controlled data loading and training utilities for ChordGNN."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from rngen.baselines.shared.chordgnn_provenance import (
    full_provenance,
    training_config_fingerprint,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    CORE_COMPONENTS,
)
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import ChordGraphExample
from rngen.baselines.chordgnn.model import (
    ChordGNN,
    ChordGraphBatch,
    SafeMaskedWLoss,
    add_reverse_edges,
    chordgnn_parameter_count,
)


OFFICIAL_BUCKET_BOUNDARIES = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def prepare_output_directory(
    output_dir: Path,
    *,
    read_only_source_roots: Sequence[Path],
    resume: bool = False,
) -> None:
    for source_root in read_only_source_roots:
        if _is_relative_to(output_dir, source_root):
            raise ValueError(
                f"Refusing to write ChordGNN output {output_dir} inside "
                f"read-only source root {source_root}"
            )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


class ChordGraphDataset(Dataset[ChordGraphExample]):
    def __init__(self, paths: Sequence[Path]):
        if not paths:
            raise ValueError("ChordGraphDataset requires at least one graph")
        self.paths = tuple(Path(path) for path in paths)
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ChordGNN graphs: {missing[:5]}")
        self.lengths = tuple(self._read_length(path) for path in self.paths)

    @staticmethod
    def _read_length(path: Path) -> int:
        with np.load(path, allow_pickle=False) as arrays:
            return int(arrays["onset_quarter"].shape[0])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ChordGraphExample:
        return ChordGraphExample.load_npz(self.paths[index])


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Correct implementation of the upstream length-bucket protocol."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        seed: int,
        boundaries: Sequence[int] = OFFICIAL_BUCKET_BOUNDARIES,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.boundaries = tuple(int(value) for value in boundaries)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._buckets: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            bucket = int(np.digitize(length, self.boundaries, right=False))
            self._buckets.setdefault(bucket, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket in sorted(self._buckets):
            indices = self._buckets[bucket].copy()
            if self.shuffle:
                generator.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._buckets.values()
        )


def collate_chord_graphs(examples: Sequence[ChordGraphExample]) -> ChordGraphBatch:
    if not examples:
        raise ValueError("Cannot collate an empty ChordGNN batch")
    ordered = sorted(
        examples, key=lambda example: len(example.onset_quarter), reverse=True
    )
    node_offset = 0
    features: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    onset_edges: list[torch.Tensor] = []
    representatives: list[torch.Tensor] = []
    onset_quarter: list[torch.Tensor] = []
    for example in ordered:
        x = torch.from_numpy(example.x)
        edge_index = torch.from_numpy(example.edge_index)
        edge_type = torch.from_numpy(example.edge_type.astype(np.int64))
        features.append(x)
        edges.append(edge_index + node_offset)
        edge_types.append(edge_type)
        onset_edges.append(edge_index[:, edge_type == 0] + node_offset)
        representatives.append(
            torch.from_numpy(example.onset_representatives) + node_offset
        )
        onset_quarter.append(torch.from_numpy(example.onset_quarter))
        node_offset += len(example.x)

    edge_index = torch.cat(edges, dim=1)
    edge_type = torch.cat(edge_types, dim=0)
    edge_index, edge_type = add_reverse_edges(edge_index, edge_type)
    lengths = torch.tensor(
        [len(example.onset_quarter) for example in ordered], dtype=torch.long
    )
    max_length = int(lengths.max())
    labels: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for component in COMPONENTS:
        component_labels = torch.full((len(ordered), max_length), -1, dtype=torch.long)
        component_masks = torch.zeros((len(ordered), max_length), dtype=torch.bool)
        for batch_index, example in enumerate(ordered):
            length = len(example.onset_quarter)
            component_labels[batch_index, :length] = torch.from_numpy(
                np.asarray(example.labels[component], dtype=np.int64)
            )
            component_masks[batch_index, :length] = torch.from_numpy(
                np.asarray(example.masks[component], dtype=np.bool_)
            )
        labels[component] = component_labels
        masks[component] = component_masks
    return ChordGraphBatch(
        x=torch.cat(features, dim=0),
        edge_index=edge_index,
        edge_type=edge_type,
        onset_edges=torch.cat(onset_edges, dim=1),
        onset_representatives=torch.cat(representatives, dim=0),
        lengths=lengths,
        labels=labels,
        masks=masks,
        onset_quarter=tuple(onset_quarter),
        relative_paths=tuple(example.relative_path for example in ordered),
    )


@dataclass(frozen=True)
class ChordGNNTrainingConfig:
    output_dir: Path
    ontology_path: Path
    protocol: str = "formal"
    epochs: int = 100
    batch_size: int = 100
    hidden_size: int = 256
    n_layers: int = 1
    dropout: float = 0.5
    learning_rate: float = 0.0015
    weight_decay: float = 0.005
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    resume: bool = False
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 0.0
    read_only_source_roots: tuple[Path, ...] = ()
    data_provenance: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.protocol not in {"formal", "gate"}:
            raise ValueError("ChordGNN protocol must be formal or gate")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.hidden_size != 256 or self.n_layers != 1:
            raise ValueError("Controlled ChordGNN requires hidden=256 and n_layers=1")
        if self.protocol == "formal" and (self.epochs != 100 or self.batch_size != 100):
            raise ValueError("Formal ChordGNN requires 100 epochs and batch size 100")
        if self.protocol == "formal" and self.dropout != 0.5:
            raise ValueError("Controlled ChordGNN requires paper dropout=0.5")
        if self.protocol == "formal" and (
            self.learning_rate != 0.0015 or self.weight_decay != 0.005
        ):
            raise ValueError("Controlled ChordGNN requires paper AdamW settings")
        if self.protocol == "formal" and (
            self.early_stopping_patience != 20 or self.early_stopping_min_delta != 0.0
        ):
            raise ValueError(
                "Formal ChordGNN requires early stopping patience=20 and min_delta=0"
            )
        if self.protocol == "formal" and self.seed not in (42, 43, 44):
            raise ValueError("Formal ChordGNN seeds are 42, 43, and 44")


def set_reproducibility(seed: int) -> dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": True,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _runtime_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        payload["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
        }
    return payload


def _head_sizes(ontology: Mapping[str, Any]) -> dict[str, int]:
    vocabularies = ontology["vocabularies"]
    if set(vocabularies) != set(COMPONENTS):
        raise ValueError("Ontology component order does not match ChordGNN")
    return {component: len(vocabularies[component]) for component in COMPONENTS}


def _masked_counts(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    counts: dict[str, dict[str, float | int]] = {}
    for component in COMPONENTS:
        mask = masks[component].bool()
        support = int(mask.sum())
        if support:
            component_logits = logits[component][mask]
            component_labels = labels[component][mask]
            loss_sum = float(
                F.cross_entropy(
                    component_logits, component_labels, reduction="sum"
                ).item()
            )
            correct = int(
                (component_logits.argmax(dim=-1) == component_labels).sum().item()
            )
        else:
            loss_sum = 0.0
            correct = 0
        counts[component] = {
            "loss_sum": loss_sum,
            "correct": correct,
            "support": support,
        }
    return counts


def _merge_counts(
    aggregate: dict[str, dict[str, float | int]],
    update: Mapping[str, Mapping[str, float | int]],
) -> None:
    for component in COMPONENTS:
        aggregate[component]["loss_sum"] = float(
            aggregate[component]["loss_sum"]
        ) + float(update[component]["loss_sum"])
        aggregate[component]["correct"] = int(aggregate[component]["correct"]) + int(
            update[component]["correct"]
        )
        aggregate[component]["support"] = int(aggregate[component]["support"]) + int(
            update[component]["support"]
        )


def _finalize_counts(counts: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for component in COMPONENTS:
        support = int(counts[component]["support"])
        loss_sum = float(counts[component]["loss_sum"])
        correct = int(counts[component]["correct"])
        heads[component] = {
            "loss": loss_sum / support if support else 0.0,
            "accuracy": correct / support if support else 0.0,
            "correct": correct,
            "support": support,
        }
    monitored_loss = sum(heads[component]["loss"] for component in CORE_COMPONENTS)
    monitored_accuracy = sum(
        heads[component]["accuracy"] for component in CORE_COMPONENTS
    ) / len(CORE_COMPONENTS)
    return {
        "heads": heads,
        "monitored_loss": monitored_loss,
        "monitored_accuracy": monitored_accuracy,
    }


@torch.no_grad()
def evaluate_loader(
    model: ChordGNN,
    loader: DataLoader[ChordGraphBatch],
    *,
    device: torch.device,
    example_limit: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    counts = {
        component: {"loss_sum": 0.0, "correct": 0, "support": 0}
        for component in COMPONENTS
    }
    examples: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        _merge_counts(counts, _masked_counts(logits, batch.labels, batch.masks))
        if len(examples) < example_limit:
            for batch_index, relative_path in enumerate(batch.relative_paths):
                if len(examples) >= example_limit:
                    break
                length = int(batch.lengths[batch_index])
                examples.append(
                    {
                        "relative_path": relative_path,
                        "onsets": batch.onset_quarter[batch_index].tolist(),
                        "targets": {
                            component: batch.labels[component][batch_index, :length]
                            .detach()
                            .cpu()
                            .tolist()
                            for component in COMPONENTS
                        },
                        "predictions": {
                            component: logits[component][batch_index, :length]
                            .argmax(dim=-1)
                            .detach()
                            .cpu()
                            .tolist()
                            for component in COMPONENTS
                        },
                    }
                )
    return _finalize_counts(counts), examples


def _checkpoint_payload(
    *,
    model: ChordGNN,
    loss_module: SafeMaskedWLoss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_monitored_loss: float,
    epochs_without_improvement: int,
    config_hash: str,
    ontology_hash: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_monitored_loss": best_monitored_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config_sha256": config_hash,
        "ontology_sha256": ontology_hash,
    }


def _write_curves(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    columns = sorted({key for row in history for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def fit_chordgnn(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    config: ChordGNNTrainingConfig,
) -> dict[str, Any]:
    config.validate()
    prepare_output_directory(
        config.output_dir,
        read_only_source_roots=config.read_only_source_roots,
        resume=config.resume,
    )
    reproducibility = set_reproducibility(config.seed)
    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ontology = json.loads(config.ontology_path.read_text())
    ontology_hash = _sha256(config.ontology_path)
    head_sizes = _head_sizes(ontology)
    train_dataset = ChordGraphDataset(train_paths)
    validation_dataset = ChordGraphDataset(validation_paths)
    train_sampler = LengthBucketBatchSampler(
        train_dataset.lengths,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_chord_graphs,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_chord_graphs,
        pin_memory=device.type == "cuda",
    )
    model = ChordGNN(
        head_sizes,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    loss_module = SafeMaskedWLoss().to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "weight_decay": config.weight_decay,
            },
            {
                "params": loss_module.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
    )
    provenance = full_provenance(require_git=False)
    if config.protocol == "formal":
        if config.data_provenance is None:
            raise RuntimeError("Formal ChordGNN training requires cache provenance")
        if not config.data_provenance.get("cache_report_sha256"):
            raise RuntimeError("Formal ChordGNN training is missing the cache report hash")
    serialized_config = asdict(config)
    serialized_config["output_dir"] = str(config.output_dir.resolve())
    serialized_config["ontology_path"] = str(config.ontology_path.resolve())
    serialized_config["read_only_source_roots"] = [
        str(path.resolve()) for path in config.read_only_source_roots
    ]
    config_hash = training_config_fingerprint(
        serialized_config, output_dir=config.output_dir, resume=config.resume
    )
    run_config = {
        "training": serialized_config,
        "controlled_protocol": {
            "optimizer": "AdamW",
            "fp32": True,
            "amp": False,
            "paper_hyperparameters": True,
            "upstream_code_defaults_recorded_only": {
                "dropout": 0.44,
                "weight_decay": 0.0035,
            },
            "checkpoint_monitor": "unweighted sum of six core masked validation CE",
            "early_stopping": {
                "enabled": config.early_stopping_patience is not None,
                "patience": config.early_stopping_patience,
                "min_delta": config.early_stopping_min_delta,
                "mode": "min",
                "restore_best_checkpoint": True,
            },
            "bucket_boundaries": list(OFFICIAL_BUCKET_BOUNDARIES),
        },
        "runtime": _runtime_payload(device),
        "provenance": provenance,
        "reproducibility": reproducibility,
        "model": {
            "head_sizes": head_sizes,
            "parameters": chordgnn_parameter_count(model),
        },
        "data": {
            "train_graphs": len(train_dataset),
            "validation_graphs": len(validation_dataset),
            "train_onsets": sum(train_dataset.lengths),
            "validation_onsets": sum(validation_dataset.lengths),
            "provenance": config.data_provenance,
        },
        "ontology_sha256": ontology_hash,
        "config_sha256": config_hash,
    }
    _json_dump(config.output_dir / "run_config.json", run_config)
    (config.output_dir / "ontology.json").write_text(config.ontology_path.read_text())
    (config.output_dir / "model_summary.txt").write_text(str(model) + "\n")
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    best_path = checkpoints / "best.pt"
    last_path = checkpoints / "last.pt"

    start_epoch = 1
    best_monitored_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    history_path = config.output_dir / "history.json"
    if config.resume:
        if not last_path.is_file() or not history_path.is_file():
            raise FileNotFoundError("Resume requires last.pt and history.json")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("Resume config hash differs from last checkpoint")
        if checkpoint["ontology_sha256"] != ontology_hash:
            raise ValueError("Resume ontology hash differs from last checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        loss_module.load_state_dict(checkpoint["loss_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_monitored_loss = float(checkpoint["best_monitored_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            raise ValueError("Resume checkpoint already met early-stopping patience")
        history = json.loads(history_path.read_text())

    for epoch in range(start_epoch, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        loss_module.train()
        train_total = 0.0
        train_batches = 0
        train_counts = {
            component: {"loss_sum": 0.0, "correct": 0, "support": 0}
            for component in COMPONENTS
        }
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            total_loss, _ = loss_module(logits, batch.labels, batch.masks)
            if not torch.isfinite(total_loss):
                _json_dump(
                    config.output_dir / "failure_audit.json",
                    {"status": "failed", "stage": "train_loss", "epoch": epoch},
                )
                raise RuntimeError(f"Non-finite ChordGNN train loss at epoch {epoch}")
            total_loss.backward()
            optimizer.step()
            train_total += float(total_loss.detach().item())
            train_batches += 1
            with torch.no_grad():
                _merge_counts(
                    train_counts, _masked_counts(logits, batch.labels, batch.masks)
                )
        validation_metrics, validation_examples = evaluate_loader(
            model, validation_loader, device=device
        )
        monitored = float(validation_metrics["monitored_loss"])
        if not math_isfinite(monitored):
            _json_dump(
                config.output_dir / "failure_audit.json",
                {"status": "failed", "stage": "validation_loss", "epoch": epoch},
            )
            raise RuntimeError(f"Non-finite ChordGNN validation loss at epoch {epoch}")
        train_metrics = _finalize_counts(train_counts)
        improved = monitored < (best_monitored_loss - config.early_stopping_min_delta)
        if improved:
            best_monitored_loss = monitored
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_wloss": train_total / max(1, train_batches),
            "train_monitored_loss": train_metrics["monitored_loss"],
            "validation_monitored_loss": monitored,
            "validation_monitored_accuracy": validation_metrics["monitored_accuracy"],
            "early_stopping_counter": epochs_without_improvement,
            "is_best": improved,
        }
        for component in COMPONENTS:
            row[f"train_{component}_loss"] = train_metrics["heads"][component]["loss"]
            row[f"train_{component}_accuracy"] = train_metrics["heads"][component][
                "accuracy"
            ]
            row[f"validation_{component}_loss"] = validation_metrics["heads"][
                component
            ]["loss"]
            row[f"validation_{component}_accuracy"] = validation_metrics["heads"][
                component
            ]["accuracy"]
        history.append(row)
        if not all(
            math_isfinite(float(value))
            for value in row.values()
            if isinstance(value, (float, int))
        ):
            _json_dump(
                config.output_dir / "failure_audit.json",
                {"status": "failed", "stage": "history", "epoch": epoch},
            )
            raise RuntimeError(f"Non-finite ChordGNN history at epoch {epoch}")
        if improved:
            torch.save(
                _checkpoint_payload(
                    model=model,
                    loss_module=loss_module,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_monitored_loss=best_monitored_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    config_hash=config_hash,
                    ontology_hash=ontology_hash,
                ),
                best_path,
            )
            _json_dump(
                config.output_dir / "validation_examples.json",
                validation_examples,
            )
        torch.save(
            _checkpoint_payload(
                model=model,
                loss_module=loss_module,
                optimizer=optimizer,
                epoch=epoch,
                best_monitored_loss=best_monitored_loss,
                epochs_without_improvement=epochs_without_improvement,
                config_hash=config_hash,
                ontology_hash=ontology_hash,
            ),
            last_path,
        )
        _json_dump(history_path, history)
        _write_curves(config.output_dir / "curves.csv", history)
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_metrics, validation_examples = evaluate_loader(
        model, validation_loader, device=device, example_limit=10
    )
    _json_dump(
        config.output_dir / "best_validation_metrics.json",
        {
            "checkpoint": str(best_path.resolve()),
            "checkpoint_sha256": _sha256(best_path),
            "epoch": int(best_checkpoint["epoch"]),
            "metrics": best_metrics,
        },
    )
    _json_dump(config.output_dir / "validation_examples.json", validation_examples)
    summary = {
        "status": "completed",
        "stop_reason": "early_stopping" if stopped_early else "max_epochs",
        "early_stopped": stopped_early,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_counter_at_stop": epochs_without_improvement,
        "seed": config.seed,
        "requested_epochs": config.epochs,
        "completed_epochs": len(history),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_monitored_loss": best_metrics["monitored_loss"],
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _sha256(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": _sha256(last_path),
        "history": str(history_path.resolve()),
        "run_config": str((config.output_dir / "run_config.json").resolve()),
    }
    _json_dump(config.output_dir / "run_summary.json", summary)
    return summary


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))
