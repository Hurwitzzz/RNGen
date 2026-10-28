"""Controlled training for the learned ChordGNN +Post stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from rngen.baselines.shared.chordgnn_provenance import (
    full_provenance,
    training_config_fingerprint,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import COMPONENTS, CORE_COMPONENTS
from rngen.baselines.chordgnn.model import ChordGNN, SafeMaskedWLoss
from rngen.baselines.chordgnn_post.model import ChordGNNPostProcessor, freeze_chordgnn
from rngen.baselines.chordgnn.trainer import (
    ChordGraphDataset,
    LengthBucketBatchSampler,
    _finalize_counts,
    _head_sizes,
    _json_dump,
    _masked_counts,
    _merge_counts,
    _runtime_payload,
    _sha256,
    _write_curves,
    collate_chord_graphs,
    math_isfinite,
    prepare_output_directory,
    set_reproducibility,
)


OFFICIAL_POST_MAX_EPOCHS = 31
CONTROLLED_POST_PATIENCE = 10
EXPECTED_POST_INPUT_SIZE = 379


@dataclass(frozen=True)
class ChordGNNPostTrainingConfig:
    output_dir: Path
    ontology_path: Path
    base_checkpoint: Path
    base_run_config: Path
    protocol: str = "formal"
    epochs: int = OFFICIAL_POST_MAX_EPOCHS
    batch_size: int = 100
    hidden_size: int = 256
    n_layers: int = 1
    dropout: float = 0.4
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    base_seed: int = 42
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    resume: bool = False
    early_stopping_patience: int | None = CONTROLLED_POST_PATIENCE
    early_stopping_min_delta: float = 0.0
    read_only_source_roots: tuple[Path, ...] = ()
    data_provenance: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.protocol not in {"formal", "gate"}:
            raise ValueError("ChordGNN +Post protocol must be formal or gate")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("ChordGNN +Post epochs and batch size must be positive")
        if self.hidden_size <= 0 or self.n_layers <= 0:
            raise ValueError("ChordGNN +Post hidden size and layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ChordGNN +Post dropout must be in [0, 1)")
        if self.learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("ChordGNN +Post optimizer settings must be non-negative")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("ChordGNN +Post patience must be positive or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("ChordGNN +Post min_delta must be non-negative")
        if self.protocol == "formal":
            expected = {
                "epochs": OFFICIAL_POST_MAX_EPOCHS,
                "batch_size": 100,
                "hidden_size": 256,
                "n_layers": 1,
                "dropout": 0.4,
                "learning_rate": 0.001,
                "weight_decay": 5e-4,
                "early_stopping_patience": CONTROLLED_POST_PATIENCE,
                "early_stopping_min_delta": 0.0,
            }
            for key, value in expected.items():
                if getattr(self, key) != value:
                    raise ValueError(f"Formal ChordGNN +Post requires {key}={value!r}")
            if self.seed not in (42, 43, 44):
                raise ValueError("Formal Post seeds are 42, 43, and 44")
            if self.base_seed != self.seed:
                raise ValueError(
                    "Formal Post requires same-seed base pairing "
                    "(42->42, 43->43, or 44->44)"
                )


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_base(
    *,
    checkpoint_path: Path,
    run_config_path: Path,
    ontology: Mapping[str, Any],
    ontology_hash: str,
    expected_seed: int,
    device: torch.device,
    require_completed_formal_run: bool,
) -> tuple[ChordGNN, dict[str, Any]]:
    run_config = json.loads(run_config_path.read_text())
    training = run_config["training"]
    if int(training["seed"]) != expected_seed:
        raise ValueError("ChordGNN +Post base seed differs from the requested seed")
    if run_config.get("ontology_sha256") != ontology_hash:
        raise ValueError("ChordGNN +Post base ontology differs from Post ontology")
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config_sha256") != run_config.get("config_sha256"):
        raise ValueError("ChordGNN +Post base checkpoint config hash mismatch")
    if checkpoint.get("ontology_sha256") != ontology_hash:
        raise ValueError("ChordGNN +Post base checkpoint ontology hash mismatch")
    if require_completed_formal_run:
        if training.get("protocol") != "formal":
            raise ValueError("Formal +Post requires a formal base ChordGNN run")
        summary_path = run_config_path.parent / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("Formal +Post requires the base run summary")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "completed":
            raise ValueError("Formal +Post base run is not completed")
        if summary.get("best_checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Formal +Post did not receive the base best checkpoint")

    head_sizes = _head_sizes(ontology)
    model = ChordGNN(
        head_sizes,
        hidden_size=int(training["hidden_size"]),
        n_layers=int(training["n_layers"]),
        dropout=float(training["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    freeze_chordgnn(model)
    identity = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "run_config": str(run_config_path.resolve()),
        "run_config_sha256": _sha256(run_config_path),
        "config_sha256": run_config["config_sha256"],
        "ontology_sha256": ontology_hash,
        "seed": int(training["seed"]),
        "git_commit": run_config.get("provenance", {}).get("git", {}).get("commit"),
        "state_sha256": _state_sha256(model),
    }
    return model, identity


@torch.no_grad()
def evaluate_post_loader(
    base_model: ChordGNN,
    post_model: ChordGNNPostProcessor,
    loader: DataLoader,
    *,
    device: torch.device,
    example_limit: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_model.eval()
    post_model.eval()
    counts = {
        component: {"loss_sum": 0.0, "correct": 0, "support": 0}
        for component in COMPONENTS
    }
    core_correct = 0
    core_support = 0
    examples: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        base_logits = base_model(batch)
        logits = post_model(base_logits, batch.lengths)
        _merge_counts(counts, _masked_counts(logits, batch.labels, batch.masks))
        core_mask = torch.stack(
            [batch.masks[component].bool() for component in CORE_COMPONENTS]
        ).all(dim=0)
        conjunction = core_mask.clone()
        for component in CORE_COMPONENTS:
            conjunction &= logits[component].argmax(dim=-1) == batch.labels[component]
        core_correct += int(conjunction.sum().item())
        core_support += int(core_mask.sum().item())
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
    metrics = _finalize_counts(counts)
    metrics["core_conjunction"] = {
        "correct": core_correct,
        "support": core_support,
        "accuracy": core_correct / core_support if core_support else 0.0,
    }
    return metrics, examples


def _checkpoint_payload(
    *,
    post_model: ChordGNNPostProcessor,
    loss_module: SafeMaskedWLoss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_monitored_loss: float,
    epochs_without_improvement: int,
    config_hash: str,
    ontology_hash: str,
    base_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "post_state": post_model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_monitored_loss": best_monitored_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config_sha256": config_hash,
        "ontology_sha256": ontology_hash,
        "base_checkpoint_sha256": base_identity["checkpoint_sha256"],
        "base_config_sha256": base_identity["config_sha256"],
        "base_state_sha256": base_identity["state_sha256"],
        "base_seed": base_identity["seed"],
    }


def fit_chordgnn_post(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    config: ChordGNNPostTrainingConfig,
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
    if (
        config.protocol == "formal"
        and sum(head_sizes.values()) != EXPECTED_POST_INPUT_SIZE
    ):
        raise ValueError("Formal ChordGNN +Post requires the 379-class ontology")

    provenance = full_provenance(require_git=False)
    if config.protocol == "formal":
        if config.data_provenance is None:
            raise RuntimeError("Formal ChordGNN +Post requires cache provenance")
        if not config.data_provenance.get("cache_report_sha256"):
            raise RuntimeError("Formal ChordGNN +Post is missing the cache report hash")

    base_model, base_identity = load_frozen_base(
        checkpoint_path=config.base_checkpoint,
        run_config_path=config.base_run_config,
        ontology=ontology,
        ontology_hash=ontology_hash,
        expected_seed=config.base_seed,
        device=device,
        require_completed_formal_run=config.protocol == "formal",
    )
    base_file_hash_before = _sha256(config.base_checkpoint)
    base_state_hash_before = _state_sha256(base_model)

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
    post_model = ChordGNNPostProcessor(
        head_sizes,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    loss_module = SafeMaskedWLoss().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": post_model.parameters(), "weight_decay": config.weight_decay},
            {"params": loss_module.parameters(), "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    serialized_config = asdict(config)
    for key in ("output_dir", "ontology_path", "base_checkpoint", "base_run_config"):
        serialized_config[key] = str(Path(serialized_config[key]).resolve())
    serialized_config["read_only_source_roots"] = [
        str(path.resolve()) for path in config.read_only_source_roots
    ]
    config_hash = training_config_fingerprint(
        serialized_config, output_dir=config.output_dir, resume=config.resume
    )
    run_config = {
        "training": serialized_config,
        "controlled_protocol": {
            "stage": "ChordGNN+Post",
            "base_frozen": True,
            "post_input": "379 concatenated per-head softmax probabilities",
            "post_lstm_packing": False,
            "optimizer": "AdamW",
            "fp32": True,
            "amp": False,
            "checkpoint_monitor": "unweighted sum of six core masked validation CE",
            "early_stopping": {
                "patience": config.early_stopping_patience,
                "min_delta": config.early_stopping_min_delta,
                "mode": "min",
            },
            "official_differences": {
                "official_max_epochs": 31,
                "official_checkpoint_monitor": "Val RomNum (max)",
                "declared_but_unattached_early_stop": {
                    "patience": 5,
                    "min_delta": 0.02,
                },
            },
        },
        "runtime": _runtime_payload(device),
        "provenance": provenance,
        "reproducibility": reproducibility,
        "base_model": base_identity,
        "model": {
            "head_sizes": head_sizes,
            "input_size": post_model.input_size,
            "post_parameters": sum(p.numel() for p in post_model.parameters()),
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
    (config.output_dir / "model_summary.txt").write_text(str(post_model) + "\n")
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    best_path = checkpoints / "best.pt"
    last_path = checkpoints / "last.pt"
    history_path = config.output_dir / "history.json"

    start_epoch = 1
    best_monitored_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    if config.resume:
        if not last_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(
                "ChordGNN +Post resume requires last.pt and history"
            )
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        expected_checkpoint_fields = {
            "config_sha256": config_hash,
            "ontology_sha256": ontology_hash,
            "base_checkpoint_sha256": base_identity["checkpoint_sha256"],
            "base_config_sha256": base_identity["config_sha256"],
            "base_state_sha256": base_identity["state_sha256"],
            "base_seed": config.base_seed,
        }
        for key, value in expected_checkpoint_fields.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"ChordGNN +Post resume mismatch for {key}")
        post_model.load_state_dict(checkpoint["post_state"])
        loss_module.load_state_dict(checkpoint["loss_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_monitored_loss = float(checkpoint["best_monitored_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            raise ValueError("ChordGNN +Post resume already met early-stop patience")
        history = json.loads(history_path.read_text())

    for epoch in range(start_epoch, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        base_model.eval()
        post_model.train()
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
            with torch.no_grad():
                base_logits = base_model(batch)
            logits = post_model(base_logits, batch.lengths)
            total_loss, _ = loss_module(logits, batch.labels, batch.masks)
            if not torch.isfinite(total_loss):
                _json_dump(
                    config.output_dir / "failure_audit.json",
                    {"status": "failed", "stage": "train_loss", "epoch": epoch},
                )
                raise RuntimeError(f"Non-finite ChordGNN +Post loss at epoch {epoch}")
            total_loss.backward()
            if any(parameter.grad is not None for parameter in base_model.parameters()):
                raise RuntimeError("Frozen ChordGNN received gradients during +Post")
            optimizer.step()
            train_total += float(total_loss.detach().item())
            train_batches += 1
            with torch.no_grad():
                _merge_counts(
                    train_counts, _masked_counts(logits, batch.labels, batch.masks)
                )

        validation_metrics, validation_examples = evaluate_post_loader(
            base_model,
            post_model,
            validation_loader,
            device=device,
        )
        monitored = float(validation_metrics["monitored_loss"])
        if not math_isfinite(monitored):
            raise RuntimeError(
                f"Non-finite ChordGNN +Post validation loss at epoch {epoch}"
            )
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
            "validation_core_conjunction": validation_metrics["core_conjunction"][
                "accuracy"
            ],
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
            raise RuntimeError(f"Non-finite ChordGNN +Post history at epoch {epoch}")
        payload = _checkpoint_payload(
            post_model=post_model,
            loss_module=loss_module,
            optimizer=optimizer,
            epoch=epoch,
            best_monitored_loss=best_monitored_loss,
            epochs_without_improvement=epochs_without_improvement,
            config_hash=config_hash,
            ontology_hash=ontology_hash,
            base_identity=base_identity,
        )
        if improved:
            torch.save(payload, best_path)
            _json_dump(
                config.output_dir / "validation_examples.json", validation_examples
            )
        torch.save(payload, last_path)
        _json_dump(history_path, history)
        _write_curves(config.output_dir / "curves.csv", history)
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    expected_best_row = min(
        history,
        key=lambda row: (
            float(row["validation_monitored_loss"]),
            int(row["epoch"]),
        ),
    )
    if int(best_checkpoint["epoch"]) != int(expected_best_row["epoch"]):
        raise RuntimeError(
            "ChordGNN +Post best checkpoint is not the strict global "
            "validation-core-CE minimum"
        )
    _json_dump(
        config.output_dir / "best_selection_audit.json",
        {
            "status": "passed",
            "monitor": "unweighted sum of six core masked validation CE",
            "strict_improvement": True,
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_history_value": float(expected_best_row["validation_monitored_loss"]),
            "epochs_compared": len(history),
        },
    )
    post_model.load_state_dict(best_checkpoint["post_state"])
    best_metrics, validation_examples = evaluate_post_loader(
        base_model,
        post_model,
        validation_loader,
        device=device,
        example_limit=10,
    )
    base_file_hash_after = _sha256(config.base_checkpoint)
    base_state_hash_after = _state_sha256(base_model)
    freeze_audit = {
        "base_checkpoint_sha256_before": base_file_hash_before,
        "base_checkpoint_sha256_after": base_file_hash_after,
        "base_state_sha256_before": base_state_hash_before,
        "base_state_sha256_after": base_state_hash_after,
        "all_parameters_require_grad_false": all(
            not parameter.requires_grad for parameter in base_model.parameters()
        ),
        "all_parameter_gradients_none": all(
            parameter.grad is None for parameter in base_model.parameters()
        ),
    }
    freeze_audit["passed"] = (
        base_file_hash_before == base_file_hash_after
        and base_state_hash_before == base_state_hash_after
        and freeze_audit["all_parameters_require_grad_false"]
        and freeze_audit["all_parameter_gradients_none"]
    )
    if not freeze_audit["passed"]:
        raise RuntimeError("ChordGNN +Post base freeze audit failed")
    _json_dump(config.output_dir / "base_freeze_audit.json", freeze_audit)
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
        "stage": "ChordGNN+Post",
        "base_seed": config.base_seed,
        "seed": config.seed,
        "stop_reason": "early_stopping" if stopped_early else "max_epochs",
        "early_stopped": stopped_early,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_counter_at_stop": epochs_without_improvement,
        "requested_epochs": config.epochs,
        "completed_epochs": len(history),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_monitored_loss": best_metrics["monitored_loss"],
        "best_validation_core_conjunction": best_metrics["core_conjunction"][
            "accuracy"
        ],
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _sha256(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": _sha256(last_path),
        "base_checkpoint": base_identity["checkpoint"],
        "base_checkpoint_sha256": base_identity["checkpoint_sha256"],
        "base_freeze_audit": str(
            (config.output_dir / "base_freeze_audit.json").resolve()
        ),
        "history": str(history_path.resolve()),
        "run_config": str((config.output_dir / "run_config.json").resolve()),
    }
    _json_dump(config.output_dir / "run_summary.json", summary)
    return summary
