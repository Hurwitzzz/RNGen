"""Metrics for RNA Stage 2 fine-grained curriculum SFT."""

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

TASKS = (
    "harmony_span_prediction",
    "chord_tone_inventory_prediction",
    "tone_factor_pair_prediction",
    "quality_prediction",
    "inversion_figure_prediction",
    "ps_function_prediction",
)


def _flatten_losses(losses: Any) -> np.ndarray:
    if losses is None:
        return np.array([], dtype=np.float64)
    if isinstance(losses, list):
        if not losses:
            return np.array([], dtype=np.float64)
        return np.concatenate([np.asarray(loss).reshape(-1) for loss in losses])
    return np.asarray(losses, dtype=np.float64).reshape(-1)


def _safe_mean(values: Iterable[float]) -> float:
    value_list = [float(value) for value in values if np.isfinite(value)]
    if not value_list:
        return float("nan")
    return float(np.mean(value_list))


def create_compute_rna_curriculum_stage2_metrics_fn(
    tokenizer,
    compound_ratio: int,
    output_dir: str,
):
    """Create a compute_metrics callable for Stage 2 curriculum evaluation."""

    del tokenizer, compound_ratio
    os.makedirs(output_dir, exist_ok=True)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        losses = _flatten_losses(getattr(eval_pred, "losses", None))
        metadata_list: List[Dict[str, Any]] = list(
            getattr(eval_pred, "metadata_list", None) or []
        )

        metrics: Dict[str, float] = {}
        task_losses: Dict[str, List[float]] = defaultdict(list)
        unknown_task_count = 0

        for idx, metadata in enumerate(metadata_list):
            if idx >= len(losses):
                break
            task_name = (
                metadata.get("task_name") if isinstance(metadata, dict) else None
            )
            if not task_name:
                unknown_task_count += 1
                continue
            task_losses[str(task_name)].append(float(losses[idx]))

        per_task_means = []
        summary = {
            "loss_count": int(len(losses)),
            "metadata_count": int(len(metadata_list)),
            "unknown_task_count": int(unknown_task_count),
            "tasks": {},
            "loss_note": (
                "Per-task losses are computed from the trainer-provided eval losses. "
                "In the current trainer these are batch losses repeated for each "
                "example in the batch. With evaluation batch size 1, these are "
                "individual example losses. With larger batches, per-task means "
                "are proxies rather than exact per-example loss aggregates."
            ),
        }

        for task_name in TASKS:
            values = task_losses.get(task_name, [])
            task_loss = _safe_mean(values)
            metrics[f"loss_{task_name}"] = task_loss
            metrics[f"count_{task_name}"] = len(values)
            summary["tasks"][task_name] = {
                "count": len(values),
                "loss": task_loss,
            }
            if values:
                per_task_means.append(task_loss)

        metrics["stage2_selection_loss"] = _safe_mean(per_task_means)
        metrics["loss_metadata_count"] = len(metadata_list)
        metrics["loss_value_count"] = int(len(losses))
        metrics["unknown_task_count"] = unknown_task_count

        summary["stage2_selection_loss"] = metrics["stage2_selection_loss"]
        with open(os.path.join(output_dir, "stage2_eval_loss_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return metrics

    compute_metrics.supports_loss_metadata_metrics = True
    return compute_metrics
