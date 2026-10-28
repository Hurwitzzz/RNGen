"""Lazy compatibility exports for the :mod:`rngen.utils` package.

PyTorch-dependent helpers stay unloaded so the controlled TensorFlow baseline
runtime can import the RNA parser and metric repair helper independently.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "repair_and_decode_json": ("metrics_utils", "repair_and_decode_json"),
    "parse_gt_rnbert": ("rna", "parse_gt_rnbert"),
    "log_model_parameter_stats": (
        "model_debug",
        "log_model_parameter_stats",
    ),
    "attach_slurm_logs": ("training_helpers", "attach_slurm_logs"),
    "create_random_subset": ("training_helpers", "create_random_subset"),
    "generate_wandb_run_name": (
        "training_helpers",
        "generate_wandb_run_name",
    ),
    "parse_freeze_layers": ("training_helpers", "parse_freeze_layers"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
