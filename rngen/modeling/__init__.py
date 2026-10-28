"""RNGen model package exports.

The package keeps its PyTorch-dependent public exports lazy for lightweight
inspection. Baseline implementations live in :mod:`rngen.baselines`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "RNBertProjector": ("components", "RNBertProjector", False),
    "RNGenModel": ("core", "RNGenModel", True),
    "create_rngen_model": ("factory", "create_rngen_model", True),
    "_save_weights": ("io", "_save_weights", False),
    "_load_weights_from_path": ("io", "_load_weights_from_path", False),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name, optional = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    try:
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    except ModuleNotFoundError:
        if not optional:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
