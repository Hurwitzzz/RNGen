"""Shared YAML config loading helpers for RNGen entrypoints."""

from pathlib import Path

import yaml
from omegaconf import OmegaConf


def _normalize_base_paths(base_value):
    if base_value is None:
        return []
    if isinstance(base_value, str):
        return [base_value]
    if isinstance(base_value, list):
        return base_value
    raise TypeError("_base_ must be a string or a list of strings.")


def load_yaml_config(config_path):
    """Load a YAML config with optional recursive `_base_` inheritance."""
    return _load_yaml_config(Path(config_path).resolve(), set())


def _load_yaml_config(config_path, seen_paths):
    if config_path in seen_paths:
        raise ValueError(f"Circular config inheritance detected for {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    seen_paths = seen_paths | {config_path}
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    base_paths = _normalize_base_paths(config_dict.pop("_base_", None))
    merged_config = OmegaConf.create()

    for base_path in base_paths:
        base_config_path = (config_path.parent / base_path).resolve()
        base_config = _load_yaml_config(base_config_path, seen_paths)
        merged_config = OmegaConf.merge(merged_config, OmegaConf.create(base_config))

    merged_config = OmegaConf.merge(merged_config, OmegaConf.create(config_dict))
    return OmegaConf.to_container(merged_config, resolve=True)
