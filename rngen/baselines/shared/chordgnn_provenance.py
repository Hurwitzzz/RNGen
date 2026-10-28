"""Shared immutable provenance records for controlled ChordGNN commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "partitura",
    "music21",
    "scipy",
    "torch-scatter",
)


def _command(arguments: Sequence[str]) -> str:
    return subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    commit = _command(("git", "-C", str(root), "rev-parse", "HEAD"))
    status = _command(("git", "-C", str(root), "status", "--short"))
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "available": True,
        "repository": str(root),
        "commit": commit,
        "branch": _command(
            ("git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD")
        ),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def environment_provenance() -> dict[str, Any]:
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "temporary_paths": {
            name: os.environ.get(name)
            for name in (
                "TMPDIR",
                "PIP_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }
    try:
        import torch
    except ImportError:
        payload["torch"] = None
    else:
        payload["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
            }
    return payload


def full_provenance(*, require_git: bool = True) -> dict[str, Any]:
    """Record the environment, optionally accepting a checkout without Git."""
    try:
        git = git_provenance()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if require_git:
            raise
        git = {
            "available": False,
            "commit": None,
            "dirty": None,
            "branch": None,
            "status_short": [],
            "working_tree_diff_sha256": None,
        }
    return {
        "git": git,
        "environment": environment_provenance(),
    }


def training_config_fingerprint(
    training: dict[str, Any], *, output_dir: Path, resume: bool
) -> str:
    """Identify training settings independently of incidental Git state.

    A matching resumed run retains its saved identifier, including identifiers
    from older checkpoints that incorporated Git metadata. Checkpoint consumers
    can therefore continue checking the original config/checkpoint pair.
    """
    values = dict(training)
    values.pop("resume", None)
    normalized = json.loads(json.dumps(values, sort_keys=True, default=str))
    if resume:
        saved = json.loads((output_dir / "run_config.json").read_text())
        previous = dict(saved["training"])
        previous.pop("resume", None)
        if previous != normalized:
            raise ValueError("Resume training settings differ from the saved run config")
        return saved["config_sha256"]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
