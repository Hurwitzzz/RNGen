"""Translate public baseline YAML parameters to the existing training CLIs.

Run in the selected baseline's environment (requires PyYAML). No model or
framework is imported by this launcher before starting the training process.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "augnet_mi20": "rngen.baselines.augnet_mi20.train_augmentednet_baseline",
    "chordgnn": "rngen.baselines.chordgnn.train_chordgnn_baseline",
    "chordgnn_post": "rngen.baselines.chordgnn_post.train_chordgnn_post",
}


def load_parameters(path, overrides):
    with path.open(encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle)
    if not isinstance(parameters, dict):
        raise ValueError("The baseline YAML must contain a parameter mapping.")
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or key not in parameters:
            raise ValueError("Expected an existing config key followed by =value: " + override)
        parameters[key] = yaml.safe_load(value)
    return parameters


def build_command(baseline, parameters, dry_run=False):
    parameters = dict(parameters)
    if not dry_run:
        pending = [key for key, value in parameters.items() if "/path/to/" in str(value)]
        if pending:
            raise ValueError("Replace path placeholders before training: " + ", ".join(pending))
    if baseline == "augnet_mi20":
        # Expand NPZ manifests into the trainer's nargs='+' arguments.
        for split in ("train", "validation"):
            manifest = Path(parameters.pop(split + "_manifest")).expanduser()
            if dry_run and not manifest.is_file():
                paths = ["<NPZ entries from " + str(manifest) + ">"]
            else:
                paths = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
                if not paths:
                    raise ValueError("Empty NPZ manifest: " + str(manifest))
            parameters[split + "_npz"] = paths

    command = [sys.executable, "-m", MODULES[baseline]]
    for key, value in parameters.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            if value:
                command.extend([flag] + [str(item) for item in value])
        else:
            command.extend([flag, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=MODULES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting training.")
    parser.add_argument("overrides", nargs="*", help="Config overrides such as seed=43 or architecture=Micchi2020.")
    args = parser.parse_intermixed_args()
    config = args.config or REPO_ROOT / "configs" / args.baseline / "train.yaml"
    try:
        command = build_command(args.baseline, load_parameters(config, args.overrides), args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        if args.baseline.startswith("chordgnn"):
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Resolve relative dataset paths from the repository root.
        raise SystemExit(subprocess.call(command, cwd=REPO_ROOT, env=env))


if __name__ == "__main__":
    main()
