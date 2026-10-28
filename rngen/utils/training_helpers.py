import atexit
import ast
import datetime
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch


def create_random_subset(dataset, max_samples, seed=42):
    """Create a random fixed subset of a dataset."""
    if len(dataset) <= max_samples:
        return dataset

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return torch.utils.data.Subset(dataset, indices)


def generate_wandb_run_name(model_args, data_args, training_args, is_eval=False):
    """Generate a descriptive WandB run name and update output_dir."""
    parts = model_args.llm_model_name.split("-")
    model_size = parts[-2] if parts[-1] in ("Instruct", "Base") else parts[-1]

    if model_args.freeze_layers_rnbert != "none":
        rnbert_status = (
            "F"
            if model_args.freeze_layers_rnbert == "all"
            else f"F{model_args.freeze_layers_rnbert}"
        )
    else:
        rnbert_status = "U"
    freeze_status = (
        f"RN{rnbert_status}_LLM{'F' if model_args.freeze_llm else 'U'}"
        f"_Proj{'F' if model_args.freeze_projector else 'U'}"
    )

    proj_info = f"L{model_args.projector_layers}_D{model_args.projector_dropout}"
    lr_info = f"lr{training_args.learning_rate:.0e}"
    bs_info = f"bs{training_args.per_device_train_batch_size}"
    epochs_info = f"ep{training_args.num_train_epochs}"

    lora_info = ""
    if hasattr(model_args, "use_lora") and model_args.use_lora:
        lora_info = f"_LoRA_r{model_args.lora_r}_a{model_args.lora_alpha}"
        if hasattr(model_args, "lora_dropout") and model_args.lora_dropout > 0:
            lora_info += f"_d{model_args.lora_dropout:.1f}"

        if (
            hasattr(model_args, "lora_target_modules")
            and model_args.lora_target_modules
        ):
            lora_info += f"_m{len(model_args.lora_target_modules)}"

    pretrained_info = ""
    if (
        hasattr(model_args, "load_from_fted_rnbert")
        and model_args.load_from_fted_rnbert
    ):
        pretrained_info = "_RN"
    if not getattr(model_args, "use_spelled_musicbert", False):
        pretrained_info += "_NA"

    run_name = (
        f"{model_size}_{freeze_status}_{proj_info}_{lr_info}_{bs_info}_{epochs_info}"
        f"{lora_info}{pretrained_info}"
    )
    if is_eval:
        run_name = "Eval_" + run_name

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{run_name}_{timestamp}"

    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    return run_name


def _expand_slurm_path_template(path_template, env):
    """Expand a small subset of Slurm filename tokens in a path template."""
    if not path_template:
        return None

    replacements = {
        "%": "%",
        "A": env.get("SLURM_ARRAY_JOB_ID", ""),
        "a": env.get("SLURM_ARRAY_TASK_ID", ""),
        "j": env.get("SLURM_JOB_ID", ""),
        "x": env.get("SLURM_JOB_NAME", ""),
    }

    def replace(match):
        token = match.group(1)
        if token not in replacements or replacements[token] == "":
            return match.group(0)
        return replacements[token]

    expanded = re.sub(r"%(.)", replace, path_template)
    expanded = os.path.expanduser(expanded)

    if not os.path.isabs(expanded):
        submit_dir = env.get("SLURM_SUBMIT_DIR") or os.getcwd()
        expanded = os.path.join(submit_dir, expanded)

    return os.path.abspath(expanded)


def _resolve_slurm_log_path(kind, env):
    explicit_var = f"RNGEN_SLURM_{kind.upper()}_PATH"
    explicit_path = env.get(explicit_var)
    if explicit_path:
        return _expand_slurm_path_template(explicit_path, env)

    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        return None

    job_name = env.get("SLURM_JOB_NAME")
    submit_dir = env.get("SLURM_SUBMIT_DIR") or os.getcwd()

    candidates = []
    if kind == "stdout":
        candidates.append(os.path.join(submit_dir, f"slurm-{job_id}.out"))
    if job_name:
        suffix = "out" if kind == "stdout" else "err"
        candidates.append(os.path.join(submit_dir, f"{job_name}-{job_id}.{suffix}"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(candidates[0]) if candidates else None


def _symlink_into_output_dir(target_path, link_path):
    """Create or refresh a symlink without overwriting regular files."""
    if not target_path:
        return False

    if os.path.lexists(link_path):
        if os.path.islink(link_path):
            current_target = os.readlink(link_path)
            if current_target == target_path:
                return True
            os.remove(link_path)
        else:
            return False

    os.symlink(target_path, link_path)
    return True


def _materialize_log_in_output_dir(source_path, destination_path):
    """Replace a symlink with a real file copy so run dirs stay self-contained."""
    if not source_path or not os.path.exists(source_path):
        return False

    if os.path.lexists(destination_path):
        os.remove(destination_path)

    shutil.copy2(source_path, destination_path)
    return True


def attach_slurm_logs(output_dir):
    """Write Slurm metadata and symlink logs into the run output directory."""
    env = dict(os.environ)
    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = _resolve_slurm_log_path("stdout", env)
    stderr_path = _resolve_slurm_log_path("stderr", env)

    stdout_link = output_dir / "slurm_stdout.out"
    stderr_link = output_dir / "slurm_stderr.err"

    stdout_linked = _symlink_into_output_dir(
        stdout_path,
        os.fspath(stdout_link),
    )
    stderr_linked = _symlink_into_output_dir(
        stderr_path,
        os.fspath(stderr_link),
    )

    metadata = {
        "job_id": job_id,
        "job_name": env.get("SLURM_JOB_NAME"),
        "array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env.get("SLURM_ARRAY_TASK_ID"),
        "submit_dir": env.get("SLURM_SUBMIT_DIR"),
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "stdout_link": os.fspath(stdout_link) if stdout_linked else None,
        "stderr_link": os.fspath(stderr_link) if stderr_linked else None,
    }

    metadata_path = output_dir / "slurm_job_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    def finalize_log_artifacts():
        if not output_dir.exists():
            return

        stdout_materialized = _materialize_log_in_output_dir(
            stdout_path,
            os.fspath(stdout_link),
        )
        stderr_materialized = _materialize_log_in_output_dir(
            stderr_path,
            os.fspath(stderr_link),
        )
        finalized_metadata = dict(metadata)
        finalized_metadata["stdout_link"] = (
            os.fspath(stdout_link) if stdout_materialized else metadata["stdout_link"]
        )
        finalized_metadata["stderr_link"] = (
            os.fspath(stderr_link) if stderr_materialized else metadata["stderr_link"]
        )
        finalized_metadata["stdout_materialized"] = stdout_materialized
        finalized_metadata["stderr_materialized"] = stderr_materialized
        try:
            with open(metadata_path, "w") as f:
                json.dump(finalized_metadata, f, indent=2, sort_keys=True)
        except FileNotFoundError:
            return

    atexit.register(finalize_log_artifacts)

    return metadata


def parse_freeze_layers(freeze_layers_str):
    """Convert a string representation into the freeze_layers config type."""
    if freeze_layers_str.lower() == "all":
        return "all"
    if freeze_layers_str.lower() == "none":
        return "none"

    try:
        return int(freeze_layers_str)
    except ValueError:
        try:
            value = ast.literal_eval(freeze_layers_str)
            if isinstance(value, list) and all(isinstance(x, int) for x in value):
                return value
            raise ValueError("List must contain only integers")
        except Exception as exc:
            raise ValueError(
                "freeze_layers_rnbert must be 'all', 'none', an integer, or a list of integers"
            ) from exc
