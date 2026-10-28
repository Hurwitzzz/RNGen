#!/usr/bin/env python3
"""Manual evaluation entrypoint for a fine-tuned RNGen checkpoint.

This script supports either direct CLI flags or a YAML config file for post-hoc
evaluation of an already trained checkpoint.
"""

import argparse
import os
from types import SimpleNamespace

from omegaconf import OmegaConf
from transformers import TrainingArguments

from rngen.cli.arguments import TASK_CHOICES, build_generation_kwargs
from rngen.cli.config_loading import load_yaml_config
from rngen.cli.annotation_paths import resolve_annotation_json_path
from rngen.data import DataCollatorForRNGen, RNGenSFTDataset
from rngen.cli.entrypoint_helpers import (
    get_model_compound_ratio,
    load_pretrained_rngen_checkpoint,
)
from rngen.training.sft_trainer import RNGenSFTTrainer
from rngen.training.task_registry import build_compute_metrics_fn
from rngen.utils import attach_slurm_logs

_REQUIRED_INCLUDE_FOR_METRICS = ("loss", "metadata")


def build_default_eval_json_path(config_dict):
    """Infer the test annotation JSON path from a training-style config."""
    json_path = config_dict.get("json_path")
    if json_path:
        return json_path

    jsonl_dir = config_dict.get("jsonl_dir")
    if not jsonl_dir:
        return None

    use_spelled_musicbert = config_dict.get("use_spelled_musicbert", False)
    return resolve_annotation_json_path(jsonl_dir, "test", use_spelled_musicbert)


def load_config_dict(config_path):
    """Load a YAML config and return it as a plain dictionary."""
    config_dict = load_yaml_config(config_path)
    return OmegaConf.to_container(OmegaConf.create(config_dict), resolve=True)


def resolve_include_for_metrics(config_dict):
    """Normalize post-hoc metric fields and keep metadata enabled by default."""
    include_for_metrics = config_dict.get("include_for_metrics")
    if include_for_metrics is None:
        resolved = []
    elif isinstance(include_for_metrics, str):
        resolved = [include_for_metrics]
    else:
        resolved = list(include_for_metrics)

    for required_field in _REQUIRED_INCLUDE_FOR_METRICS:
        if required_field not in resolved:
            resolved.append(required_field)

    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned RNGen model.")
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Optional YAML config for post-hoc evaluation.",
    )
    parser.add_argument(
        "--model-path",
        default=argparse.SUPPRESS,
        help="Path to the checkpoint.",
    )
    parser.add_argument(
        "--json-path",
        default=argparse.SUPPRESS,
        help="Path to the evaluation annotation JSON file.",
    )
    parser.add_argument(
        "--task",
        default=argparse.SUPPRESS,
        choices=TASK_CHOICES,
        help="Task name passed through to the dataset and metric setup.",
    )
    parser.add_argument(
        "--output-dir",
        default=argparse.SUPPRESS,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum text length used by the dataset.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--include-spelling-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Include spelling tokens in the prompt when the dataset provides them.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use sampling for generation instead of deterministic decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS,
        help="Sampling temperature used when --do-sample is enabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=argparse.SUPPRESS,
        help="Top-p nucleus sampling threshold used when --do-sample is enabled.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=argparse.SUPPRESS,
        help="Repetition penalty passed to generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=argparse.SUPPRESS,
        help="Beam width for generation.",
    )
    parser.add_argument(
        "--use-spelled-musicbert",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Load the checkpoint with spelled MusicBert support.",
    )
    parser.add_argument(
        "--spelling-tokens",
        nargs="*",
        default=argparse.SUPPRESS,
        help="Optional spelling tokens to pass through when loading a spelled checkpoint.",
    )
    parser.add_argument(
        "--spelling-downsampling-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Scale applied to the ninth Octuple+ downsampling slice after loading a spelled checkpoint.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use bf16 for evaluation when supported.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable gradient checkpointing in the evaluation trainer.",
    )

    raw_args = parser.parse_args()
    config_dict = (
        load_config_dict(raw_args.config) if hasattr(raw_args, "config") else {}
    )

    resolved_args = {
        "model_path": getattr(
            raw_args,
            "model_path",
            config_dict.get("model_path")
            or config_dict.get("finetuned_rngen_checkpoint"),
        ),
        "json_path": getattr(
            raw_args,
            "json_path",
            build_default_eval_json_path(config_dict),
        ),
        "task": getattr(raw_args, "task", config_dict.get("task", "key_sft")),
        "output_dir": getattr(
            raw_args,
            "output_dir",
            config_dict.get("output_dir", "rngen_eval_outputs"),
        ),
        "max_text_length": getattr(
            raw_args,
            "max_text_length",
            config_dict.get("max_text_length", 1024),
        ),
        "per_device_eval_batch_size": getattr(
            raw_args,
            "per_device_eval_batch_size",
            config_dict.get("per_device_eval_batch_size", 1),
        ),
        "include_spelling_in_prompt": getattr(
            raw_args,
            "include_spelling_in_prompt",
            config_dict.get("include_spelling_in_prompt", False),
        ),
        "do_sample": getattr(
            raw_args, "do_sample", config_dict.get("do_sample", False)
        ),
        "temperature": getattr(
            raw_args, "temperature", config_dict.get("temperature", 1.0)
        ),
        "top_p": getattr(raw_args, "top_p", config_dict.get("top_p", 1.0)),
        "repetition_penalty": getattr(
            raw_args,
            "repetition_penalty",
            config_dict.get("repetition_penalty", 1.0),
        ),
        "max_new_tokens": getattr(
            raw_args,
            "max_new_tokens",
            config_dict.get("max_new_tokens", 256),
        ),
        "num_beams": getattr(raw_args, "num_beams", config_dict.get("num_beams", 4)),
        "use_spelled_musicbert": getattr(
            raw_args,
            "use_spelled_musicbert",
            config_dict.get("use_spelled_musicbert", False),
        ),
        "spelling_tokens": getattr(
            raw_args,
            "spelling_tokens",
            config_dict.get("spelling_tokens"),
        ),
        "spelling_downsampling_scale": getattr(
            raw_args,
            "spelling_downsampling_scale",
            config_dict.get("spelling_downsampling_scale", 1.0),
        ),
        "bf16": getattr(raw_args, "bf16", config_dict.get("bf16", True)),
        "gradient_checkpointing": getattr(
            raw_args,
            "gradient_checkpointing",
            config_dict.get("gradient_checkpointing", True),
        ),
        "include_for_metrics": resolve_include_for_metrics(config_dict),
    }

    missing_args = [
        name for name in ("model_path", "json_path") if not resolved_args[name]
    ]
    if missing_args:
        parser.error(
            "the following arguments are required after resolving config values: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing_args)
        )

    return SimpleNamespace(**resolved_args)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    slurm_info = attach_slurm_logs(args.output_dir)
    if slurm_info is not None:
        print(
            f"Attached Slurm logs for job {slurm_info['job_id']} into {args.output_dir}"
        )

    print(f"Loading RNGenModel from: {args.model_path}")
    rngen_model, tokenizer = load_pretrained_rngen_checkpoint(
        model_path=args.model_path,
        is_lora_trainable=False,
        use_spelled_musicbert=args.use_spelled_musicbert,
        spelling_tokens=args.spelling_tokens,
        spelling_downsampling_scale=args.spelling_downsampling_scale,
    )
    rngen_model.eval()
    print("Model and tokenizer loaded successfully.")

    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON annotations file not found: {args.json_path}")

    data_args = SimpleNamespace(task=args.task)

    eval_dataset = RNGenSFTDataset(
        json_path=args.json_path,
        tokenizer=tokenizer,
        data_args=data_args,
        max_text_length=args.max_text_length,
        compound_ratio=get_model_compound_ratio(rngen_model),
        is_eval=True,
        include_spelling_in_prompt=args.include_spelling_in_prompt,
    )

    data_collator = DataCollatorForRNGen(
        tokenizer=tokenizer,
        llm_pad_token_id=tokenizer.pad_token_id,
        musicbert_pad_token_id=1,
    )

    generate_kwargs = build_generation_kwargs(args)
    compound_ratio = get_model_compound_ratio(rngen_model)

    compute_metrics = build_compute_metrics_fn(
        task=args.task,
        tokenizer=tokenizer,
        compound_ratio=compound_ratio,
        output_dir=args.output_dir,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        include_for_metrics=args.include_for_metrics,
        # Loss/metadata metrics (Mixed) do not consume generated text.
        prediction_loss_only=bool(
            getattr(compute_metrics, "supports_loss_metadata_metrics", False)
        ),
        report_to="none",
    )

    trainer = RNGenSFTTrainer(
        model=rngen_model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        generate_kwargs=generate_kwargs,
    )

    metrics = trainer.evaluate(eval_dataset)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
