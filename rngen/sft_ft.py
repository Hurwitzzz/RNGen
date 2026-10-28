#!/usr/bin/env python
"""Supervised fine-tuning entrypoint for RNGen.

This script loads model/data/training/generation arguments from a YAML config,
allows OmegaConf-style CLI overrides, builds the RNGen SFT datasets for the
requested task, and runs the custom training/evaluation loop defined in
``rngen.training``.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path

from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from omegaconf import OmegaConf
from rngen.cli.entrypoint_helpers import (
    get_model_compound_ratio,
    load_pretrained_rngen_checkpoint,
    resolve_training_device_map,
)
from rngen.cli.config_loading import load_yaml_config
from rngen.cli.annotation_paths import resolve_annotation_json_path
from rngen.cli.arguments import (
    ModelArguments,
    DataArguments,
    GenerationArguments,
    build_generation_kwargs,
    validate_task_name,
)

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rngen.modeling import create_rngen_model
from rngen.data import (
    RNGenSFTDataset,
    DataCollatorForRNGen,
)
from rngen.training.callbacks import WandBCallback
from rngen.training.sft_trainer import RNGenSFTTrainer
from rngen.training.task_registry import build_compute_metrics_fn
from rngen.utils import (
    attach_slurm_logs,
    create_random_subset,
    generate_wandb_run_name,
    parse_freeze_layers,
    log_model_parameter_stats,
)


def parse_args():
    """Parse command line arguments and load config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    args, remaining = parser.parse_known_args()

    config_dict = {}
    if args.config and os.path.exists(args.config):
        config_dict = load_yaml_config(args.config)

    # Parse remaining args with OmegaConf
    cli_conf = OmegaConf.from_dotlist(remaining)

    # Merge config sources with CLI overrides taking precedence
    config_dict = OmegaConf.merge(OmegaConf.create(config_dict), cli_conf)
    config_dict = OmegaConf.to_container(config_dict)

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, training_args, generation_args = parser.parse_dict(
        config_dict
    )
    data_args.task = validate_task_name(data_args.task)

    return model_args, data_args, training_args, generation_args, config_dict


def setup_logging(training_args):
    """Set up logging configuration."""
    is_main_process = training_args.local_rank in [-1, 0]
    handlers = [logging.NullHandler()]
    if is_main_process:
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, "train.log")),
        ]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if is_main_process else logging.ERROR,
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Log on each process the small summary:
    logger.info(f"Process rank: {training_args.local_rank}")
    logger.info(f"Distributed training: {bool(training_args.local_rank != -1)}")
    logger.info(f"Training/evaluation parameters {training_args}")

    return logger


def build_annotation_json_path(jsonl_dir, split, use_spelled_musicbert, task):
    """Build the annotation JSON path for a train/valid/test split."""
    return resolve_annotation_json_path(jsonl_dir, split, use_spelled_musicbert)


def create_sft_dataset(
    *,
    split,
    jsonl_dir,
    use_spelled_musicbert,
    tokenizer,
    data_args,
    max_text_length,
    compound_ratio,
    include_spelling_in_prompt,
    is_eval,
):
    """Create one RNGen SFT dataset for the requested split."""
    return RNGenSFTDataset(
        json_path=build_annotation_json_path(
            jsonl_dir,
            split,
            use_spelled_musicbert,
            data_args.task,
        ),
        tokenizer=tokenizer,
        data_args=data_args,
        max_text_length=max_text_length,
        compound_ratio=compound_ratio,
        is_eval=is_eval,
        include_spelling_in_prompt=include_spelling_in_prompt,
    )


def main():
    """Main training function."""
    # Parse arguments
    model_args, data_args, training_args, generation_args, config_dict = parse_args()
    model_args.freeze_layers_rnbert = parse_freeze_layers(
        model_args.freeze_layers_rnbert
    )  # Do this because HF.Argparser does not support "Optional[Union[int, List[int], Literal["all", "none"]]]"

    training_args.run_name = generate_wandb_run_name(
        model_args, data_args, training_args
    )
    is_main_process = training_args.local_rank in [-1, 0]
    os.makedirs(training_args.output_dir, exist_ok=True)
    slurm_info = (
        attach_slurm_logs(training_args.output_dir) if is_main_process else None
    )

    # Save config to output directory
    if is_main_process:
        with open(
            os.path.join(training_args.output_dir, "training_config.yaml"), "w"
        ) as f:
            yaml.dump(config_dict, f)

    logger = setup_logging(training_args)
    if slurm_info is not None:
        logger.info(
            "Attached Slurm logs for job %s into %s",
            slurm_info["job_id"],
            training_args.output_dir,
        )

    set_seed(training_args.seed)
    device_map = resolve_training_device_map()

    # Create model and tokenizer
    logger.info(f"Creating RNGen")
    if model_args.finetuned_rngen_checkpoint:
        model, tokenizer = load_pretrained_rngen_checkpoint(
            model_path=model_args.finetuned_rngen_checkpoint,
            is_lora_trainable=True,
            use_spelled_musicbert=model_args.use_spelled_musicbert,
            spelling_tokens=model_args.spelling_tokens,
            spelling_downsampling_scale=model_args.spelling_downsampling_scale,
            device_map=device_map,
        )
    else:
        model, tokenizer = create_rngen_model(
            rnbert_checkpoint=model_args.rnbert_checkpoint_path,
            llm_name=model_args.llm_model_name,
            device_map=device_map,
            freeze_layers_rnbert=model_args.freeze_layers_rnbert,
            freeze_llm=model_args.freeze_llm,
            projector_layers=model_args.projector_layers,
            projector_dropout=model_args.projector_dropout,
            use_spelled_musicbert=model_args.use_spelled_musicbert,
            spelling_tokens=model_args.spelling_tokens,
            spelling_downsampling_scale=model_args.spelling_downsampling_scale,
            use_lora=model_args.use_lora,
            lora_r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            lora_target_modules=model_args.lora_target_modules,
        )

    # Freeze or unfreeze components based on arguments
    if model_args.freeze_projector:
        for param in model.projector.parameters():
            param.requires_grad = False

    # Create datasets
    jsonl_dir = data_args.jsonl_dir
    logger.info(f"Creating datasets from: {jsonl_dir}")
    compound_ratio = get_model_compound_ratio(model)
    train_dataset = create_sft_dataset(
        split="train",
        jsonl_dir=jsonl_dir,
        use_spelled_musicbert=model_args.use_spelled_musicbert,
        tokenizer=tokenizer,
        data_args=data_args,
        max_text_length=data_args.max_text_length,
        compound_ratio=compound_ratio,
        include_spelling_in_prompt=model_args.include_spelling_in_prompt,
        is_eval=False,
    )

    eval_dataset = create_sft_dataset(
        split="valid",
        jsonl_dir=jsonl_dir,
        use_spelled_musicbert=model_args.use_spelled_musicbert,
        tokenizer=tokenizer,
        data_args=data_args,
        max_text_length=data_args.max_text_length,
        compound_ratio=compound_ratio,
        include_spelling_in_prompt=model_args.include_spelling_in_prompt,
        is_eval=True,
    )

    # Used for debugging
    if (
        hasattr(data_args, "max_eval_samples")
        and data_args.max_eval_samples is not None
    ):
        logger.info(f"Limiting eval dataset to {data_args.max_eval_samples} examples")
        eval_dataset = create_random_subset(eval_dataset, data_args.max_eval_samples)

    # Data collator
    data_collator = DataCollatorForRNGen(
        tokenizer=tokenizer,
        llm_pad_token_id=tokenizer.pad_token_id,
        musicbert_pad_token_id=1,  # 1 is the token_id for <pad> in musicbert
    )

    generate_kwargs = build_generation_kwargs(generation_args)

    metrics_output_dir = os.path.join(Path(training_args.output_dir), "predictions")
    logger.info(f"Setting up compute_metrics for task: {data_args.task}")
    compute_metrics_fn = build_compute_metrics_fn(
        task=data_args.task,
        tokenizer=tokenizer,
        compound_ratio=compound_ratio,
        output_dir=metrics_output_dir,
    )

    callbacks = []
    if model_args.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        )
    if is_main_process:
        callbacks.append(
            WandBCallback(
                tokenizer=tokenizer,
                project_name=model_args.wandb_project,
                run_name=training_args.run_name,
            )
        )

    # Initialize Trainer
    trainer = RNGenSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        generate_kwargs=generate_kwargs,
    )

    # Log parameter statistics
    log_model_parameter_stats(model, logger)
    log_model_parameter_stats(model.llm, logger)

    # Train model
    logger.info("Starting training")
    if training_args.do_train is True:
        train_result = trainer.train()

    # Create test dataset and run evaluation
    test_dataset = create_sft_dataset(
        split="test",
        jsonl_dir=jsonl_dir,
        use_spelled_musicbert=model_args.use_spelled_musicbert,
        tokenizer=tokenizer,
        data_args=data_args,
        max_text_length=data_args.max_text_length,
        compound_ratio=compound_ratio,
        include_spelling_in_prompt=model_args.include_spelling_in_prompt,
        is_eval=True,
    )
    # logger.info(f"Limiting test dataset to 50 examples")
    # test_dataset = create_random_subset(test_dataset, 10)

    logger.info("Running evaluation on test set")
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    # Log and save metrics
    logger.info(f"Test metrics: {metrics}")
    trainer.save_metrics("test", metrics)

    logger.info(f"Training completed. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
