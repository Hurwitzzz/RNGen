#!/usr/bin/env python
"""Legacy standalone entrypoint for Octuple+ spelled-pitch warm-up.

This module is retained for backward compatibility and is not part of the
default RNGen training pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from rngen.legacy.spell_warmup_data import (
    CANONICAL_SPELLED_PITCH_CLASSES,
    DEFAULT_SPELLING_TOKENS,
    MUSICBERT_PAD_TOKEN_ID,
    DataCollatorForSpellWarmup,
    SpellWarmupDataset,
    load_event_vocab,
)
from rngen.legacy.spelled_pitch_warmup import SpelledPitchWarmupModel
from rngen.utils import attach_slurm_logs


@dataclass
class SpellWarmupModelArguments:
    rnbert_checkpoint_path: str = field(
        metadata={
            "help": "Path to the unconditioned RNBERT checkpoint used to initialize the warm-up encoder."
        }
    )
    spelling_tokens: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Optional spelled-token list. Defaults to seven accidental tokens, including triple sharps and flats."
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Weights & Biases project name for this run."},
    )
    trainable_scope: str = field(
        default="spelling_path",
        metadata={
            "help": "Which parameters to train: 'spelling_path' or 'all_encoder'."
        },
    )


@dataclass
class SpellWarmupDataArguments:
    jsonl_dir: str = field(
        metadata={
            "help": "Directory containing train.json, valid.json, and optional test.json files."
        }
    )
    annotation_suffix: str = field(
        default=".json",
        metadata={
            "help": "Suffix appended to split names when locating warm-up annotation JSON files."
        },
    )
    max_notes: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional note-level truncation cap after compound downsampling."
        },
    )


def parse_args():
    from omegaconf import OmegaConf
    from rngen.cli.config_loading import load_yaml_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args, remaining = parser.parse_known_args()

    config_dict = {}
    if args.config and os.path.exists(args.config):
        config_dict = load_yaml_config(args.config)

    cli_conf = OmegaConf.from_dotlist(remaining)
    config_dict = OmegaConf.merge(OmegaConf.create(config_dict), cli_conf)
    config_dict = OmegaConf.to_container(config_dict)

    parser = HfArgumentParser(
        (SpellWarmupModelArguments, SpellWarmupDataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_dict(config_dict)
    return model_args, data_args, training_args, config_dict


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, "train.log")),
        ],
    )
    return logging.getLogger(__name__)


def build_annotation_json_path(jsonl_dir: str, split: str, suffix: str) -> str:
    return os.path.join(jsonl_dir, f"{split}{suffix}")


def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = predictions.argmax(axis=-1)
    mask = labels != -100
    total = mask.sum()
    if total == 0:
        return {"accuracy": 0.0}
    correct = (pred_ids[mask] == labels[mask]).sum()
    return {"accuracy": float(correct / total)}


class SpellWarmupTrainer(Trainer):
    """Trainer that checkpoints only the warmed-up encoder."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )

        encoder_output_dir = os.path.join(output_dir, "encoder")
        self.model.save_encoder(encoder_output_dir)
        with open(os.path.join(output_dir, "spelled_pitch_labels.json"), "w") as f:
            json.dump(CANONICAL_SPELLED_PITCH_CLASSES, f, indent=2)
        with open(
            os.path.join(encoder_output_dir, "spelled_pitch_labels.json"), "w"
        ) as f:
            json.dump(CANONICAL_SPELLED_PITCH_CLASSES, f, indent=2)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch_save_path = os.path.join(output_dir, "training_args.bin")
        torch.save(self.args, torch_save_path)


def main():
    model_args, data_args, training_args, config_dict = parse_args()
    os.makedirs(training_args.output_dir, exist_ok=True)
    slurm_info = attach_slurm_logs(training_args.output_dir)

    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project

    with open(os.path.join(training_args.output_dir, "training_config.yaml"), "w") as f:
        yaml.dump(config_dict, f)

    logger = setup_logging(training_args)
    if slurm_info is not None:
        logger.info(
            "Attached Slurm logs for job %s into %s",
            slurm_info["job_id"],
            training_args.output_dir,
        )

    set_seed(training_args.seed)

    spelling_tokens = model_args.spelling_tokens or DEFAULT_SPELLING_TOKENS
    event_vocab = load_event_vocab(spelling_tokens)
    spelling_token_ids = [event_vocab[token] for token in spelling_tokens]
    model = SpelledPitchWarmupModel(
        model_args.rnbert_checkpoint_path,
        spelling_tokens=spelling_tokens,
        spelling_token_ids=spelling_token_ids,
        vocab_size=max(event_vocab.values()) + 1,
        num_labels=len(CANONICAL_SPELLED_PITCH_CLASSES),
        trainable_scope=model_args.trainable_scope,
    )
    encoder_embeddings = model.rnbert.bert.embeddings
    logger.info(
        "Spell warm-up vocab sizes: event_vocab_size=%d, max_event_token_id=%d, "
        "encoder_word_embeddings=%d, encoder_position_embeddings=%d",
        max(event_vocab.values()) + 1,
        max(event_vocab.values()),
        model.rnbert.get_input_embeddings().num_embeddings,
        encoder_embeddings.position_embeddings.num_embeddings,
    )

    train_dataset = SpellWarmupDataset(
        build_annotation_json_path(
            data_args.jsonl_dir, "train", data_args.annotation_suffix
        ),
        max_notes=data_args.max_notes,
        event_vocab=event_vocab,
        is_eval=False,
    )
    eval_dataset = SpellWarmupDataset(
        build_annotation_json_path(
            data_args.jsonl_dir, "valid", data_args.annotation_suffix
        ),
        max_notes=data_args.max_notes,
        event_vocab=event_vocab,
        is_eval=False,
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
    ]

    trainer = SpellWarmupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSpellWarmup(MUSICBERT_PAD_TOKEN_ID),
        compute_metrics=compute_accuracy,
        callbacks=callbacks,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_state()

        encoder_save_dir = os.path.join(training_args.output_dir, "best_encoder")
        trainer.model.save_encoder(encoder_save_dir)
        with open(
            os.path.join(encoder_save_dir, "spelled_pitch_labels.json"), "w"
        ) as f:
            json.dump(CANONICAL_SPELLED_PITCH_CLASSES, f, indent=2)
        logger.info("Saved best warm-up encoder to %s", encoder_save_dir)
        trainer.remove_callback(EarlyStoppingCallback)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        test_path = build_annotation_json_path(
            data_args.jsonl_dir, "test", data_args.annotation_suffix
        )
        if os.path.exists(test_path):
            test_dataset = SpellWarmupDataset(
                test_path,
                max_notes=data_args.max_notes,
                event_vocab=event_vocab,
                is_eval=False,
            )
            test_metrics = trainer.evaluate(
                eval_dataset=test_dataset, metric_key_prefix="test"
            )
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
