"""Trainer callbacks used by the RNGen SFT entrypoints.

The training flow uses ``WandBCallback`` from ``sft_ft.py`` to log examples.
Prediction snapshots are saved by the task metric modules.
"""

import json

import numpy as np
import wandb
from transformers import TrainerCallback


class WandBCallback(TrainerCallback):
    """Log sampled prediction/reference pairs to Weights & Biases."""

    def __init__(self, tokenizer, project_name=None, run_name=None):
        self.tokenizer = tokenizer
        self.project_name = project_name
        self.run_name = run_name

    def on_init_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if self.project_name:
            wandb.init(project=self.project_name, name=self.run_name)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if "eval_predictions" in kwargs and "eval_labels" in kwargs:
            predictions = kwargs["eval_predictions"]
            labels = kwargs["eval_labels"]

            # Sample a few examples for WandB
            sample_indices = np.random.choice(
                len(predictions), min(5, len(predictions)), replace=False
            )
            samples = []

            for idx in sample_indices:
                pred = self.tokenizer.decode(predictions[idx], skip_special_tokens=True)
                label_ids = [
                    token for token in np.asarray(labels[idx]).tolist() if token != -100
                ]
                label = self.tokenizer.decode(label_ids, skip_special_tokens=True)

                samples.append(
                    {
                        "prediction": pred,
                        "reference": label,
                        "is_valid_json_pred": self._is_valid_json(pred),
                        "is_valid_json_ref": self._is_valid_json(label),
                    }
                )

            wandb.log({"examples": wandb.Table(data=samples)}, step=state.global_step)

    def _is_valid_json(self, text):
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
