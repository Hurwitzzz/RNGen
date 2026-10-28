"""Custom Hugging Face trainer used by RNGen supervised fine-tuning flows."""

import logging
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from rngen.modeling import RNGenModel
from rngen.cli.entrypoint_helpers import resolve_training_device_map
from transformers import Trainer
from transformers.trainer_utils import denumpify_detensorize, has_length
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    find_batch_size,
    IterableDatasetShard,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.utils import is_torch_xla_available
from transformers.integrations.deepspeed import deepspeed_init

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)


def _supports_loss_metadata_metrics(compute_metrics):
    return bool(getattr(compute_metrics, "supports_loss_metadata_metrics", False))


class RNGenSFTTrainer(Trainer):
    def __init__(self, *args, generate_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_kwargs = generate_kwargs or {}

    def _gather_batch_metadata(self, inputs):
        batch_metadata = inputs.get("batch_metadata")
        if batch_metadata is None:
            return []

        gathered_parts = self.accelerator.gather_for_metrics(
            batch_metadata,
            use_gather_object=True,
        )

        metadata_list = []
        for part in gathered_parts:
            if isinstance(part, dict):
                metadata_list.append(part)
            elif isinstance(part, (list, tuple)):
                metadata_list.extend(part)
            else:
                raise TypeError(
                    f"Unexpected gathered metadata type: {type(part).__name__}"
                )
        return metadata_list

    def _get_num_samples(self, eval_dataset, dataloader, observed_num_examples):
        if has_length(eval_dataset):
            return len(eval_dataset)

        if (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            return eval_dataset.num_examples

        if has_length(dataloader):
            return self.num_examples(dataloader)

        return observed_num_examples

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        metric_key_prefix,  # (Hewei) We want to compute loss in validation, but generate outputs in test.
        ignore_keys=None,
    ):
        """Override to support generation during evaluation."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        loss_outputs = None

        # Compute loss with labels first
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, loss_outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()
            else:
                loss = None

        # For prediction_loss_only just return the loss
        if prediction_loss_only and metric_key_prefix != "test":
            return (loss, None, None)

        # Generate predictions if requested
        if metric_key_prefix == "test" or (prediction_loss_only is False):
            # Get generation settings
            gen_kwargs = self.generate_kwargs.copy()

            # Ensure special handling for RNGenModel
            with torch.no_grad():
                if "music_input_ids" in inputs:
                    # RNGen model needs music inputs
                    generated_tokens = model.generate(
                        music_input_ids=inputs["music_input_ids"],
                        music_attention_mask=inputs.get("music_attention_mask", None),
                        prompt_input_ids=inputs["eval_input_ids"],
                        prompt_attention_mask=inputs.get("eval_attention_mask", None),
                        **gen_kwargs,
                    )
                else:
                    raise ValueError(
                        "RNGenSFTTrainer expects music_input_ids in the inputs"
                    )

            # Get labels for metrics computation
            labels = inputs.get("labels") if has_labels else None

            return (loss, generated_tokens, labels)

        # (Hewei 2025-03-11) TODO: we should not compute "outputs" again.
        # Default behavior: return logits without running a second forward pass when
        # compute_loss already gave us model outputs.
        if loss_outputs is None:
            with torch.no_grad():
                loss_outputs = model(**inputs)

        return (loss, loss_outputs.logits, inputs.get("labels"))

    def _load_best_model(self):
        """Override to use RNGenModel.from_pretrained for proper component loading."""
        if self.state.best_model_checkpoint is None:
            logger.warning(
                "No best model checkpoint found, skipping best model loading."
            )
            return

        logger.info(
            f"Loading best RNGen model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )

        # Store current model state
        was_training = self.model.training
        use_spelled_musicbert = self.model.config.use_spelled_musicbert

        # ``device_map="auto"`` is only safe for single-process loading. During
        # DDP reloads each process should load a normal local replica.
        device_map = resolve_training_device_map(
            self.accelerator.device_map
            if hasattr(self.accelerator, "device_map")
            else "auto"
        )

        # (Hewei 2025-08-01) Explicitly delete the old model and clear the cache to free VRAM
        del self.model
        if hasattr(self, "model_wrapped"):
            del self.model_wrapped
        torch.cuda.empty_cache()

        # Reload
        self.model = RNGenModel.from_pretrained(
            self.state.best_model_checkpoint,
            use_spelled_musicbert=use_spelled_musicbert,
            is_lora_trainable=False,
            device_map=device_map,
        )

        # Restore training state
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        # Update wrapper reference
        self.model_wrapped = self.model

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_preds = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_labels = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_inputs = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_metadata_list = []

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            all_metadata_list.extend(self._gather_batch_metadata(inputs))

            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name])
                if "inputs" in args.include_for_metrics
                else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100
                )
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            if self.args.batch_eval_metrics:
                if (
                    self.compute_metrics is not None
                    and logits is not None
                    and labels is not None
                ):
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = (
                        losses if "loss" in args.include_for_metrics else None
                    )
                    batch_kwargs["inputs"] = (
                        inputs if "inputs" in args.include_for_metrics else None
                    )
                    metrics = self.compute_metrics(
                        EvalPrediction(
                            predictions=logits, label_ids=labels, **batch_kwargs
                        ),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        num_samples = self._get_num_samples(
            eval_dataset,
            dataloader,
            observed_num_examples,
        )
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        has_prediction_metrics = all_preds is not None and all_labels is not None
        has_loss_metadata_metrics = (
            all_losses is not None
            and "loss" in args.include_for_metrics
            and "metadata" in args.include_for_metrics
            and bool(all_metadata_list)
            and _supports_loss_metadata_metrics(self.compute_metrics)
        )
        if (
            self.compute_metrics is not None
            and (has_prediction_metrics or has_loss_metadata_metrics)
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = (
                all_losses if "loss" in args.include_for_metrics else None
            )
            eval_set_kwargs["inputs"] = (
                all_inputs if "inputs" in args.include_for_metrics else None
            )
            eval_set_kwargs["metadata_list"] = (
                all_metadata_list if "metadata" in args.include_for_metrics else None
            )
            metrics = self.compute_metrics(
                CustomEvalPrediction(
                    predictions=all_preds, label_ids=all_labels, **eval_set_kwargs
                )
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = (
                np.concatenate(all_losses).mean().item()
            )
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[
                f"{metric_key_prefix}_model_preparation_time"
            ] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )


class CustomEvalPrediction(EvalPrediction):
    """
    Custom EvalPrediction to handle additional metadata.
    The order of reutrn values is: predictions, label_ids, inputs, losses, metadata_list.
    """

    def __init__(
        self, predictions, label_ids, inputs=None, losses=None, metadata_list=None
    ):
        super().__init__(
            predictions=predictions, label_ids=label_ids, inputs=inputs, losses=losses
        )
        self.metadata_list = metadata_list if metadata_list is not None else []
        self.elements += (metadata_list,)
