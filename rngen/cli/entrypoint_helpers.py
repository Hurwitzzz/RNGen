"""Shared model-loading helpers for RNGen entrypoints and the SFT trainer."""

import os
import torch

from rngen.modeling import RNGenModel


def is_distributed_training():
    """Return whether the current process is part of a multi-process job."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def resolve_training_device_map(default="auto"):
    """Choose a model loading device_map compatible with the training launcher.

    ``device_map="auto"`` is useful for single-process loading, but it can create
    DTensor/Accelerate placements before DDP wraps the model. In multi-process
    training each rank should load an ordinary full replica and let DDP move it.
    """
    return None if is_distributed_training() else default


def load_pretrained_rngen_checkpoint(
    model_path,
    *,
    use_spelled_musicbert,
    is_lora_trainable,
    spelling_tokens=None,
    spelling_downsampling_scale=1.0,
    device_map="auto",
    torch_dtype=torch.bfloat16,
):
    """Load a saved RNGen checkpoint and return both model and tokenizer."""
    model = RNGenModel.from_pretrained(
        model_path=model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        is_lora_trainable=is_lora_trainable,
        use_spelled_musicbert=use_spelled_musicbert,
        spelling_tokens=spelling_tokens,
        spelling_downsampling_scale=spelling_downsampling_scale,
    )
    return model, model.tokenizer


def get_model_compound_ratio(model):
    """Read the compound ratio used by the model's RNBert encoder."""
    return model.rnbert.bert.embeddings.compound_ratio
