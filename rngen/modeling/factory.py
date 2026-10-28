import logging
from typing import List, Literal, Optional, Union

import torch
from peft import LoraConfig, TaskType

from rngen.modeling.core import RNGenModel
from rngen.modeling.encoder import (
    MusicBertBackbone,
    load_spelled_musicbert_checkpoint,
    scale_spelling_downsampling_slice,
)

logger = logging.getLogger(__name__)


def create_rngen_model(
    rnbert_checkpoint: Optional[str] = None,
    llm_name: str = "Qwen/Qwen2.5-7B",
    device_map="auto",
    freeze_layers_rnbert: Optional[Union[int, List[int], Literal["all", "none"]]] = 9,
    freeze_llm: bool = True,
    projector_layers: int = 2,
    projector_dropout: float = 0.1,
    use_spelled_musicbert: bool = False,
    spelling_tokens: List[str] = None,
    spelling_downsampling_scale: float = 1.0,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
):
    """Create an RNGenModel with RNBert and LLM components."""
    if rnbert_checkpoint is not None:
        from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

        if use_spelled_musicbert:
            rnbert, _ = load_spelled_musicbert_checkpoint(
                rnbert_checkpoint, spelling_tokens=spelling_tokens
            )
            scale_spelling_downsampling_slice(rnbert, spelling_downsampling_scale)
        else:
            original_model = load_musicbert_from_fairseq_checkpoint(
                rnbert_checkpoint, print_missing_keys=True
            )
            rnbert = MusicBertBackbone(original_model.config, original_model.bert)
        logger.info("Loaded RNBert from %s", rnbert_checkpoint)
    else:
        raise NotImplementedError(
            "RNBert checkpoint path not provided. We should not RNBert from scratch yet."
        )

    rnbert = rnbert.to(torch.bfloat16)

    from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

    tokenizer = AutoTokenizer.from_pretrained(llm_name, eos_token="<|im_end|>")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.bfloat16, device_map=device_map
    )

    special_tokens = ["<music>", "</music>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    llm.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = None
    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info("Created LoRA config with target modules: %s", lora_target_modules)

    config = PretrainedConfig()
    config.rnbert_checkpoint_path = rnbert_checkpoint
    config.llm_model_name = llm_name
    config.freeze_layers_rnbert = freeze_layers_rnbert
    config.freeze_llm = freeze_llm
    config.projector_layers = projector_layers
    config.projector_dropout = projector_dropout
    config.use_lora = use_lora
    config.use_spelled_musicbert = use_spelled_musicbert
    config.spelling_downsampling_scale = spelling_downsampling_scale

    model = RNGenModel(
        config,
        rnbert=rnbert,
        llm=llm,
        tokenizer=tokenizer,
        use_lora=use_lora,
        lora_config=lora_config,
    )

    return model, tokenizer
