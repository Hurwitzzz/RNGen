import inspect
import json
import logging
import os

import torch
from musicbert_hf.models import MusicBert, freeze_layers
from peft import PeftModel, get_peft_model
from transformers import PreTrainedModel

from rngen.modeling.components import RNBertProjector
from rngen.modeling.encoder import (
    MusicBertBackbone,
    load_finetuned_musicbert_checkpoint,
    scale_spelling_downsampling_slice,
)
from rngen.modeling.io import _load_weights_from_path, _save_weights

logger = logging.getLogger(__name__)


class RNGenModel(PreTrainedModel):
    """Combined RNBert + LLM model for music understanding."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        rnbert=None,
        llm=None,
        tokenizer=None,
        use_lora=False,
        lora_config=None,
    ):
        super().__init__(config)
        self.config = config

        if rnbert is None or llm is None or tokenizer is None:
            raise ValueError("RNGenModel requires rnbert, llm, and tokenizer. ")

        if isinstance(rnbert, MusicBert):
            rnbert = MusicBertBackbone(rnbert.config, rnbert.bert)
        self.rnbert = rnbert

        if use_lora and lora_config is not None:
            logger.info("Applying LoRA to LLM component")
            self.llm = get_peft_model(llm, lora_config)
        else:
            self.llm = llm

        self.tokenizer = tokenizer

        self.projector = RNBertProjector(
            input_dim=self.rnbert.config.hidden_size,
            output_dim=self.llm.config.hidden_size,
            num_layers=config.projector_layers,
            dropout=config.projector_dropout,
        )
        self.projector = self.projector.to(torch.bfloat16)

        if hasattr(config, "freeze_layers_rnbert"):
            if config.freeze_layers_rnbert == "all":
                for param in self.rnbert.parameters():
                    param.requires_grad = False
                logger.info("Freezing all RNBert parameters")
            elif config.freeze_layers_rnbert == "none":
                logger.info("Not freezing any RNBert parameters")
            else:
                freeze_layers(self.rnbert, config.freeze_layers_rnbert, freeze_cls=True)
                logger.info("Freezing %s RNBert layers.", config.freeze_layers_rnbert)

        if config.freeze_llm and not use_lora:
            for param in self.llm.parameters():
                param.requires_grad = False

    def get_music_embeddings(self, music_input_ids, music_attention_mask=None):
        rnbert_outputs = self.rnbert.bert(
            input_ids=music_input_ids,
            attention_mask=music_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            upsample=False,
        )
        return self.projector(rnbert_outputs.last_hidden_state)

    def prepare_inputs_with_music(
        self, music_embeddings, prompt_input_ids, prompt_attention_mask=None
    ):
        batch_size = music_embeddings.shape[0]
        device = music_embeddings.device

        music_start_id = self.tokenizer.convert_tokens_to_ids("<music>")
        music_end_id = self.tokenizer.convert_tokens_to_ids("</music>")

        prompt_embeddings = self.llm.get_input_embeddings()(prompt_input_ids)

        combined_embeddings = []
        combined_attention_mask = []

        for b in range(batch_size):
            start_pos = (prompt_input_ids[b] == music_start_id).nonzero(as_tuple=True)[
                0
            ]
            end_pos = (prompt_input_ids[b] == music_end_id).nonzero(as_tuple=True)[0]

            assert (
                len(start_pos) == 1 and len(end_pos) == 1
            ), "Prompt must contain and only contain one pair of <music> and </music> tokens"

            start_pos = start_pos[0].item()
            end_pos = end_pos[0].item()

            concat_embeddings = torch.cat(
                [
                    prompt_embeddings[b, :start_pos],
                    music_embeddings[b],
                    prompt_embeddings[b, end_pos + 1 :],
                ],
                dim=0,
            )
            combined_embeddings.append(concat_embeddings)

            if prompt_attention_mask is not None:
                new_mask = torch.cat(
                    [
                        prompt_attention_mask[b, :start_pos],
                        torch.ones(music_embeddings[b].shape[0], device=device),
                        prompt_attention_mask[b, end_pos + 1 :],
                    ],
                    dim=0,
                )
                combined_attention_mask.append(new_mask)

        max_len = max(emb.shape[0] for emb in combined_embeddings)
        pad_token_id = self.tokenizer.pad_token_id
        pad_embeds = self.llm.get_input_embeddings()(
            torch.tensor([pad_token_id], device=device)
        )

        padded_embeddings = []
        padded_attention_mask = []

        for i, emb in enumerate(combined_embeddings):
            padding_len = max_len - emb.shape[0]
            if padding_len > 0:
                padding = pad_embeds.expand(padding_len, -1)
                padded_emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(padded_emb)

                if combined_attention_mask:
                    padded_mask = torch.cat(
                        [
                            combined_attention_mask[i],
                            torch.zeros(padding_len, device=device),
                        ],
                        dim=0,
                    )
                    padded_attention_mask.append(padded_mask)
            else:
                padded_embeddings.append(emb)
                if combined_attention_mask:
                    padded_attention_mask.append(combined_attention_mask[i])

        inputs = {"inputs_embeds": torch.stack(padded_embeddings, dim=0)}
        if combined_attention_mask:
            inputs["attention_mask"] = torch.stack(padded_attention_mask, dim=0)

        return inputs

    def forward(
        self,
        music_input_ids=None,
        music_attention_mask=None,
        eval_input_ids=None,
        eval_attention_mask=None,
        sample_metadata=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        music_embeddings = self.get_music_embeddings(
            music_input_ids=music_input_ids,
            music_attention_mask=music_attention_mask,
        )

        music_start_id = self.tokenizer.convert_tokens_to_ids("<music>")
        music_end_id = self.tokenizer.convert_tokens_to_ids("</music>")

        combined_inputs = self.prepare_inputs_with_music(
            music_embeddings=music_embeddings,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )

        assert labels is not None, "Labels must be provided for training"

        batch_size = prompt_input_ids.size(0)
        new_labels = []

        for b in range(batch_size):
            start_pos = (prompt_input_ids[b] == music_start_id).nonzero(as_tuple=True)[
                0
            ]
            end_pos = (prompt_input_ids[b] == music_end_id).nonzero(as_tuple=True)[0]

            assert (
                len(start_pos) == 1 and len(end_pos) == 1
            ), "Prompt must contain <music> and </music> tokens"
            start_pos = start_pos[0].item()
            end_pos = end_pos[0].item()

            music_embed_len = music_embeddings[b].size(0)
            prefix_labels = labels[b, :start_pos]
            music_labels = torch.full(
                (music_embed_len,),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            suffix_labels = labels[b, end_pos + 1 :]
            adjusted_labels = torch.cat([prefix_labels, music_labels, suffix_labels])

            combined_length = combined_inputs["inputs_embeds"].size(1)
            pad_length = combined_length - adjusted_labels.size(0)

            if pad_length > 0:
                raise ValueError(
                    "Adjusted labels should match the length of combined embeddings when we do not remove paddings from music_embs before executing `prepare_inputs_with_music()`."
                )
            if pad_length < 0:
                raise ValueError(
                    "Adjusted labels should not be longer than combined embeddings"
                )

            new_labels.append(adjusted_labels)
        labels = torch.stack(new_labels, dim=0)

        return self.llm(
            inputs_embeds=combined_inputs["inputs_embeds"],
            attention_mask=combined_inputs.get("attention_mask", None),
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(
        self,
        music_input_ids=None,
        music_attention_mask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        max_new_tokens=None,
        **kwargs,
    ):
        music_embeddings = self.get_music_embeddings(
            music_input_ids=music_input_ids,
            music_attention_mask=music_attention_mask,
        )

        combined_inputs = self.prepare_inputs_with_music(
            music_embeddings=music_embeddings,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )

        kwargs["bos_token_id"] = self.tokenizer.pad_token_id
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        kwargs["forced_eos_token_id"] = self.tokenizer.eos_token_id
        kwargs["stop_strings"] = self.tokenizer.eos_token
        kwargs["tokenizer"] = self.tokenizer

        return self.llm.generate(
            inputs_embeds=combined_inputs["inputs_embeds"],
            attention_mask=combined_inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        main_weight_map = {}
        using_lora = hasattr(self.llm, "save_pretrained") and isinstance(
            self.llm, PeftModel
        )

        rnbert_path = os.path.join(save_directory, "rnbert")
        unfrozen_params_rnbert = {}
        for name, param in self.rnbert.named_parameters():
            if param.requires_grad and not name.startswith("cls."):
                unfrozen_params_rnbert[name] = param.data.clone()
                main_weight_map[
                    f"rnbert.{name}"
                ] = f"rnbert/{'model.safetensors' if safe_serialization else 'pytorch_model.bin'}"

        _save_weights(
            unfrozen_params_rnbert,
            rnbert_path,
            "unfrozen RNBert weights",
            safe_serialization,
        )

        projector_path = os.path.join(save_directory, "projector")
        os.makedirs(projector_path, exist_ok=True)
        for name in self.projector.state_dict():
            main_weight_map[
                f"projector.{name}"
            ] = f"projector/{'model.safetensors' if safe_serialization else 'pytorch_model.bin'}"

        _save_weights(
            self.projector.state_dict(),
            projector_path,
            "projector weights",
            safe_serialization,
        )

        if using_lora:
            lora_path = os.path.join(save_directory, "llm_lora")
            self.llm.save_pretrained(
                lora_path,
                safe_serialization=safe_serialization,
                save_embedding_layers=False,
            )
            logger.info("Saved LoRA weights to %s", lora_path)
            main_weight_map[
                "llm.lora"
            ] = f"llm_lora/{'adapter_model.safetensors' if safe_serialization else 'pytorch_model.bin'}"
        else:
            llm_path = os.path.join(save_directory, "llm")
            unfrozen_params_llm = {}
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    unfrozen_params_llm[name] = param.data.clone()
                    main_weight_map[
                        f"llm.{name}"
                    ] = f"llm/{'model.safetensors' if safe_serialization else 'pytorch_model.bin'}"

            _save_weights(
                unfrozen_params_llm,
                llm_path,
                "unfrozen LLM weights",
                safe_serialization,
            )

        main_index = {"metadata": {"format": "pt"}, "weight_map": main_weight_map}
        main_index_file = os.path.join(
            save_directory,
            "model.safetensors.index.json"
            if safe_serialization
            else "pytorch_model.bin.index.json",
        )
        with open(main_index_file, "w") as f:
            json.dump(main_index, f)
        logger.info("Saved main weight index to %s", main_index_file)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
            logger.info("Saved tokenizer to %s", save_directory)

        metadata = {
            "using_lora": using_lora,
            "rnbert_checkpoint_path": getattr(
                self.config, "rnbert_checkpoint_path", None
            ),
            "llm_model_name": getattr(self.config, "llm_model_name", None),
            "use_safetensors": safe_serialization,
            "spelling_downsampling_scale": getattr(
                self.config, "spelling_downsampling_scale", None
            ),
        }

        with open(os.path.join(save_directory, "rngen_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        use_spelled_musicbert: bool,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        *args,
        **kwargs,
    ):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

        logger.info("Loading RNGen model from %s", model_path)

        config = PretrainedConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        metadata_path = os.path.join(model_path, "rngen_metadata.json")
        if not os.path.exists(metadata_path):
            # Checkpoints published under the previous model name use this filename.
            metadata_path = os.path.join(model_path, "rninfer_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        using_lora = metadata.get("using_lora", getattr(config, "use_lora", False))
        spelling_downsampling_scale = kwargs.get("spelling_downsampling_scale", 1.0)

        rnbert_orig_path = getattr(config, "rnbert_checkpoint_path", None)
        if rnbert_orig_path and not os.path.isabs(rnbert_orig_path):
            rnbert_orig_path = os.path.join(model_path, rnbert_orig_path)
        if rnbert_orig_path and os.path.exists(rnbert_orig_path):
            rnbert = load_finetuned_musicbert_checkpoint(
                rnbert_orig_path,
                os.path.join(model_path, "rnbert"),
                use_spelled_musicbert=use_spelled_musicbert,
                spelling_tokens=kwargs.get("spelling_tokens", None),
                freeze_layers_rnbert=getattr(config, "freeze_layers_rnbert", "none"),
            )
        else:
            raise NotImplementedError(
                "MusicBert checkpoint path not found. We don't support loading MusicBert from scratch yet."
            )

        llm_name = getattr(config, "llm_model_name")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, device_map=device_map, torch_dtype=torch_dtype
        )
        if len(tokenizer) != llm.get_input_embeddings().num_embeddings:
            logger.info(
                "Resizing LLM token embeddings from %s to %s.",
                llm.get_input_embeddings().num_embeddings,
                len(tokenizer),
            )
            llm.resize_token_embeddings(len(tokenizer))
        rnbert = rnbert.to(llm.device)

        model_init_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(cls.__init__).parameters
        }
        model = cls(
            config=config,
            rnbert=rnbert,
            llm=llm,
            tokenizer=tokenizer,
            **model_init_kwargs,
        )

        if use_spelled_musicbert:
            scale_spelling_downsampling_slice(rnbert, spelling_downsampling_scale)

        projector_path = os.path.join(model_path, "projector")
        if os.path.exists(projector_path):
            try:
                projector_state_dict = _load_weights_from_path(projector_path)
                if projector_state_dict:
                    model.projector.load_state_dict(projector_state_dict)
                    model.projector.to(llm.device)
                    logger.info("Loaded projector weights from %s", projector_path)
            except Exception as e:
                logger.error(
                    "Failed to load projector weights from %s: %s",
                    projector_path,
                    e,
                )
        else:
            logger.warning("Projector weights directory not found: %s", projector_path)

        if using_lora:
            lora_path = os.path.join(model_path, "llm_lora")
            if os.path.exists(lora_path):
                logger.info("Loading LoRA adapter from %s", lora_path)
                model.llm = PeftModel.from_pretrained(
                    model.llm,
                    lora_path,
                    is_trainable=kwargs.get("is_lora_trainable", False),
                    autocast_adapter_dtype=False,
                    device_map=device_map,
                )
                logger.info("Applied LoRA adapter to LLM.")
            else:
                logger.warning(
                    "LoRA is enabled, but LoRA adapter path not found: %s", lora_path
                )
        else:
            llm_ft_path = os.path.join(model_path, "llm")
            loaded_llm_ft = False
            if os.path.exists(llm_ft_path):
                try:
                    llm_state_dict = _load_weights_from_path(llm_ft_path)
                    if llm_state_dict:
                        missing_keys, unexpected_keys = model.llm.load_state_dict(
                            llm_state_dict, strict=False
                        )
                        logger.info(
                            "Loaded fine-tuned LLM weights from %s", llm_ft_path
                        )
                        if missing_keys:
                            logger.warning("LLM missing keys: %s", missing_keys)
                        if unexpected_keys:
                            logger.warning("LLM unexpected keys: %s", unexpected_keys)
                        loaded_llm_ft = True
                except Exception as e:
                    logger.warning(
                        "Could not load fine-tuned LLM weights from %s: %s",
                        llm_ft_path,
                        e,
                    )
            if not loaded_llm_ft:
                logger.info(
                    "Not using LoRA and no fine-tuned LLM weights found or loaded."
                )

        if hasattr(config, "freeze_layers_rnbert"):
            if config.freeze_layers_rnbert == "all":
                for param in model.rnbert.parameters():
                    param.requires_grad = False
            elif config.freeze_layers_rnbert == "none":
                for param in model.rnbert.parameters():
                    param.requires_grad = True
            else:
                freeze_layers(
                    model.rnbert, config.freeze_layers_rnbert, freeze_cls=True
                )

        if hasattr(config, "freeze_llm") and config.freeze_llm and not using_lora:
            for param in model.llm.parameters():
                param.requires_grad = False

        if hasattr(config, "freeze_projector") and config.freeze_projector:
            for param in model.projector.parameters():
                param.requires_grad = False

        return model
