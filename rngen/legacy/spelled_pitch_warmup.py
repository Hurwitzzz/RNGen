"""Encoder-only model wrapper for Octuple+ spelled-pitch warm-up."""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput

from rngen.legacy.spell_warmup_data import CANONICAL_SPELLED_PITCH_CLASSES
from rngen.modeling.encoder import load_spelled_musicbert_checkpoint


class SpelledPitchWarmupModel(nn.Module):
    """SpelledMusicBert plus a note-level linear classifier head."""

    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(
        self,
        rnbert_checkpoint_path: str,
        *,
        spelling_tokens: Optional[List[str]] = None,
        spelling_token_ids: Optional[Sequence[int]] = None,
        vocab_size: Optional[int] = None,
        num_labels: int = len(CANONICAL_SPELLED_PITCH_CLASSES),
        trainable_scope: Literal["spelling_path", "all_encoder"] = "spelling_path",
    ):
        super().__init__()
        self.rnbert, self.spelling_tokens = load_spelled_musicbert_checkpoint(
            rnbert_checkpoint_path,
            spelling_tokens=spelling_tokens,
            vocab_size=vocab_size,
        )
        hidden_size = self.rnbert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = self.rnbert.config
        self.config.rnbert_checkpoint_path = rnbert_checkpoint_path
        self.trainable_scope = trainable_scope
        self._gradient_hooks = []
        self._validated_first_batch = False
        self.configure_trainable_parameters(
            trainable_scope, spelling_token_ids=spelling_token_ids
        )

    @staticmethod
    def _apply_gradient_mask(grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(device=grad.device, dtype=grad.dtype)

    def configure_trainable_parameters(
        self,
        trainable_scope: Literal["spelling_path", "all_encoder"],
        *,
        spelling_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if trainable_scope == "all_encoder":
            for parameter in self.parameters():
                parameter.requires_grad = True
            return

        if trainable_scope != "spelling_path":
            raise ValueError(f"Unsupported trainable scope: {trainable_scope}")
        if spelling_token_ids is None:
            raise ValueError("spelling_token_ids are required for spelling_path scope.")

        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

        word_embeddings = self.rnbert.get_input_embeddings()
        word_embeddings.weight.requires_grad = True
        row_mask = torch.zeros_like(word_embeddings.weight)
        row_mask[list(spelling_token_ids)] = 1
        self._gradient_hooks.append(
            word_embeddings.weight.register_hook(
                lambda grad: self._apply_gradient_mask(grad, row_mask)
            )
        )

        downsampling = self.rnbert.bert.embeddings.downsampling[0]
        downsampling.weight.requires_grad = True
        hidden_size = self.rnbert.config.hidden_size
        compound_ratio = getattr(self.rnbert.config, "compound_ratio", 9)
        spelling_start = (compound_ratio - 1) * hidden_size
        spelling_end = compound_ratio * hidden_size
        column_mask = torch.zeros_like(downsampling.weight)
        column_mask[:, spelling_start:spelling_end] = 1
        self._gradient_hooks.append(
            downsampling.weight.register_hook(
                lambda grad: self._apply_gradient_mask(grad, column_mask)
            )
        )

    def _validate_batch_indices(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> None:
        """Fail before CUDA embedding kernels if a warm-up batch is malformed."""
        if input_ids.numel() == 0:
            raise ValueError("Warm-up input_ids cannot be empty.")

        min_input_id = int(input_ids.min().detach().cpu())
        max_input_id = int(input_ids.max().detach().cpu())
        vocab_size = self.rnbert.get_input_embeddings().num_embeddings
        if min_input_id < 0 or max_input_id >= vocab_size:
            raise ValueError(
                "Warm-up input_ids contain token ids outside the encoder "
                f"vocabulary: min={min_input_id}, max={max_input_id}, "
                f"vocab_size={vocab_size}."
            )

        compound_ratio = getattr(self.rnbert.config, "compound_ratio", 9)
        if input_ids.shape[1] % compound_ratio != 0:
            raise ValueError(
                "Warm-up input length must be divisible by the compound ratio: "
                f"input_length={input_ids.shape[1]}, compound_ratio={compound_ratio}."
            )
        note_seq_len = input_ids.shape[1] // compound_ratio

        if attention_mask is not None and attention_mask.shape[-1] != note_seq_len:
            raise ValueError(
                "Warm-up attention_mask length must match note sequence length: "
                f"attention_mask_length={attention_mask.shape[-1]}, "
                f"note_seq_len={note_seq_len}."
            )

        if labels is not None:
            if labels.shape[-1] != note_seq_len:
                raise ValueError(
                    "Warm-up label length must match note sequence length: "
                    f"label_length={labels.shape[-1]}, note_seq_len={note_seq_len}."
                )
            valid_labels = labels[labels != -100]
            if valid_labels.numel() > 0:
                min_label = int(valid_labels.min().detach().cpu())
                max_label = int(valid_labels.max().detach().cpu())
                if min_label < 0 or max_label >= self.num_labels:
                    raise ValueError(
                        "Warm-up labels contain class ids outside the classifier "
                        f"label space: min={min_label}, max={max_label}, "
                        f"num_labels={self.num_labels}."
                    )

        embeddings = self.rnbert.bert.embeddings
        if hasattr(embeddings, "position_embeddings"):
            position_vocab_size = embeddings.position_embeddings.num_embeddings
            if note_seq_len + 1 >= position_vocab_size:
                raise ValueError(
                    "Warm-up note sequence exceeds position embedding range: "
                    f"note_seq_len={note_seq_len}, "
                    f"position_embeddings={position_vocab_size}."
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        kwargs.pop("num_items_in_batch", None)
        if not self._validated_first_batch:
            self._validate_batch_indices(input_ids, attention_mask, labels)
            self._validated_first_batch = True
        outputs = self.rnbert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_encoder(self, save_directory: str) -> str:
        os.makedirs(save_directory, exist_ok=True)
        # Save the encoder independently of this task's classifier.
        self.rnbert.save_pretrained(save_directory, safe_serialization=False)
        return save_directory
