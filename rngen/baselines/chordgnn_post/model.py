"""Learned sequential post-processing for frozen ChordGNN predictions."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import COMPONENTS


class ChordGNNPostProcessor(nn.Module):
    """Official ChordGNN +Post sequence model with stable head ordering."""

    def __init__(
        self,
        head_sizes: Mapping[str, int],
        *,
        hidden_size: int = 256,
        n_layers: int = 1,
        dropout: float = 0.4,
        components: Sequence[str] = COMPONENTS,
    ):
        super().__init__()
        self.components = tuple(components)
        if set(head_sizes) != set(self.components):
            raise ValueError("ChordGNN +Post head set differs from the ontology")
        self.head_sizes = {
            component: int(head_sizes[component]) for component in self.components
        }
        if any(size <= 0 for size in self.head_sizes.values()):
            raise ValueError("ChordGNN +Post head sizes must be positive")
        self.input_size = sum(self.head_sizes.values())
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.dropout_probability = float(dropout)
        if self.hidden_size <= 0 or self.n_layers <= 0:
            raise ValueError("ChordGNN +Post hidden size and layers must be positive")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("ChordGNN +Post dropout must be in [0, 1)")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_probability,
        )
        self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.classifiers = nn.ModuleDict(
            {
                component: nn.Linear(self.hidden_size, size)
                for component, size in self.head_sizes.items()
            }
        )

    def concatenate_probabilities(
        self, base_logits: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if set(base_logits) != set(self.components):
            raise ValueError("ChordGNN +Post received an incomplete base-logit set")
        reference_shape: tuple[int, ...] | None = None
        probabilities: list[torch.Tensor] = []
        for component in self.components:
            logits = base_logits[component]
            if logits.ndim != 3:
                raise ValueError(
                    f"ChordGNN +Post {component} logits must have shape [B,T,C]"
                )
            if logits.shape[-1] != self.head_sizes[component]:
                raise ValueError(
                    f"ChordGNN +Post {component} class count "
                    f"{logits.shape[-1]} != {self.head_sizes[component]}"
                )
            if reference_shape is None:
                reference_shape = tuple(logits.shape[:2])
            elif tuple(logits.shape[:2]) != reference_shape:
                raise ValueError("ChordGNN +Post base-logit sequence shapes differ")
            probabilities.append(F.softmax(logits, dim=-1))
        return torch.cat(probabilities, dim=-1)

    def forward(
        self,
        base_logits: Mapping[str, torch.Tensor],
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        probabilities = self.concatenate_probabilities(base_logits)
        if lengths is not None:
            if lengths.ndim != 1 or len(lengths) != probabilities.shape[0]:
                raise ValueError("ChordGNN +Post lengths must have shape [B]")
            if int(lengths.min()) <= 0 or int(lengths.max()) > probabilities.shape[1]:
                raise ValueError("ChordGNN +Post lengths exceed the padded sequence")
        hidden, _ = self.lstm(probabilities)
        hidden = F.relu(self.projection(hidden))
        hidden = F.normalize(hidden, p=2, dim=-1)
        hidden = self.dropout(hidden)
        return {
            component: classifier(hidden)
            for component, classifier in self.classifiers.items()
        }


def freeze_chordgnn(model: nn.Module) -> nn.Module:
    """Freeze a base model and force evaluation mode."""

    model.requires_grad_(False)
    model.eval()
    return model
