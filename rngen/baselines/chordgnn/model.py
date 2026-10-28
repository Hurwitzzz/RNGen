"""Controlled PyTorch implementation of the public ChordGNN architecture."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import COMPONENTS
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import (
    FEATURE_DIM,
    REVERSE_EDGE_TYPES,
)

try:
    from torch_scatter import scatter as _torch_scatter
except ImportError:  # pragma: no cover - formal environment always pins it.
    _torch_scatter = None


UPSTREAM_COMMIT = "bcbdc72d2850e986c5f577727f800dee0f95f0c4"
RELATION_TYPES = OrderedDict(
    (
        ("onset", 0),
        ("consecutive", 1),
        ("during", 2),
        ("silence", 3),
        ("consecutive_rev", REVERSE_EDGE_TYPES["consecutive_rev"]),
        ("during_rev", REVERSE_EDGE_TYPES["during_rev"]),
        ("silence_rev", REVERSE_EDGE_TYPES["silence_rev"]),
    )
)


def _scatter_mean(
    source: torch.Tensor,
    index: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    if _torch_scatter is not None:
        return _torch_scatter(source, index, 0, out=output, reduce="mean")
    result = output.clone()
    result.index_add_(0, index, source)
    count = torch.zeros(output.shape[0], dtype=output.dtype, device=output.device)
    count.index_add_(0, index, torch.ones_like(index, dtype=output.dtype))
    return result / count.clamp_min(1).unsqueeze(-1)


def add_reverse_edges(
    edge_index: torch.Tensor, edge_type: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce upstream relation numbering: 1/2/3 -> 4/5/6."""

    indices = [edge_index]
    types = [edge_type]
    for forward_type, reverse_type in ((1, 4), (2, 5), (3, 6)):
        mask = edge_type == forward_type
        indices.append(edge_index[:, mask].flip(0))
        types.append(
            torch.full(
                (int(mask.sum()),),
                reverse_type,
                dtype=edge_type.dtype,
                device=edge_type.device,
            )
        )
    return torch.cat(indices, dim=1), torch.cat(types, dim=0)


class SageConvScatter(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.neigh_linear = nn.Linear(input_size, input_size, bias=bias)
        self.linear = nn.Linear(input_size * 2, output_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.neigh_linear.bias is not None:
            nn.init.zeros_(self.neigh_linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neighbors = self.neigh_linear(x)
        aggregated = _scatter_mean(
            neighbors[edge_index[1]], edge_index[0], output=neighbors.clone()
        )
        return self.linear(torch.cat((x, aggregated), dim=-1))


class HeteroSageConvLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.convs = nn.ModuleDict(
            {
                relation: SageConvScatter(input_size, output_size)
                for relation in RELATION_TYPES
            }
        )

    def reset_parameters(self) -> None:
        for convolution in self.convs.values():
            convolution.reset_parameters()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        outputs = [
            self.convs[name](x, edge_index[:, edge_type == relation])
            for name, relation in RELATION_TYPES.items()
        ]
        return torch.stack(outputs, dim=0).mean(dim=0)


class HeteroGraphSage(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("ChordGNN n_layers must be at least one")
        layers = [HeteroSageConvLayer(input_size, hidden_size)]
        layers.extend(
            HeteroSageConvLayer(hidden_size, hidden_size) for _ in range(n_layers - 1)
        )
        layers.append(HeteroSageConvLayer(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        hidden = x
        for layer in self.layers[:-1]:
            hidden = layer(hidden, edge_index, edge_type)
            hidden = F.relu(hidden)
            hidden = F.normalize(hidden)
            hidden = self.dropout(hidden)
        return self.layers[-1](hidden, edge_index, edge_type)


class OnsetEdgePoolingV2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.trans = nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        self.trans.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        onset_edges: torch.Tensor,
        representatives: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = torch.arange(x.shape[0], device=x.device)
        self_loops = node_ids.view(1, -1).repeat(2, 1)
        edges = torch.cat((onset_edges, self_loops), dim=1)
        transformed = self.trans(x)
        pooled = _scatter_mean(
            transformed[edges[0]], edges[1], output=torch.zeros_like(x)
        )
        return pooled[representatives]


class ChordTaskMLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float
    ):
        super().__init__()
        self.first = nn.Linear(input_size, hidden_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.first.weight, gain=gain)
        nn.init.xavier_uniform_(self.output.weight, gain=gain)
        self.batch_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        hidden = self.first(x.reshape(-1, shape[-1]))
        hidden = F.relu(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        logits = self.output(hidden)
        return logits.reshape(*shape[:-1], logits.shape[-1])


@dataclass(frozen=True)
class ChordGraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    onset_edges: torch.Tensor
    onset_representatives: torch.Tensor
    lengths: torch.Tensor
    labels: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]
    onset_quarter: Sequence[torch.Tensor]
    relative_paths: Sequence[str]

    def to(self, device: torch.device | str) -> "ChordGraphBatch":
        return ChordGraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            onset_edges=self.onset_edges.to(device),
            onset_representatives=self.onset_representatives.to(device),
            lengths=self.lengths.to(device),
            labels={key: value.to(device) for key, value in self.labels.items()},
            masks={key: value.to(device) for key, value in self.masks.items()},
            onset_quarter=self.onset_quarter,
            relative_paths=self.relative_paths,
        )


class ChordGNN(nn.Module):
    """Base ChordGNN without Post, NADE, JK, Rotograd, or GradNorm."""

    def __init__(
        self,
        head_sizes: Mapping[str, int],
        *,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 256,
        n_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        if set(head_sizes) != set(COMPONENTS):
            raise ValueError("ChordGNN head set must match the controlled ontology")
        self.head_sizes = {
            component: int(head_sizes[component]) for component in COMPONENTS
        }
        self.hidden_size = hidden_size
        self.pitch_embedding = nn.Embedding(128, 16)
        self.spelling_embedding = nn.Embedding(49, 16)
        self.feature_embedding = nn.Linear(input_size - 3, 32)
        self.encoder = HeteroGraphSage(
            64,
            hidden_size * 2,
            hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.pool = OnsetEdgePoolingV2(hidden_size)
        self.proj1 = nn.Linear(hidden_size + 1, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.gru = nn.GRU(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_norm = nn.LayerNorm(hidden_size)
        self.classifiers = nn.ModuleDict(
            {
                component: ChordTaskMLP(hidden_size, hidden_size, size, dropout=dropout)
                for component, size in self.head_sizes.items()
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.feature_embedding.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj1.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj2.weight, gain=gain)
        nn.init.zeros_(self.feature_embedding.bias)
        nn.init.zeros_(self.proj1.bias)
        nn.init.zeros_(self.proj2.bias)
        self.batch_norm1.reset_parameters()
        self.batch_norm2.reset_parameters()
        self.gru.reset_parameters()
        self.gru_norm.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def encode(self, batch: ChordGraphBatch) -> torch.Tensor:
        x = batch.x
        pitch = self.pitch_embedding(x[:, 0].long())
        # Upstream feeds the first spelling one-hot column to this embedding.
        spelling = self.spelling_embedding(x[:, 1].long())
        other = self.feature_embedding(x[:, 2:-1])
        hidden = torch.cat((other, pitch, spelling), dim=-1)
        hidden = self.encoder(hidden, batch.edge_index, batch.edge_type)
        hidden = F.normalize(F.relu(hidden))
        hidden = self.pool(hidden, batch.onset_edges, batch.onset_representatives)
        duration = x[:, -1][batch.onset_representatives].unsqueeze(-1)
        hidden = torch.cat((hidden, duration), dim=-1)
        hidden = self.batch_norm1(F.relu(self.proj1(hidden)))
        hidden = self.batch_norm2(F.relu(self.proj2(hidden)))
        split = torch.split(hidden, batch.lengths.detach().cpu().tolist(), dim=0)
        padded = nn.utils.rnn.pad_sequence(split, batch_first=True, padding_value=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded,
            batch.lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed, _ = self.gru(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, padding_value=-1
        )
        return self.gru_norm(padded)

    def forward(self, batch: ChordGraphBatch) -> dict[str, torch.Tensor]:
        hidden = self.encode(batch)
        return {
            component: classifier(hidden)
            for component, classifier in self.classifiers.items()
        }


class SafeMaskedWLoss(nn.Module):
    """Upstream learned task weighting with zero-supervision safety."""

    def __init__(self, components: Sequence[str] = COMPONENTS):
        super().__init__()
        self.components = tuple(components)
        self.task_scales = nn.Parameter(torch.ones(len(self.components)))

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = next(iter(logits.values())).sum() * 0
        losses: dict[str, torch.Tensor] = {}
        for index, component in enumerate(self.components):
            mask = masks[component].bool()
            if mask.any():
                loss = F.cross_entropy(logits[component][mask], labels[component][mask])
                scale = self.task_scales[index]
                total = total + 0.5 / (scale**2) * loss + torch.log1p(scale**2)
            else:
                loss = logits[component].sum() * 0
            losses[component] = loss
        losses["total"] = total
        return total, losses


def chordgnn_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
