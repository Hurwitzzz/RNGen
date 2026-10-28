import torch
import torch.nn as nn


class RNBertProjector(nn.Module):
    """Project RNBert hidden states into the LLM hidden space."""

    def __init__(
        self, input_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, output_dim))
        self.layers.append(nn.LayerNorm(output_dim))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(output_dim, output_dim))
            self.layers.append(nn.LayerNorm(output_dim))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
