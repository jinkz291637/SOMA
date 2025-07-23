# layers/encoder.py
# Author: jinkz

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba 


class FeedForward(nn.Module):
    """
    Point-wise FeedForward network:
    Conv1d -> Activation -> Conv1d with optional dropout.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.Dropout(dropout),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to (B, C, L) for Conv1d, then back
        x = x.transpose(-1, -2)
        x = self.net(x)
        return x.transpose(-1, -2)


class EncoderLayer(nn.Module):
    """
    A single dual-Mamba encoder layer:
    - Forward Mamba
    - Backward Mamba (via time-reversed input)
    - FeedForward + LayerNorm with residuals
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_model, d_conv=2, expand=1)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_model, d_conv=2, expand=1)

        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_text: torch.Tensor = None) -> tuple[torch.Tensor, None]:
        # Bidirectional Mamba processing
        x_rev = torch.flip(x, dims=[1])
        mamba_out = self.mamba_fwd(x, x_text) + torch.flip(self.mamba_bwd(x_rev, x_text), dims=[1])
        x = self.norm1(x + mamba_out)

        # FeedForward processing
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, None


class Encoder(nn.Module):
    """
    Stack of EncoderLayers with optional final normalization.
    """
    def __init__(self, layers: list[nn.Module], norm_layer: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, x_text: torch.Tensor = None) -> tuple[torch.Tensor, list[None]]:
        attention_outputs = []
        for layer in self.layers:
            x, attn = layer(x, x_text)
            attention_outputs.append(attn)
        if self.norm:
            x = self.norm(x)
        return x, attention_outputs


def build_mamba_stack(cfg: dict) -> Encoder:
    """
    Build a stack of Mamba-based encoder layers.

    Args:
        cfg: model configuration dictionary.

    Returns:
        Encoder module with multiple EncoderLayers.
    """
    layers = [
        EncoderLayer(
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            activation=cfg.get("activation", "relu")
        )
        for _ in range(cfg["e_layers"])
    ]
    norm = nn.LayerNorm(cfg["d_model"]) if cfg.get("use_norm", True) else None
    return Encoder(layers, norm_layer=norm)
