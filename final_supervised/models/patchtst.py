import math

import torch
import torch.nn as nn

from .patching import make_patches


class PatchTSTEncoderLayer(nn.Module):
    """
    Transformer encoder layer using BatchNorm instead of LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def _bn(self, norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self._bn(self.norm1, x + attn_out)
        x = self._bn(self.norm2, x + self.ff(x))
        return x


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST for supervised forecasting.

    Each channel is processed independently through a shared Transformer
    backbone.
    """

    def __init__(
        self,
        M: int,
        L: int,
        T: int,
        P: int = 16,
        S: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.M = M
        self.L = L
        self.T = T
        self.P = P
        self.S = S
        self.N = math.floor((L - P) / S) + 2

        self.patch_proj = nn.Linear(P, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.N, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.encoder = nn.ModuleList(
            [
                PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(self.N * d_model, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (B, M, L).

        Returns:
            Tensor with shape (B, M, T).
        """
        B, M, _ = x.shape

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x_n = (x - mean) / std

        patches = make_patches(x_n, self.P, self.S)
        z = self.patch_proj(patches) + self.pos_enc

        for layer in self.encoder:
            z = layer(z)

        z_flat = z.reshape(B * M, -1)
        out = self.head(z_flat)
        out = out.reshape(B, M, self.T)

        out = out * std + mean
        return out
