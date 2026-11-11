"""DBS-BS-RoFormer.

:class:`DBSBSRoformer` keeps the DBS tokenizer, decoder, and hard Top-K
routing of DBS-BSRNN and replaces only the shared separator with alternating
time-axis and band-axis rotary transformer blocks.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, pack, unpack

from .bsrnn import DBSBSRNN, RMSNorm


class RotaryAxisBlock(nn.Module):
    """Pre-norm gated RoPE attention and feed-forward along one sequence axis."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float, dropout: float):
        super().__init__()
        if min(dim, heads, dim_head) <= 0 or dim_head % 2 or ff_mult <= 0:
            raise ValueError("dimensions must be positive and dim_head must be even")
        self.heads, self.dim_head, self.dropout = heads, dim_head, dropout
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Linear(heads * dim_head, dim, bias=False)
        self.ff = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, int(dim * ff_mult)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(int(dim * ff_mult), dim), nn.Dropout(dropout),
        )

    def _rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        # Angles in fp32 even under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv = 10000.0 ** (-torch.arange(0, self.dim_head, 2, device=x.device).float() / self.dim_head)
            angles = positions.float()[:, None, :, None] * inv
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: Tensor, mask: Tensor | None = None, positions: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        if positions is None:
            positions = torch.arange(length, device=x.device).expand(batch, -1)
        normalized = self.norm(x)
        q, k, v = (self.to_qkv(normalized).view(batch, length, 3, self.heads, self.dim_head)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        q, k = self._rotate(q, positions), self._rotate(k, positions)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        gates = self.to_gates(normalized).sigmoid().transpose(1, 2).unsqueeze(-1)
        x = x + self.to_out((attended * gates).transpose(1, 2).reshape(batch, length, -1))
        x = x + self.ff(x)
        return x if mask is None else x.masked_fill(~mask[..., None], 0)


class AxialRoformerSeparator(nn.Module):
    """Alternating time and band attention over the tokens of all regions.

    Band-axis RoPE uses the ordinal of each active token, so masked slots
    between regions do not create position gaps; masked slots are excluded
    from attention and zeroed after every block.
    """

    def __init__(self, dim: int, depth: int, heads: int = 8, dim_head: int = 64, ff_mult: float = 4,
                 dropout: float = 0.0, activation_checkpointing: bool = False):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.activation_checkpointing = activation_checkpointing
        self.layers = nn.ModuleList([
            nn.ModuleList([RotaryAxisBlock(dim, heads, dim_head, ff_mult, dropout),
                           RotaryAxisBlock(dim, heads, dim_head, ff_mult, dropout)])
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def _axes(self, x: Tensor, mask: Tensor, time_block: nn.Module, freq_block: nn.Module) -> Tensor:
        batch, frames, bands, dim = x.shape
        x = time_block(x.transpose(1, 2).reshape(batch * bands, frames, dim))
        x = x.reshape(batch, bands, frames, dim).transpose(1, 2)
        x = x.masked_fill(~mask[:, None, :, None], 0)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        x = freq_block(
            x.reshape(batch * frames, bands, dim),
            mask[:, None, :].expand(-1, frames, -1).reshape(batch * frames, bands),
            positions[:, None, :].expand(-1, frames, -1).reshape(batch * frames, bands),
        )
        return x.reshape(batch, frames, bands, dim)

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        if token_mask is None:
            token_mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        if token_mask.shape != (x.shape[0], x.shape[2]) or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be bool [batch, bands]")
        x = x.masked_fill(~token_mask[:, None, :, None], 0)
        for time_block, freq_block in self.layers:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(self._axes, x, token_mask, time_block, freq_block, use_reentrant=False)
            else:
                x = self._axes(x, token_mask, time_block, freq_block)
        return self.norm(x).masked_fill(~token_mask[:, None, :, None], 0)


MUSDB_REGION_CONFIGS = (
    dict(start=0, end=64, target_ratio=2, max_chunks=32),
    dict(start=64, end=128, target_ratio=2, max_chunks=32),
    dict(start=128, end=576, target_ratio=14, max_chunks=32),
    dict(start=576, end=1025, target_ratio=14.03125, max_chunks=32),
)


class DBSBSRoformer(DBSBSRNN):
    """DBS front/back end with an :class:`AxialRoformerSeparator`.

    Defaults follow the MUSDB18-HQ recipe: stereo input, four stems, and four
    regions of 32 tokens each.
    """

    def __init__(self, dim: int = 384, depth: int = 6, *, heads: int = 8, dim_head: int = 64,
                 ff_mult: float = 4, **kwargs):
        if dim < 2 or dim % 2:
            raise ValueError("DBS requires a positive even dim")
        for key in ("rnn_hidden_size", "rnn_type", "bidirectional"):
            if key in kwargs:
                raise TypeError(f"{key} is an RNN option, not a RoFormer option")
        kwargs.setdefault("stereo", True)
        kwargs.setdefault("num_stems", 4)
        kwargs.setdefault("region_configs", MUSDB_REGION_CONFIGS)
        kwargs.setdefault("squeeze_mono_output", False)
        self._roformer = dict(heads=heads, dim_head=dim_head, ff_mult=ff_mult)
        super().__init__(dim, depth, rnn_hidden_size=None, **kwargs)

    def _build_separator(self, *, dim, depth, dropout, **_):
        return AxialRoformerSeparator(dim, depth, dropout=dropout,
                                      activation_checkpointing=self.activation_checkpointing, **self._roformer)
