"""DBS-BSRNN: the BSRNN separator with Dynamic Band Splitting."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .modules.band_split import validate_region_configs
from .modules.dbs import BoundaryTokenizer, TokenMaskDecoder
from .modules.stft import STFT, multi_resolution_stft_loss


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Normalize in fp32: an fp16 reduction can overflow before the epsilon guard applies.
        norm = x.float().square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12).sqrt()
        return (x.float() / norm * self.scale * self.gamma.float()).to(dtype=x.dtype)


def MLP(dim_in: int, dim_out: int, dim_hidden: int | None = None, depth: int = 1,
        activation=nn.Tanh) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden or dim_in,) * max(0, depth - 1)), dim_out)
    layers: list[nn.Module] = []
    for index, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if index < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class AxisRNNBlock(nn.Module):
    """Pre-norm residual BiLSTM/GRU along one axis."""

    def __init__(self, dim: int, hidden_size: int | None = None, rnn_type: str = "lstm",
                 bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        hidden_size = hidden_size or dim
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.norm = nn.LayerNorm(dim)
        self.rnn = rnn_cls(input_size=dim, hidden_size=hidden_size, num_layers=1,
                           batch_first=True, bidirectional=bidirectional)
        self.to_out = nn.Sequential(nn.Linear(hidden_size * (2 if bidirectional else 1), dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(self.norm(x))
        return x + self.to_out(out)


class BSRNNSeparator(nn.Module):
    """Interleaved time-axis and band-axis RNNs over band tokens ``[B, T, N, D]``."""

    def __init__(self, dim: int, depth: int, hidden_size: int | None = None, rnn_type: str = "lstm",
                 bidirectional: bool = True, dropout: float = 0.0, activation_checkpointing: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                AxisRNNBlock(dim, hidden_size, rnn_type, bidirectional, dropout),
                AxisRNNBlock(dim, hidden_size, rnn_type, bidirectional, dropout),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.activation_checkpointing = bool(activation_checkpointing)

    def _run(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        batch, frames, bands, _ = x.shape
        mask = None if token_mask is None else token_mask[:, None, :, None].to(x.dtype)
        if mask is not None:
            x = x * mask
        for time_rnn, band_rnn in self.layers:
            x = rearrange(x, "b t f d -> (b f) t d")
            x = rearrange(self._run(time_rnn, x), "(b f) t d -> b t f d", b=batch, f=bands)
            if mask is not None:
                x = x * mask
            x = rearrange(x, "b t f d -> (b t) f d")
            x = rearrange(self._run(band_rnn, x), "(b t) f d -> b t f d", b=batch, t=frames)
            if mask is not None:
                x = x * mask
        return self.norm(x)


class _BandSplitModel(nn.Module):
    """STFT front/back end and the MSST-style training loss shared by the band-split models."""

    def _init_common(self, *, stereo: bool, num_stems: int, sample_rate: int, stft_n_fft: int,
                     stft_hop_length: int, stft_win_length: int, stft_normalized: bool,
                     multi_stft_resolution_loss_weight: float,
                     multi_stft_resolutions_window_sizes: Sequence[int], multi_stft_hop_size: int,
                     squeeze_mono_output: bool):
        self.stereo = bool(stereo)
        self.audio_channels = 2 if stereo else 1
        self.num_stems = int(num_stems)
        self.sample_rate = int(sample_rate)
        self.squeeze_mono_output = bool(squeeze_mono_output)
        self.stft = STFT(stft_n_fft, stft_hop_length, stft_win_length, stft_normalized)
        self.multi_stft_resolution_loss_weight = float(multi_stft_resolution_loss_weight)
        self.multi_stft_resolutions_window_sizes = tuple(int(w) for w in multi_stft_resolutions_window_sizes)
        self.multi_stft_hop_size = int(multi_stft_hop_size)

    def _spectrum(self, audio: Tensor) -> Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.shape[1] != self.audio_channels:
            raise ValueError(f"expected {self.audio_channels} audio channel(s), got {audio.shape[1]}")
        return self.stft(audio)

    def _reconstruct(self, spec: Tensor, length: int) -> Tensor:
        """``spec [B, N, C, F, T]`` -> audio ``[B, N, C, L]``, squeezed like the legacy models."""
        audio = self.stft.inverse(spec, length)
        if self.num_stems == 1:
            return audio[:, 0]
        if self.audio_channels == 1 and self.squeeze_mono_output:
            return audio[:, :, 0]
        return audio

    def reconstruction_loss(self, estimate: Tensor, target: Tensor) -> Tensor:
        """L1 waveform loss plus the weighted multi-resolution STFT loss (MSST recipe)."""
        if target.ndim == 2:
            target = target.unsqueeze(1)
        # MSSDataset preserves a source axis even for a one-stem run.  DBS
        # intentionally squeezes that axis on its inference result, so remove
        # it from the target before computing waveform and spectral losses.
        if self.num_stems == 1 and estimate.ndim == 3 and target.ndim == 4:
            target = target[:, 0]
        if estimate.ndim == 4 and target.ndim == 3:
            target = target.unsqueeze(2)
        target = target[..., :estimate.shape[-1]]
        loss = F.l1_loss(estimate, target)
        if self.multi_stft_resolution_loss_weight > 0:
            loss = loss + self.multi_stft_resolution_loss_weight * multi_resolution_stft_loss(
                estimate, target, self.multi_stft_resolutions_window_sizes, self.multi_stft_hop_size,
                self.stft.n_fft)
        return loss


class DBSBSRNN(_BandSplitModel):
    """BSRNN with Dynamic Band Splitting.

    The spectrum is divided into coarse regions (``region_configs``). Each
    region predicts its own band boundaries under a fixed token budget; the
    tokens of all regions are separated jointly by one shared separator and
    decoded back to full-resolution complex masks.

    ``forward(mixture)`` returns the separated audio; ``return_aux=True``
    returns ``(estimate, aux)`` with the boundary ratio loss and the routes.
    ``forward(mixture, target)`` returns the MSST training loss.
    """

    def __init__(
        self,
        dim: int = 128,
        depth: int = 6,
        *,
        num_stems: int = 3,
        stereo: bool = False,
        sample_rate: int = 44100,
        region_configs: Sequence[Mapping[str, Any]] = (
            dict(start=0, end=64, target_ratio=2, max_chunks=64),
            dict(start=64, end=128, target_ratio=2, max_chunks=64),
            dict(start=128, end=576, target_ratio=8, max_chunks=32),
            dict(start=576, end=1025, target_ratio=8, max_chunks=32),
        ),
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        rnn_hidden_size: int | None = 128,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.0,
        energy_feature_mode: str = "mean_peak_std",
        use_explicit_energy: bool = True,
        energy_delta_bias_weight: float = 0.15,
        enforce_chunk_budget: bool = True,
        dechunker_refine_layers: int = 1,
        dechunker_refine_kernel_size: int = 3,
        use_mixture_context: bool = False,
        use_confidence_path: bool = True,
        ratio_loss_weight: float = 0.0,
        multi_stft_resolution_loss_weight: float = 0.0,
        multi_stft_resolutions_window_sizes: Sequence[int] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        squeeze_mono_output: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self._init_common(stereo=stereo, num_stems=num_stems, sample_rate=sample_rate, stft_n_fft=stft_n_fft,
                          stft_hop_length=stft_hop_length, stft_win_length=stft_win_length,
                          stft_normalized=stft_normalized,
                          multi_stft_resolution_loss_weight=multi_stft_resolution_loss_weight,
                          multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
                          multi_stft_hop_size=multi_stft_hop_size, squeeze_mono_output=squeeze_mono_output)
        self.region_configs = [dict(cfg) for cfg in region_configs]
        validate_region_configs(self.region_configs, self.stft.num_freqs)
        self.band_slices = [(int(cfg["start"]), int(cfg["end"])) for cfg in self.region_configs]
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.activation_checkpointing = bool(activation_checkpointing)

        self.chunkers = nn.ModuleList()
        self.dechunkers = nn.ModuleList()
        for cfg, (start, end) in zip(self.region_configs, self.band_slices):
            self.chunkers.append(BoundaryTokenizer(
                band_width=end - start, dim=dim, audio_channels=self.audio_channels,
                target_ratio=float(cfg["target_ratio"]), max_chunks=int(cfg["max_chunks"]),
                energy_feature_mode=energy_feature_mode, enforce_chunk_budget=enforce_chunk_budget,
                use_explicit_energy=use_explicit_energy,
                energy_delta_bias_weight=energy_delta_bias_weight,
            ))
            self.dechunkers.append(TokenMaskDecoder(
                dim=dim, num_stems=self.num_stems, audio_channels=self.audio_channels, band_width=end - start,
                refine_layers=dechunker_refine_layers, refine_kernel_size=dechunker_refine_kernel_size,
                use_mixture_context=use_mixture_context, use_confidence_path=use_confidence_path,
            ))
        self.separator = self._build_separator(dim=dim, depth=depth, rnn_hidden_size=rnn_hidden_size,
                                               rnn_type=rnn_type, bidirectional=bidirectional, dropout=dropout)

    def _build_separator(self, *, dim, depth, rnn_hidden_size, rnn_type, bidirectional, dropout, **_):
        return BSRNNSeparator(dim, depth, rnn_hidden_size, rnn_type, bidirectional, dropout,
                              self.activation_checkpointing)

    def _checkpointed(self, function, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def forward(self, mixture: Tensor, target: Tensor | None = None, return_aux: bool = False,
                active_stem_ids=None):
        """``mixture [B, C, L]`` -> separated audio (``[B, N, L]`` for mono)."""
        spec = self._spectrum(mixture)
        spec_ri = torch.view_as_real(spec)

        tokens, masks, states = [], [], []
        ratio_loss = spec_ri.new_tensor(0.0)
        for (start, end), chunker in zip(self.band_slices, self.chunkers):
            region = rearrange(spec_ri[:, :, start:end], "b s f t c -> b (s c) f t")
            x, token_probs, hard, probs, token_mask, region_ratio = self._checkpointed(chunker, region)
            tokens.append(rearrange(x, "b n d t -> b t n d"))
            masks.append(token_mask)
            states.append((token_probs, hard, probs, token_mask, x.shape[1]))
            ratio_loss = ratio_loss + region_ratio
        ratio_loss = ratio_loss / len(self.band_slices)

        separated = rearrange(self.separator(torch.cat(tokens, dim=2), token_mask=torch.cat(masks, dim=1)),
                              "b t n d -> b n d t")
        region_specs, offset = [], 0
        for (start, end), dechunker, (token_probs, hard, probs, token_mask, count) in zip(
                self.band_slices, self.dechunkers, states):
            region_specs.append(self._checkpointed(
                dechunker, separated[:, offset:offset + count], hard, probs, token_probs, token_mask,
                spec[:, :, start:end]))
            offset += count
        estimate = self._reconstruct(torch.cat(region_specs, dim=3), mixture.shape[-1])

        if target is not None:
            return self.reconstruction_loss(estimate, target) + self.ratio_loss_weight * ratio_loss
        if not return_aux:
            return estimate
        aux = dict(
            ratio_loss=ratio_loss,
            boundary_probs=[s[2] for s in states],
            hard_boundaries=[s[1] for s in states],
            token_masks=[s[3] for s in states],
            band_slices=list(self.band_slices),
        )
        return estimate, aux
