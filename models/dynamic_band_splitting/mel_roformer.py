"""DBS-Mel-RoFormer.

:class:`DBSMelRoformer` keeps the Mel-Band RoFormer transformer trunk (axial
time/band transformers with value residuals) but starts from 128
contiguous bands obtained by resampling the BSRNN band layout. A router
predicts, from the mixture magnitude, how far the interior coarse-region
boundaries move; every fine band edge follows the coarse route, and bins are
pooled into and scattered out of the 128 bands with soft memberships, so the
token count stays fixed.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding

from .modules.stft import multi_resolution_stft_loss

BSRNN_TEMPLATE_BAND_WIDTHS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
COARSE_BOUNDARIES = (0, 64, 128, 576, 1025)


def bsrnn_prior_band_boundaries(num_bands: int = 128, template_widths: Sequence[int] = BSRNN_TEMPLATE_BAND_WIDTHS,
                                freq_bins: int = 1025) -> tuple[int, ...]:
    """Resample the 62-band BSRNN layout to ``num_bands`` contiguous bands (``num_bands + 1`` edges)."""
    template = tuple(int(w) for w in template_widths)
    if sum(template) != freq_bins or not 2 <= num_bands < freq_bins:
        raise ValueError("invalid template or band count")
    cumulative = [0]
    for width in template:
        cumulative.append(cumulative[-1] + width)
    scale = len(template) / num_bands
    raw = []
    for index in range(num_bands + 1):
        position = scale * index
        slot = min(int(position), len(template) - 1)
        alpha = position - slot
        raw.append(int(round(cumulative[slot] + alpha * (cumulative[slot + 1] - cumulative[slot]))))
    edges = [0] * (num_bands + 1)
    for index in range(1, num_bands):
        edges[index] = min(max(raw[index], edges[index - 1] + 1), freq_bins - (num_bands - index))
    edges[-1] = freq_bins
    return tuple(edges)


# --------------------------------------------------------------------------- transformer trunk


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Gated multi-head attention with rotary embeddings and an optional learned value residual."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0,
                 rotary_embed: RotaryEmbedding | None = None, add_value_residual: bool = False):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.learned_value_residual_mix = (
            nn.Sequential(nn.Linear(dim, heads), nn.Sigmoid()) if add_value_residual else None
        )
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        first_values = v
        if self.learned_value_residual_mix is not None:
            mix = rearrange(self.learned_value_residual_mix(x), "b n h -> b h n 1")
            # As trained for the paper: v + value_residual * (mix - v). Upstream BS-RoFormer gates the
            # residual the other way, v.lerp(value_residual, mix); switching would change the trained
            # model, so the paper's form is kept.
            v = v.lerp(mix, value_residual)
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out * rearrange(self.to_gates(x), "b n h -> b h n 1").sigmoid()
        return self.to_out(rearrange(out, "b h n d -> b n (h d)")), first_values


class Transformer(nn.Module):
    def __init__(self, *, dim: int, depth: int, dim_head: int = 64, heads: int = 8, attn_dropout: float = 0.0,
                 ff_dropout: float = 0.0, rotary_embed: RotaryEmbedding | None = None,
                 add_value_residual: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, attn_dropout, rotary_embed, add_value_residual),
                FeedForward(dim, dropout=ff_dropout),
            ])
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor]:
        first_values = None
        for attn, ff in self.layers:
            out, values = attn(x, value_residual=value_residual)
            x = out + x
            first_values = values if first_values is None else first_values
            x = ff(x) + x
        return self.norm(x), first_values


# --------------------------------------------------------------------------- DBS routing for Mel-RoFormer


class CoarseBoundaryRouter(nn.Module):
    """Predict bounded shifts of the interior coarse-region boundaries from the mixture magnitude.

    The output layer is zero-initialized, so training starts on the static layout.
    """

    def __init__(self, *, freq_bins: int, coarse_boundaries: Sequence[int], max_shift_bins: float = 16.0,
                 hidden_dim: int = 128):
        super().__init__()
        boundaries = torch.as_tensor(coarse_boundaries, dtype=torch.float32)
        if boundaries[0] != 0 or int(boundaries[-1]) != freq_bins or not bool(torch.all(boundaries[1:] > boundaries[:-1])):
            raise ValueError("coarse boundaries must increase from 0 to freq_bins")
        if max_shift_bins <= 0 or max_shift_bins >= float((boundaries[1:] - boundaries[:-1]).min()) / 2:
            raise ValueError("max_shift_bins must be positive and below half the narrowest coarse region")
        self.freq_bins = int(freq_bins)
        self.max_shift_bins = float(max_shift_bins)
        self.register_buffer("static_boundaries", boundaries, persistent=True)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(32 * 16, hidden_dim), nn.SiLU(),
        )
        self.to_offsets = nn.Linear(hidden_dim, boundaries.numel() - 2)
        nn.init.zeros_(self.to_offsets.weight)
        nn.init.zeros_(self.to_offsets.bias)

    def forward(self, log_magnitude: Tensor) -> Tensor:
        """``log_magnitude [B, F]`` -> dynamic coarse boundaries ``[B, regions + 1]``."""
        batch = log_magnitude.shape[0]
        offsets = torch.tanh(self.to_offsets(self.encoder(log_magnitude.unsqueeze(1)))) * self.max_shift_bins
        internal = self.static_boundaries[1:-1].to(log_magnitude.dtype).unsqueeze(0) + offsets.to(log_magnitude.dtype)
        zeros = torch.zeros(batch, 1, device=log_magnitude.device, dtype=log_magnitude.dtype)
        ends = torch.full((batch, 1), float(self.freq_bins), device=log_magnitude.device, dtype=log_magnitude.dtype)
        return torch.cat((zeros, internal, ends), dim=1)


def route_fine_boundaries(static_fine: Tensor, dynamic_coarse: Tensor, static_coarse: Tensor) -> Tensor:
    """Map every fine band edge through the piecewise-linear coarse route."""
    regions = static_coarse.numel() - 1
    region = torch.searchsorted(static_coarse[1:-1].contiguous(), static_fine, right=True).clamp(0, regions - 1)
    left, right = static_coarse[region], static_coarse[region + 1]
    u = ((static_fine - left) / (right - left).clamp_min(1e-6)).clamp(0.0, 1.0)
    dyn_left, dyn_right = dynamic_coarse[..., region], dynamic_coarse[..., region + 1]
    return dyn_left + u * (dyn_right - dyn_left)


def soft_band_membership(boundaries: Tensor, freq_bins: int, temperature: float, normalize: str) -> Tensor:
    """Sigmoid membership ``[..., K, F]`` of bin centres in bands ``[edge_k, edge_k+1)``.

    ``normalize="analysis"`` makes every band a weighted mean over bins;
    ``"synthesis"`` makes the memberships of every bin sum to one.
    """
    grid = torch.arange(freq_bins, device=boundaries.device, dtype=boundaries.dtype) + 0.5
    left, right = boundaries[..., :-1].unsqueeze(-1), boundaries[..., 1:].unsqueeze(-1)
    membership = torch.sigmoid((grid - left) / temperature) * torch.sigmoid((right - grid) / temperature)
    axis = -1 if normalize == "analysis" else -2
    return membership / membership.sum(dim=axis, keepdim=True).clamp_min(1e-6)


class VariableBandSplit(nn.Module):
    """Routed analysis: per-bin features are pooled into the 128 moving bands with a cosine basis."""

    def __init__(self, *, dim: int, freq_bins: int, audio_channels: int, band_edges: Sequence[int],
                 coarse_boundaries: Sequence[int], latent_dim: int = 32, basis_order: int = 4,
                 temperature: float = 0.25):
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.audio_channels = int(audio_channels)
        self.latent_dim = int(latent_dim)
        self.basis_order = int(basis_order)
        self.temperature = float(temperature)
        self.num_bands = len(band_edges) - 1
        self.register_buffer("static_fine", torch.tensor(band_edges, dtype=torch.float32), persistent=True)
        self.register_buffer("static_coarse", torch.tensor(coarse_boundaries, dtype=torch.float32), persistent=True)
        self.bin_projection = nn.Linear(audio_channels * 3, latent_dim)
        self.activation = nn.SiLU()
        self.position_embedding = nn.Parameter(torch.randn(freq_bins, latent_dim) * 0.02)
        self.band_projection = nn.Sequential(nn.LayerNorm(basis_order * latent_dim),
                                             nn.Linear(basis_order * latent_dim, dim))

    def kernels(self, dynamic_coarse: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Analysis and synthesis kernels ``[B, K, F, P]`` and the routed fine edges ``[B, K + 1]``."""
        fine = route_fine_boundaries(self.static_fine.to(dynamic_coarse.dtype), dynamic_coarse,
                                     self.static_coarse.to(dynamic_coarse.dtype))
        if not bool(torch.all(fine[..., 1:] > fine[..., :-1])):
            raise RuntimeError("routed band edges are not strictly increasing")
        analysis = soft_band_membership(fine, self.freq_bins, self.temperature, "analysis")
        synthesis = soft_band_membership(fine, self.freq_bins, self.temperature, "synthesis")
        grid = torch.arange(self.freq_bins, device=fine.device, dtype=fine.dtype)
        left, right = fine[..., :-1].unsqueeze(-1), fine[..., 1:].unsqueeze(-1)
        positions = ((grid - left) / (right - left).clamp_min(1e-6)).clamp(0.0, 1.0)
        harmonics = torch.arange(self.basis_order, device=fine.device, dtype=fine.dtype)
        basis = torch.cos(math.pi * positions.unsqueeze(-1) * harmonics)
        return analysis.unsqueeze(-1) * basis, synthesis.unsqueeze(-1) * basis, fine

    def forward(self, values: Tensor, pool: Tensor) -> Tensor:
        """``values [B, T, F * C * 2]`` -> band tokens ``[B, T, K, dim]``."""
        batch, frames, _ = values.shape
        structured = values.reshape(batch, frames, self.freq_bins, self.audio_channels, 2)
        magnitude = structured.square().sum(dim=-1).sqrt()
        features = torch.cat([structured.reshape(batch, frames, self.freq_bins, self.audio_channels * 2),
                              magnitude.clamp_min(1e-8).log()], dim=-1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            # fp32: this projection runs at every (frame, bin) and its fp16 gradients can overflow.
            latent = self.bin_projection(features.float())
        latent = self.activation(latent + self.position_embedding)
        pooled = torch.einsum("btfd,bkfp->btkpd", latent.to(pool.dtype), pool)
        pooled = pooled.reshape(batch, frames, self.num_bands, self.basis_order * self.latent_dim)
        return self.band_projection(pooled).to(values.dtype)


class VariableMaskEstimator(nn.Module):
    """Routed synthesis: band coefficients are scattered back to bins and decoded into masks."""

    def __init__(self, *, dim: int, freq_bins: int, audio_channels: int, num_bands: int, latent_dim: int = 32,
                 basis_order: int = 4, depth: int = 2, mlp_expansion_factor: int = 4):
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.audio_channels = int(audio_channels)
        self.latent_dim = int(latent_dim)
        self.basis_order = int(basis_order)
        self.num_bands = int(num_bands)
        hidden = dim * mlp_expansion_factor
        layers: list[nn.Module] = [nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.extend([nn.Linear(hidden, basis_order * latent_dim * 2), nn.GLU(dim=-1)])
        self.band_projection = nn.Sequential(*layers)
        self.bin_decoder = nn.Linear(latent_dim, audio_channels * 2)

    def forward(self, features: Tensor, scatter: Tensor) -> Tensor:
        batch, frames, bands, _ = features.shape
        coefficients = self.band_projection(features).reshape(batch, frames, bands, self.basis_order, self.latent_dim)
        spread = torch.einsum("btkpd,bkfp->btfd", coefficients.to(scatter.dtype), scatter)
        with torch.autocast(device_type=spread.device.type, enabled=False):
            mask = self.bin_decoder(spread.float())
        return mask.reshape(batch, frames, self.freq_bins * self.audio_channels * 2).to(features.dtype)


# --------------------------------------------------------------------------- models


class _MelRoformerTrunk(nn.Module):
    """STFT, axial time/band transformer trunk, and training loss shared by both models."""

    def __init__(self, *, dim: int, depth: int, stereo: bool, num_stems: int, time_transformer_depth: int,
                 freq_transformer_depth: int, dim_head: int, heads: int, attn_dropout: float, ff_dropout: float,
                 stft_n_fft: int, stft_hop_length: int, stft_win_length: int, stft_normalized: bool,
                 multi_stft_resolution_loss_weight: float, multi_stft_resolutions_window_sizes: Sequence[int],
                 multi_stft_hop_size: int, add_value_residual: bool):
        super().__init__()
        self.stereo = bool(stereo)
        self.audio_channels = 2 if stereo else 1
        self.num_stems = int(num_stems)
        self.freq_bins = stft_n_fft // 2 + 1
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length,
                                normalized=stft_normalized)
        self.stft_win_length = stft_win_length
        self.multi_stft_resolution_loss_weight = float(multi_stft_resolution_loss_weight)
        self.multi_stft_resolutions_window_sizes = tuple(int(w) for w in multi_stft_resolutions_window_sizes)
        self.multi_stft_hop_size = int(multi_stft_hop_size)

        kwargs = dict(dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        time_rotary, freq_rotary = RotaryEmbedding(dim=dim_head), RotaryEmbedding(dim=dim_head)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary,
                            add_value_residual=add_value_residual and index > 0, **kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary,
                            add_value_residual=add_value_residual and index > 0, **kwargs),
            ])
            for index in range(depth)
        ])

    def _transformer(self, x: Tensor) -> Tensor:
        """Band tokens ``[B, T, K, D]`` through alternating time and band transformers."""
        time_residual = freq_residual = None
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x, values = time_transformer(x, value_residual=time_residual)
            time_residual = values if time_residual is None else time_residual
            x, = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x, values = freq_transformer(x, value_residual=freq_residual)
            freq_residual = values if freq_residual is None else freq_residual
            x, = unpack(x, ps, "* f d")
        return x

    def _stft(self, audio: Tensor) -> Tensor:
        """``audio [B, C, L]`` -> ``[B, F * C, frames, 2]`` with channels interleaved per bin."""
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.shape[1] != self.audio_channels:
            raise ValueError(f"expected {self.audio_channels} audio channel(s), got {audio.shape[1]}")
        batch = audio.shape[0]
        window = torch.hann_window(self.stft_win_length, device=audio.device)
        spec = torch.view_as_real(torch.stft(audio.reshape(-1, audio.shape[-1]), **self.stft_kwargs,
                                             window=window, return_complex=True))
        spec = spec.reshape(batch, self.audio_channels, *spec.shape[1:])
        return rearrange(spec, "b s f t c -> b (f s) t c")

    def _istft(self, spec: Tensor, length: int) -> Tensor:
        """Complex ``[B, N, F * C, frames]`` -> audio ``[B, N, C, L]`` (``[B, C, L]`` for one stem)."""
        batch = spec.shape[0]
        spec = rearrange(spec, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        window = torch.hann_window(self.stft_win_length, device=spec.device)
        audio = torch.istft(spec, **self.stft_kwargs, window=window, return_complex=False, length=length)
        audio = rearrange(audio, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=self.num_stems)
        return audio[:, 0] if self.num_stems == 1 else audio

    def reconstruction_loss(self, estimate: Tensor, target: Tensor) -> Tensor:
        """L1 waveform loss plus the multi-resolution STFT loss (Mel-RoFormer recipe)."""
        target = target[..., :estimate.shape[-1]]
        loss = F.l1_loss(estimate, target)
        stft_loss = estimate.new_zeros(())
        for window in self.multi_stft_resolutions_window_sizes:
            kwargs = dict(n_fft=max(window, self.stft_kwargs["n_fft"]), win_length=window,
                          hop_length=max(self.multi_stft_hop_size, window // 4), normalized=False,
                          return_complex=True, window=torch.hann_window(window, device=estimate.device))
            stft_loss = stft_loss + F.l1_loss(torch.stft(rearrange(estimate, "... s t -> (... s) t"), **kwargs),
                                              torch.stft(rearrange(target, "... s t -> (... s) t"), **kwargs))
        return loss + stft_loss * self.multi_stft_resolution_loss_weight


class DBSMelRoformer(_MelRoformerTrunk):
    """Mel-RoFormer with Dynamic Band Splitting (MUSDB18-HQ recipe defaults).

    A router shifts the interior ``coarse_boundaries`` by at most
    ``max_shift_bins``; the 128 fine bands follow that route and are pooled
    with soft memberships, so the separator always sees 128 tokens.
    """

    def __init__(self, dim: int = 256, depth: int = 6, *, stereo: bool = True, num_stems: int = 4,
                 num_bands: int = 128, coarse_boundaries: Sequence[int] = COARSE_BOUNDARIES,
                 max_shift_bins: float = 16.0, router_hidden_dim: int = 128, latent_dim: int = 32,
                 basis_order: int = 4, temperature: float = 0.25, decoder_depth: int = 2,
                 decoder_expansion: int = 4, time_transformer_depth: int = 1, freq_transformer_depth: int = 1,
                 dim_head: int = 64, heads: int = 8, attn_dropout: float = 0.1, ff_dropout: float = 0.1,
                 stft_n_fft: int = 2048, stft_hop_length: int = 441, stft_win_length: int = 2048,
                 stft_normalized: bool = False, multi_stft_resolution_loss_weight: float = 1.0,
                 multi_stft_resolutions_window_sizes: Sequence[int] = (4096, 2048, 1024, 512, 256),
                 multi_stft_hop_size: int = 147, add_value_residual: bool = True):
        super().__init__(dim=dim, depth=depth, stereo=stereo, num_stems=num_stems,
                         time_transformer_depth=time_transformer_depth, freq_transformer_depth=freq_transformer_depth,
                         dim_head=dim_head, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                         stft_n_fft=stft_n_fft, stft_hop_length=stft_hop_length, stft_win_length=stft_win_length,
                         stft_normalized=stft_normalized,
                         multi_stft_resolution_loss_weight=multi_stft_resolution_loss_weight,
                         multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
                         multi_stft_hop_size=multi_stft_hop_size, add_value_residual=add_value_residual)
        edges = bsrnn_prior_band_boundaries(num_bands, freq_bins=self.freq_bins)
        coarse = tuple(int(c) for c in coarse_boundaries)
        self.num_bands = int(num_bands)
        self.router = CoarseBoundaryRouter(freq_bins=self.freq_bins, coarse_boundaries=coarse,
                                           max_shift_bins=max_shift_bins, hidden_dim=router_hidden_dim)
        self.band_split = VariableBandSplit(dim=dim, freq_bins=self.freq_bins, audio_channels=self.audio_channels,
                                            band_edges=edges, coarse_boundaries=coarse, latent_dim=latent_dim,
                                            basis_order=basis_order, temperature=temperature)
        self.mask_estimators = nn.ModuleList([
            VariableMaskEstimator(dim=dim, freq_bins=self.freq_bins, audio_channels=self.audio_channels,
                                  num_bands=self.num_bands, latent_dim=latent_dim, basis_order=basis_order,
                                  depth=decoder_depth, mlp_expansion_factor=decoder_expansion)
            for _ in range(self.num_stems)
        ])

    def forward(self, mixture: Tensor, target: Tensor | None = None, return_aux: bool = False,
                active_stem_ids=None):
        """``mixture [B, C, L]`` -> ``[B, stems, C, L]``; with ``target`` returns the training loss."""
        spec = self._stft(mixture)
        values = rearrange(spec, "b f t c -> b t (f c)")
        batch, frames, _ = values.shape
        # The router sees the mixture log-magnitude averaged over time and channels.
        magnitude = values.reshape(batch, frames, self.freq_bins, self.audio_channels, 2).square().sum(dim=-1).sqrt()
        coarse = self.router(magnitude.mean(dim=(1, 3)).clamp_min(1e-8).log1p())
        pool, scatter, fine = self.band_split.kernels(coarse)

        x = self._transformer(self.band_split(values, pool))
        masks = torch.stack([estimator(x, scatter) for estimator in self.mask_estimators], dim=1)
        masks = torch.view_as_complex(rearrange(masks, "b n t (f c) -> b n f t c", c=2).contiguous())
        spec = torch.view_as_complex(spec.unsqueeze(1).contiguous())
        estimate = self._istft(spec * masks, mixture.shape[-1])
        if target is not None:
            # A one-stem model returns [B, C, T], while MSSDataset keeps the
            # explicit source axis as [B, 1, C, T].  Align the target before
            # the shared waveform/STFT loss to avoid accidental broadcasting.
            if self.num_stems == 1 and target.ndim == 4:
                target = target[:, 0]
            return self.reconstruction_loss(estimate, target)
        if return_aux:
            return estimate, dict(coarse_boundaries=coarse, band_edges=fine)
        return estimate
