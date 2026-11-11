"""DBS front end and mask decoder for the BSRNN and BS-RoFormer backbones.

For each coarse frequency region, :class:`BoundaryTokenizer` scores every STFT
bin as a band boundary, keeps a fixed number of boundaries with hard Top-K,
and pools each variable-width band into one token. After separation,
:class:`TokenMaskDecoder` smooths the tokens, expands them back to the STFT
grid with a straight-through assignment estimator, refines them along
frequency, and decodes complex masks with a frequency-shared head.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from .band_split import desired_token_count


def segment_ids(hard: Tensor, num_slots: int) -> Tensor:
    """Token slot of every bin ``[B, F]``: each boundary starts a new token."""
    return (torch.cumsum((hard.detach() > 0.5).long(), dim=1) - 1).clamp(0, num_slots - 1)


def flat_segment_ids(seg: Tensor, num_slots: int) -> Tensor:
    """Offset slot indices by batch so all segments share one flat index space."""
    offsets = torch.arange(seg.shape[0], device=seg.device).unsqueeze(1) * num_slots
    return (seg + offsets).reshape(-1)


def segment_softmax_pool(features: Tensor, scores: Tensor, flat_ids: Tensor, num_segments: int) -> Tensor:
    """Pool ``features [B, F, D, T]`` into ``num_segments`` tokens.

    ``scores [B, F]`` are softmax-normalized within each segment, so every
    token is a content-weighted average of the bins it covers.
    """
    bsz, freq, dim, time = features.shape
    flat_scores = scores.reshape(-1)
    max_scores = torch.full((num_segments,), torch.finfo(flat_scores.dtype).min,
                            device=features.device, dtype=flat_scores.dtype)
    max_scores.scatter_reduce_(0, flat_ids, flat_scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(flat_scores - max_scores.gather(0, flat_ids))
    denominator = torch.zeros(num_segments, device=features.device, dtype=exp_scores.dtype)
    denominator.scatter_add_(0, flat_ids, exp_scores)
    weights = exp_scores / denominator.gather(0, flat_ids).clamp(min=1e-8)

    weighted = features.reshape(bsz * freq, dim, time) * weights.to(features.dtype).view(bsz * freq, 1, 1)
    pooled = torch.zeros(num_segments, dim, time, device=features.device, dtype=weighted.dtype)
    pooled.scatter_add_(0, flat_ids.view(bsz * freq, 1, 1).expand(-1, dim, time), weighted)
    return pooled


def topk_boundaries(probs: Tensor, num_tokens: int) -> Tensor:
    """Binary boundary map: bin 0 plus the ``num_tokens - 1`` most probable bins."""
    hard = probs.new_zeros(probs.shape)
    hard[:, 0] = 1.0
    internal = min(num_tokens - 1, probs.shape[1] - 1)
    if internal > 0:
        hard.scatter_(1, probs[:, 1:].topk(internal, dim=1).indices + 1, 1.0)
    return hard


class BoundaryTokenizer(nn.Module):
    """Boundary prediction, hard Top-K selection, and band pooling for one region.

    ``max_chunks`` is the number of token slots the region occupies in the
    separator input; the hard Top-K fills ``round(width / target_ratio)`` of
    them (capped at ``max_chunks``) and the rest stay masked.
    """

    def __init__(
        self,
        *,
        band_width: int,
        dim: int,
        audio_channels: int,
        target_ratio: float,
        max_chunks: int,
        energy_feature_mode: str = "mean_peak_std",
        enforce_chunk_budget: bool = True,
        boundary_temperature: float = 6.0,
        use_explicit_energy: bool = True,
        energy_delta_bias_weight: float = 0.15,
    ):
        super().__init__()
        if energy_feature_mode not in ("mean_peak_std", "mean"):
            raise ValueError(f"unknown energy_feature_mode {energy_feature_mode!r}")
        self.band_width = int(band_width)
        self.input_channels = 2 * int(audio_channels)
        self.target_ratio = float(target_ratio)
        self.max_chunks = int(max_chunks)
        self.energy_feature_mode = energy_feature_mode
        self.enforce_chunk_budget = bool(enforce_chunk_budget)
        self.use_explicit_energy = bool(use_explicit_energy)
        self.energy_delta_bias_weight = float(energy_delta_bias_weight)
        self.num_tokens = desired_token_count(self.band_width, self.target_ratio, self.max_chunks)

        # Boundary scorer: conv features (+ energy descriptors) summarized over time by a BiLSTM.
        self.router = nn.Sequential(
            nn.Conv2d(self.input_channels, dim // 2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.GroupNorm(1, dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        num_energy = 3 if energy_feature_mode == "mean_peak_std" else 1
        energy_dim = max(1, dim // 4)
        self.energy_proj = (
            nn.Sequential(nn.Linear(num_energy, energy_dim), nn.GELU())
            if self.use_explicit_energy else None
        )
        self.temporal_rnn = nn.LSTM(
            input_size=dim + (energy_dim if self.use_explicit_energy else 0),
            hidden_size=dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.boundary_temperature = nn.Parameter(torch.tensor(float(boundary_temperature)))
        self.boundary_bias = nn.Parameter(torch.zeros(self.band_width))

        # Token content path.
        self.encoder_proj = nn.Sequential(
            nn.GroupNorm(1, self.input_channels),
            nn.Conv2d(self.input_channels, dim, kernel_size=1),
        )
        self.chunk_attn = nn.Sequential(
            nn.Linear(dim, max(1, dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, dim // 2), 1),
        )

    def _energy_features(self, spec_ri: Tensor) -> Tensor:
        magnitude = torch.sqrt(spec_ri.square().sum(dim=1).clamp(min=1e-8))
        mean = magnitude.mean(dim=-1)
        if self.energy_feature_mode == "mean":
            return torch.log1p(mean).unsqueeze(-1)
        return torch.stack([
            torch.log1p(mean),
            torch.log1p(magnitude.amax(dim=-1) / mean.clamp(min=1e-6)),
            torch.log1p(magnitude.std(dim=-1, unbiased=False) / mean.clamp(min=1e-6)),
        ], dim=-1)

    def boundary_probs(self, spec_ri: Tensor) -> Tensor:
        """Boundary probability of every bin, ``[B, F]``; bin 0 always starts a band."""
        bsz, _, freq, _ = spec_ri.shape
        feats = rearrange(self.router(spec_ri), "b d f t -> (b f) t d")
        if self.energy_proj is not None:
            energy = self.energy_proj(self._energy_features(spec_ri))
            energy = rearrange(energy, "b f d -> (b f) 1 d").expand(-1, feats.shape[1], -1)
            feats = torch.cat([feats, energy], dim=-1)
        _, (hidden, _) = self.temporal_rnn(feats)
        router_feat = rearrange(torch.cat([hidden[-2], hidden[-1]], dim=-1), "(b f) d -> b f d", b=bsz, f=freq)

        # Content dissimilarity between adjacent bins.
        q = F.normalize(self.to_q(router_feat), dim=-1)
        k = F.normalize(self.to_k(router_feat), dim=-1)
        score = 0.5 * (1.0 - (q[:, 1:] * k[:, :-1]).sum(dim=-1))
        if self.use_explicit_energy:
            # Acoustic transition prior: energy change between adjacent bins.
            energy = self._energy_features(spec_ri)
            delta = (energy[:, 1:] - energy[:, :-1]).abs().mean(dim=-1)
            score = score + self.energy_delta_bias_weight * torch.tanh(delta)
        score = F.pad(score, (1, 0), value=1.0)
        logits = self.boundary_temperature.clamp(min=1.0) * (score - 0.5) + self.boundary_bias
        logits[:, 0] = 10.0
        return torch.sigmoid(logits)

    def route(self, spec_ri: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(hard_boundaries, boundary_probs)``, both ``[B, F]``."""
        probs = self.boundary_probs(spec_ri)
        if self.enforce_chunk_budget:
            return topk_boundaries(probs, self.num_tokens), probs
        hard = (probs >= 0.5).to(probs.dtype)
        hard[:, 0] = 1.0
        return hard, probs

    def ratio_loss(self, probs: Tensor) -> Tensor:
        """Keeps the mean boundary probability near the target token density."""
        return (probs.mean(dim=1) - 1.0 / max(self.target_ratio, 1.01)).square().mean()

    def forward(self, spec_ri: Tensor):
        """``spec_ri [B, 2C, F, T]`` -> ``(tokens, token_probs, hard, probs, token_mask, ratio_loss)``.

        ``tokens`` is ``[B, max_chunks, D, T]``; ``token_probs`` and
        ``token_mask`` are ``[B, max_chunks]``.
        """
        hard, probs = self.route(spec_ri)
        features = rearrange(self.encoder_proj(spec_ri), "b d f t -> b f d t")
        bsz, _, dim, time = features.shape
        num_segments = bsz * self.max_chunks
        flat_ids = flat_segment_ids(segment_ids(hard, self.max_chunks), self.max_chunks)

        scores = self.chunk_attn(features.mean(dim=-1)).squeeze(-1)
        tokens = segment_softmax_pool(features, scores, flat_ids, num_segments).view(bsz, self.max_chunks, dim, time)

        counts = torch.zeros(num_segments, device=probs.device, dtype=probs.dtype)
        counts.scatter_add_(0, flat_ids, torch.ones_like(probs.reshape(-1)))
        counts = counts.view(bsz, self.max_chunks)
        token_mask = counts > 0
        prob_sum = torch.zeros(num_segments, device=probs.device, dtype=probs.dtype)
        prob_sum.scatter_add_(0, flat_ids, probs.reshape(-1))
        token_probs = prob_sum.view(bsz, self.max_chunks) / counts.clamp(min=1.0)
        token_probs = token_probs * token_mask.to(token_probs.dtype)
        return tokens, token_probs, hard, probs, token_mask, self.ratio_loss(probs)


class TokenMaskDecoder(nn.Module):
    """Token smoothing and expansion, frequency refinement, and complex mask decoding for one region."""

    def __init__(
        self,
        *,
        dim: int,
        num_stems: int,
        audio_channels: int,
        band_width: int,
        refine_layers: int = 1,
        refine_kernel_size: int = 3,
        use_mixture_context: bool = False,
        use_confidence_path: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_stems = int(num_stems)
        self.audio_channels = int(audio_channels)
        self.band_width = int(band_width)
        self.use_confidence_path = bool(use_confidence_path)

        # Optional projection of the full-resolution mixture spectrum (MUSDB DBS-BSRNN).
        self.spec_context_proj = (
            nn.Sequential(
                nn.Conv2d(2 * self.audio_channels + 1, dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, dim),
                nn.PReLU(),
            )
            if use_mixture_context else None
        )
        padding = int(refine_kernel_size) // 2
        self.refine_blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                dw=nn.Conv2d(dim, dim, kernel_size=(refine_kernel_size, 1), padding=(padding, 0), groups=dim, bias=False),
                pw=nn.Conv2d(dim, dim, kernel_size=1),
                norm=nn.GroupNorm(1, dim),
                act=nn.PReLU(),
            ))
            for _ in range(int(refine_layers))
        ])
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(dim * 2, self.num_stems * self.audio_channels * 4, kernel_size=1),
        )

    @staticmethod
    def smooth(tokens: Tensor, token_probs: Tensor, token_mask: Tensor) -> Tensor:
        """Probability-guided recursive smoothing across adjacent tokens."""
        bsz, count = tokens.shape[:2]
        outs = [tokens[:, 0]]
        for index in range(1, count):
            p = token_probs[:, index].view(bsz, 1, 1)
            candidate = p * tokens[:, index] + (1.0 - p) * outs[-1]
            valid = token_mask[:, index].view(bsz, 1, 1)
            outs.append(torch.where(valid, candidate, torch.zeros_like(candidate)))
        return torch.stack(outs, dim=1)

    @staticmethod
    def assignment_ste(expanded: Tensor, tokens: Tensor, seg: Tensor, is_boundary: Tensor,
                       probs: Tensor, token_mask: Tensor) -> Tensor:
        """Hard-forward / soft-backward estimator for the bin-to-token assignment.

        The forward value is the hard gather. In the backward pass each
        bin's assignment is replaced by its expectation under ``probs``: the
        assigned token is compared with the token the opposite boundary
        decision would have chosen (previous token at a boundary, next token
        otherwise).
        """
        if not (torch.is_grad_enabled() and probs.requires_grad):
            return expanded
        bsz, freq = seg.shape
        count = tokens.shape[1]
        _, _, dim, time = expanded.shape
        alt = torch.where(is_boundary, seg - 1, seg + 1)
        valid = (alt >= 0) & (alt < count)
        valid = valid & torch.gather(token_mask, 1, alt.clamp(0, count - 1))
        alt = torch.where(valid, alt, seg)
        alternative = torch.gather(tokens, 1, alt.view(bsz, freq, 1, 1).expand(bsz, freq, dim, time))
        flag = is_boundary.view(bsz, freq, 1, 1)
        p = probs.view(bsz, freq, 1, 1)
        if_boundary = torch.where(flag, expanded, alternative)
        if_no_boundary = torch.where(flag, alternative, expanded)
        soft = p * if_boundary + (1.0 - p) * if_no_boundary
        return expanded + (soft - soft.detach())

    def forward(self, tokens: Tensor, hard: Tensor, probs: Tensor, token_probs: Tensor,
                token_mask: Tensor, spec_complex: Tensor) -> Tensor:
        """Separated tokens ``[B, max_chunks, D, T]`` -> masked spectra ``[B, stems, C, F, T]``."""
        bsz, count, dim, time = tokens.shape
        freq = self.band_width
        smoothed = self.smooth(tokens, token_probs, token_mask)
        is_boundary = hard.detach() > 0.5
        seg = (torch.cumsum(is_boundary.long(), dim=1) - 1).clamp(0, count - 1)
        expanded = torch.gather(smoothed, 1, seg.view(bsz, freq, 1, 1).expand(bsz, freq, dim, time))
        expanded = self.assignment_ste(expanded, smoothed, seg, is_boundary, probs, token_mask)

        confidence = torch.where(is_boundary, probs, 1.0 - probs).view(bsz, freq, 1, 1)
        if self.use_confidence_path:
            expanded = expanded * (confidence + (1.0 - confidence).detach())

        z = rearrange(expanded, "b f d t -> b d f t")
        if self.spec_context_proj is not None:
            spec_ri = rearrange(torch.view_as_real(spec_complex), "b s f t c -> b (s c) f t")
            confidence_map = confidence.expand(bsz, freq, 1, time).squeeze(2)
            z = z + self.spec_context_proj(torch.cat([spec_ri, confidence_map.unsqueeze(1)], dim=1))
        for block in self.refine_blocks:
            z = block["norm"](z + block["pw"](block["act"](block["dw"](z))))

        params = self.mask_net(rearrange(z, "b d f t -> (b f) d t"))
        params = rearrange(params, "(b f) (n s m g) t -> b n s f m g t", b=bsz, f=freq,
                           n=self.num_stems, s=self.audio_channels, m=2, g=2)
        mask_real = params[..., 0, 0, :] * torch.sigmoid(params[..., 0, 1, :])
        mask_imag = params[..., 1, 0, :] * torch.sigmoid(params[..., 1, 1, :])
        if self.num_stems > 1:
            # Mixture consistency: the stem masks sum to 1 + 0j.
            mask_real = mask_real - (mask_real.sum(dim=1, keepdim=True) - 1.0) / self.num_stems
            mask_imag = mask_imag - mask_imag.sum(dim=1, keepdim=True) / self.num_stems
        spec_real = spec_complex.real.unsqueeze(1)
        spec_imag = spec_complex.imag.unsqueeze(1)
        return torch.complex(spec_real * mask_real - spec_imag * mask_imag,
                             spec_real * mask_imag + spec_imag * mask_real)
