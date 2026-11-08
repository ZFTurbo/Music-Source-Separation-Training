"""STFT helpers shared by the band-split models."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange


class STFT(nn.Module):
    """Multichannel STFT / iSTFT with a Hann window.

    ``forward`` maps audio ``[B, C, T]`` to a complex spectrum ``[B, C, F, frames]``;
    ``inverse`` maps ``[B, N, C, F, frames]`` back to ``[B, N, C, T]``.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, win_length: int | None = None,
                 normalized: bool = False):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.normalized = bool(normalized)
        self.num_freqs = self.n_fft // 2 + 1

    def _kwargs(self, device) -> dict:
        return dict(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                    normalized=self.normalized, window=torch.hann_window(self.win_length, device=device))

    def forward(self, audio: Tensor) -> Tensor:
        channels = audio.shape[1]
        spec = torch.stft(rearrange(audio, "b c t -> (b c) t"), return_complex=True, **self._kwargs(audio.device))
        return rearrange(spec, "(b c) f t -> b c f t", c=channels)

    def inverse(self, spec: Tensor, length: int) -> Tensor:
        lead = spec.shape[:-2]
        audio = torch.istft(spec.reshape(-1, *spec.shape[-2:]), length=length, return_complex=False,
                            **self._kwargs(spec.device))
        return audio.reshape(*lead, length)


def multi_resolution_stft_loss(estimate: Tensor, target: Tensor, window_sizes, hop_length: int,
                               n_fft_floor: int) -> Tensor:
    """Sum of complex-STFT L1 distances over several window sizes (MSST recipe)."""
    estimate = rearrange(estimate, "... t -> (...) t")
    target = rearrange(target, "... t -> (...) t")
    losses = []
    for window in window_sizes:
        if window <= 1 or window > estimate.shape[-1]:
            continue
        kwargs = dict(n_fft=max(window, n_fft_floor), hop_length=hop_length, win_length=window,
                      window=torch.hann_window(window, device=estimate.device), return_complex=True)
        losses.append(F.l1_loss(torch.stft(estimate, **kwargs), torch.stft(target, **kwargs)))
    return torch.stack(losses).sum() if losses else estimate.new_zeros(())
