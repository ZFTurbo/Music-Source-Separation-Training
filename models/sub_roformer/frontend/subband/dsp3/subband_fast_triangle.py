"""Original FIR bank and checkpoint layout with fixed-point modulation."""

import math

import torch
from torch import Tensor

from ..dsp.analytic import analytic_to_real, real_to_analytic
from ..dsp.convolve import polyphase_fftconvolve
from ..dsp.subband_fast_triangle import SubbandFilter as OriginalSubbandFilter
from ..dsp.upsample import polyphase_fftupsample

from .utils import shift_frequency


class SubbandFilter(OriginalSubbandFilter):
    """Reuse the original filters, replacing only the modulation path."""

    def analysis(self, x: Tensor) -> Tensor:
        """Convert [batch, channels, samples] to decimated complex subbands."""
        batch, channels, length = x.shape
        analytic = real_to_analytic(x).reshape(batch * channels, length)
        band_count = len(self.banks)
        downsampled_length = math.ceil(length / self.factor)
        output = torch.empty(
            batch * channels, band_count, downsampled_length,
            dtype=analytic.dtype, device=analytic.device,
        )
        for start in range(0, band_count, self.chunk_size):
            end = min(start + self.chunk_size, band_count)
            subbands = polyphase_fftconvolve(analytic, self.w[start:end], self.factor)
            output[:, start:end] = shift_frequency(
                subbands, -self.f_center[start:end], self.sr, self.factor,
            )
        return output.reshape(batch, channels, band_count, downsampled_length)

    def synthesis(self, x: Tensor) -> Tensor:
        """Reconstruct [batch, channels, samples] from complex subbands."""
        batch, channels, band_count, downsampled_length = x.shape
        subbands = x.reshape(batch * channels, band_count, downsampled_length)
        length = downsampled_length * self.factor
        output = torch.zeros(batch * channels, length, dtype=x.dtype, device=x.device)
        for start in range(0, band_count, self.chunk_size):
            end = min(start + self.chunk_size, band_count)
            band_chunk = subbands[:, start:end].reshape(-1, downsampled_length)
            interpolated = polyphase_fftupsample(band_chunk * self.factor, self.up, self.factor)
            interpolated = interpolated.reshape(batch * channels, end - start, length)
            modulated = shift_frequency(interpolated, self.f_center[start:end], self.sr)
            output.add_(modulated.sum(dim=1))
        return analytic_to_real(output.reshape(batch, channels, length))
