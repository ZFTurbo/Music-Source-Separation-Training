from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Fourier(nn.Module):
    
    def __init__(self, 
        n_fft=2048, 
        hop_length=480, 
        return_complex=True, 
        normalized=True
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

        self.register_buffer(name="window", tensor=torch.hann_window(self.n_fft))

    def stft(self, waveform: Tensor) -> Tensor:
        r"""Compute STFT of waveforms.
        
        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            waveform: (b, c, l)

        Returns:
            complex_sp: (b, c, t, f)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c l -> (b c) l')  # (b*c, l)

        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=self.return_complex
        )  # (b*c, f, t)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)  # (b, c, t, f)
        return complex_sp

    def istft(self, complex_sp: Tensor) -> Tensor:
        r"""Reconstruct waveforms from STFT.

        b: batch_size
        c: channels_num
        t: frames_num
        f: freq_bins
        l: audio_samples

        Args:
            complex_sp: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')  # (b*c, f, t)

        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )  # (b*c, l)

        out = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # (b, c, l)
        
        return out


if __name__ == '__main__':

    model = Fourier(n_fft=2048, hop_length = 480)
    x = torch.randn(8, 4, 96000)  # (b, c, l)
    stft = model.stft(x)  # (b, c, t, f)
    out = model.istft(stft)  # (b, c, l)

    print((x - out).abs().mean())