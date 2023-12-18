from typing import Dict, Optional

import torch
import torchaudio as ta
from torch import nn


class _SpectralComponent(nn.Module):
    def __init__(
            self,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()

        assert power is None

        window_fn = torch.__dict__[window_fn]

        self.stft = (
                ta.transforms.Spectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        pad=0,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        power=power,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )

        self.istft = (
                ta.transforms.InverseSpectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        pad=0,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )
