from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .attention import Block
from .bandsplit import BandSplit
from .fourier import Fourier
from .rope import RoPE


class BSRoformer(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=44100,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        num_stems=1,
        **kwargs
    ) -> None:

        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.patch_size = patch_size
        self.audio_channels = audio_channels
        self.band_dim = band_dim
        self.num_stems = num_stems

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=audio_channels * 2,  # real + imag
            out_channels=band_dim
        )

        self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch = nn.ConvTranspose2d(dim, band_dim * num_stems, kernel_size=patch_size, stride=patch_size)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)
        T0 = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        # 1.4 Patchify
        x = self.patch(x)  # shape: (b, d, t, f)
        B = x.shape[0]
        T1 = x.shape[2]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # --- 3. Decode ---
        # 3.1 Unpatchify (produces band_dim * num_stems channels)
        x = self.unpatch(x)  # shape: (b, band_dim * num_stems, t, f)

        B = x.shape[0]
        T1 = x.shape[2]

        # Split into num_stems groups and flatten batch for inverse_transform
        x = x.view(B, self.num_stems, self.band_dim, T1, x.shape[-1])  # (b, s, d, t, f)
        x = rearrange(x, 'b s d t f -> (b s) d t f')

        # 3.2 Convert mel scale STFT to original STFT (per stem)
        x = self.bandsplit.inverse_transform(x)  # (b * s, in_channels, t, f)

        # Unpad
        x = x[:, :, 0 : T0, :]

        # 3.3 Get complex mask per stem
        x = rearrange(
            x,
            '(b s) (c k) t f -> b s c t f k',
            b=B, s=self.num_stems, c=self.audio_channels, k=2,
        ).contiguous()
        masks = torch.view_as_complex(x)  # shape: (b, num_stems, c, t, f)

        # 3.5 Calculate stft of separated audio per stem
        outputs = []
        for s in range(self.num_stems):
            sep_stft = masks[:, s] * complex_sp  # (b, c, t, f)
            out = self.istft(sep_stft)  # (b, c, l)
            outputs.append(out)

        output = torch.stack(outputs, dim=1)  # (b, num_stems, c, l)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x