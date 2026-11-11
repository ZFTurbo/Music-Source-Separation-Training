from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BandSplit(nn.Module):
    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        n_bands: int, 
        in_channels: int, 
        out_channels: int
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band width
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name='ola_window', tensor=Tensor(ola_window))  # (f,)

        nonzero_indexes = []  # (k, w)
        nonzero_melbanks = []  # (k, w)
        
        for f in range(self.n_bands):    
            idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[0]  # shape: (w,)
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(self.melbanks[f, idxes])

        max_band_width = max([len(idxes) for idxes in nonzero_indexes])
        pad = -1

        nonzero_indexes = pad_sequence(
            sequences=nonzero_indexes, 
            batch_first=True, 
            padding_value=pad
        )  # (k, w)

        nonzero_melbanks = pad_sequence(
            sequences=nonzero_melbanks,
            batch_first=True,
            padding_value=0.
        )  # (k, w)

        mask = torch.zeros_like(nonzero_melbanks)  # (k, w)
        mask[torch.where(nonzero_melbanks != 0)] = 1
        
        new_pad = self.n_fft // 2
        nonzero_indexes[nonzero_indexes == pad] = new_pad
        
        self.register_buffer(name="nonzero_indexes", tensor=nonzero_indexes)
        self.register_buffer(name="nonzero_melbanks", tensor=nonzero_melbanks)
        self.register_buffer(name="mask", tensor=mask)

        self.pre_bandnet = BandLinear(
            n_bands=self.n_bands, 
            in_channels=max_band_width * self.in_channels, 
            out_channels=self.out_channels
        )

        self.post_bandnet = BandLinear(
            n_bands=self.n_bands, 
            in_channels=self.out_channels, 
            out_channels=max_band_width * self.in_channels
        )

    def init_melbanks(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize mel bins from librosa.

        f: stft bins
        k: mel bins

        Args:
            None

        Returns:
            melbanks: (k, f)
            ola_window: (f,)
        """

        melbanks = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.n_bands - 2, 
            norm=None
        )  # shape: (k, f)

        F = self.n_fft // 2 + 1
        
        # The zeroth bank, e.g., [1., 0.66, 0.32, 0, ..., 0.]
        melbank_0 = np.zeros(F)
        idx = np.argmax(melbanks[0])
        melbank_0[0 : idx] = 1. - melbanks[0, 0 : idx]  # (f,)

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]  # (f,)

        # Concatenate
        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # (n_mels, f)

        # Calculate overlap-add window
        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert ola_window.max() >= 0.5

        return melbanks, ola_window

    def transform(self, x: Tensor) -> Tensor:
        r"""Convert STFT to mel scale STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, f)

        Returns:
            x: (b, d, t, k)
        """

        # Band split
        x = x[:, :, :, self.nonzero_indexes.flatten()]  # (b, c, t, k*w)
        x *= self.nonzero_melbanks.flatten() * self.mask.flatten()  # (b, c, t, k*w)
        x = rearrange(x, 'b c t (k w) -> b t k (w c)', k=self.n_bands)  # (b, t, k, w*c)

        # Apply individual MLP on each band
        x = self.pre_bandnet(x)  # (b, t, k, d)        
        x = rearrange(x, 'b t k d -> b d t k')  # (b, d, t, k)

        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, d, t, k)

        Outputs:
            y: (b, c, t, f)
        """
        

        # Apply individual MLP on each band
        x = rearrange(x, 'b d t k -> b t k d')  # (b, t, k, d_in)
        x = self.post_bandnet(x)  # (b, t, k, w*c)
        x = rearrange(x, 'b t k (w c) -> b c t (k w)', c=self.in_channels)  # (b, c, t, k*w)

        # Band combine
        B, C, T = x.shape[0 : 3]
        Fq = self.n_fft // 2 + 1
        buffer = torch.zeros(B, C, T, Fq).to(x.device)  # shape: (b, c, t, f)

        # Mask out
        x *= self.mask.flatten()
        
        # Scatter add
        buffer.scatter_add_(
            dim=3, 
            index=self.nonzero_indexes.flatten()[None, None, None, :].repeat(B, C, T, 1), 
            src=x
        )

        # Divide overlap add window
        buffer /= self.ola_window

        return buffer


class BandLinear(nn.Module):
    def __init__(self, n_bands: int, in_channels: int, out_channels: int) -> None:
        r"""Band split linear layer."""

        super().__init__()

        self.weight = nn.Parameter(torch.empty(n_bands, in_channels, out_channels))  # (k, i, o)
        self.bias = nn.Parameter(torch.zeros(n_bands, out_channels))  # (k, o)

        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        r"""

        b: batch_size
        t: frames_num
        k: n_bands
        i: in_channels
        o: out_channels

        Args:
            x: (b, t, k, i)

        Returns:
            x: (b, t, k, o)
        """

        # Sum along the input channels axis
        return torch.einsum('btki,kio->btko', x, self.weight) + self.bias  # (b, t, k, o)


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(8, 4, 201, 1025)  # (b, c, t, f)

    bandsplit = BandSplit(
        sr=44100, 
        n_fft=2048, 
        n_bands=64,
        in_channels=4, 
        out_channels=64
    )

    y = bandsplit.transform(x)  # (b, d, t, k)
    x_hat = bandsplit.inverse_transform(y)  # (b, c, t, f)
    print(x - x_hat)