'''
SCNet - great paper, great implementation
https://arxiv.org/pdf/2401.13276.pdf
https://github.com/amanteur/SCNet-PyTorch
'''

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.scnet_unofficial.modules import DualPathRNN, SDBlock, SUBlock
from models.scnet_unofficial.utils import compute_sd_layer_shapes, compute_gcr

from einops import rearrange, pack, unpack
from functools import partial

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class BandSplit(nn.Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


class SCNet(nn.Module):
    """
    SCNet class implements a source separation network,
    which explicitly split the spectrogram of the mixture into several subbands
    and introduce a sparsity-based encoder to model different frequency bands.

    Paper: "SCNET: SPARSE COMPRESSION NETWORK FOR MUSIC SOURCE SEPARATION"
    Authors: Weinan Tong, Jiaxu Zhu et al.
    Link: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - n_fft (int): Number of FFTs to determine the frequency dimension of the input.
    - dims (List[int]): List of channel dimensions for each block.
    - bandsplit_ratios (List[float]): List of ratios for splitting the frequency bands.
    - downsample_strides (List[int]): List of stride values for downsampling in each block.
    - n_conv_modules (List[int]): List specifying the number of convolutional modules in each block.
    - n_rnn_layers (int): Number of recurrent layers in the dual path RNN.
    - rnn_hidden_dim (int): Dimensionality of the hidden state in the dual path RNN.
    - n_sources (int, optional): Number of sources to be separated. Default is 4.

    Shapes:
    - Input: (B, C, T) where
        B is batch size,
        C is channel dim (mono / stereo),
        T is time dim
    - Output: (B, N, C, T) where
        B is batch size,
        N is the number of sources.
        C is channel dim (mono / stereo),
        T is sequence length,
    """
    @beartype
    def __init__(
        self,
        n_fft: int,
        dims: List[int],
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        n_rnn_layers: int,
        rnn_hidden_dim: int,
        n_sources: int = 4,
        hop_length: int = 1024,
        win_length: int = 4096,
        stft_window_fn: Optional[Callable] = None,
        stft_normalized: bool = False,
        **kwargs
    ):
        """
        Initializes SCNet with input parameters.
        """
        super().__init__()
        self.assert_input_data(
            bandsplit_ratios,
            downsample_strides,
            n_conv_modules,
        )

        n_blocks = len(dims) - 1
        n_freq_bins = n_fft // 2 + 1
        subband_shapes, sd_intervals = compute_sd_layer_shapes(
            input_shape=n_freq_bins,
            bandsplit_ratios=bandsplit_ratios,
            downsample_strides=downsample_strides,
            n_layers=n_blocks,
        )
        self.sd_blocks = nn.ModuleList(
            SDBlock(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                bandsplit_ratios=bandsplit_ratios,
                downsample_strides=downsample_strides,
                n_conv_modules=n_conv_modules,
            )
            for i in range(n_blocks)
        )
        self.dualpath_blocks = DualPathRNN(
            n_layers=n_rnn_layers,
            input_dim=dims[-1],
            hidden_dim=rnn_hidden_dim,
            **kwargs
        )
        self.su_blocks = nn.ModuleList(
            SUBlock(
                input_dim=dims[i + 1],
                output_dim=dims[i] if i != 0 else dims[i] * n_sources,
                subband_shapes=subband_shapes[i],
                sd_intervals=sd_intervals[i],
                upsample_strides=downsample_strides,
            )
            for i in reversed(range(n_blocks))
        )
        self.gcr = compute_gcr(subband_shapes)

        self.stft_kwargs = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), win_length)
        self.n_sources = n_sources
        self.hop_length = hop_length

    @staticmethod
    def assert_input_data(*args):
        """
        Asserts that the shapes of input features are equal.
        """
        for arg1 in args:
            for arg2 in args:
                if len(arg1) != len(arg2):
                    raise ValueError(
                        f"Shapes of input features {arg1} and {arg2} are not equal."
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SCNet.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
        - torch.Tensor: Output tensor of shape (B, N, C, T).
        """

        device = x.device
        stft_window = self.stft_window_fn(device=device)

        if x.ndim == 2:
            x = rearrange(x, 'b t -> b 1 t')

        c = x.shape[1]
        
        stft_pad = self.hop_length - x.shape[-1] % self.hop_length
        x = F.pad(x, (0, stft_pad))

        # stft
        x, ps = pack_one(x, '* t')
        x = torch.stft(x, **self.stft_kwargs, window=stft_window, return_complex=True)
        x = torch.view_as_real(x)
        x = unpack_one(x, ps, '* c f t')
        x = rearrange(x, 'b c f t r -> b f t (c r)')

        # encoder part
        x_skips = []
        for sd_block in self.sd_blocks:
            x, x_skip = sd_block(x)
            x_skips.append(x_skip)

        # separation part
        x = self.dualpath_blocks(x)

        # decoder part
        for su_block, x_skip in zip(self.su_blocks, reversed(x_skips)):
            x = su_block(x, x_skip)

        # istft
        x = rearrange(x, 'b f t (c r n) -> b n c f t r', c=c, n=self.n_sources, r=2)
        x = x.contiguous()

        x = torch.view_as_complex(x) 
        x = rearrange(x, 'b n c f t -> (b n c) f t')
        x = torch.istft(x, **self.stft_kwargs, window=stft_window, return_complex=False)
        x = rearrange(x, '(b n c) t -> b n c t', c=c, n=self.n_sources)

        x = x[..., :-stft_pad] 

        return x
