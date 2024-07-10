import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .separation import SeparationNet
import typing as tp
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.

    Args:
        channels (int): input/output channels.
        depth (int): number of layers in the residual branch. Each layer has its own
        compress (float): amount of channel compression.
        kernel (int): kernel size for the convolutions.
        """

    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = (kernel // 2)
            mods = [
                norm(channels),
                nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                nn.GLU(1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    A FusionLayer within the decoder.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer, defaults to 3.
    - stride (int, optional): Stride for the convolutional layer, defaults to 1.
    - padding (int, optional): Padding for the convolutional layer, defaults to 1.
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayer, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class SDlayer(nn.Module):
    """
    Implements a Sparse Down-sample Layer for processing different frequency bands separately.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (dict): A dictionary containing configuration for each frequency band.
                           Keys are 'low', 'mid', 'high' for each band, and values are
                           dictionaries with keys 'SR', 'stride', and 'kernel' for proportion,
                           stride, and kernel size, respectively.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        # Saving rate proportions for determining splits
        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        # Define splitting points based on sampling rates
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        # Processing each band with the corresponding convolution
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            # padding
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Implements a Sparse Up-sample Layer in decoder.

    Args:
    - channels_in: The number of input channels.
    - channels_out: The number of output channels.
    - convtr_configs: Dictionary containing the configurations for transposed convolutions.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        # Define splitting points based on input lengths
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        # Processing each band with the corresponding convolution
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # Calculate the distance to trim the output symmetrically to original length
            current_Fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2

            # Trim the output to the original length symmetrically
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]

            outputs.append(trimmed_out)

        # Concatenate trimmed outputs along the frequency dimension to return the final tensor
        x = torch.cat(outputs, dim=2)

        return x


class SDblock(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): List specifying the convolution depths for low, mid, and high frequency bands.
    """

    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        # Dynamically create convolution modules for each band based on depths
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Set the kernel_size to an odd number.
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        # B, C, f, T = band.shape
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)

        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        skip = full_band

        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


class SCNet(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 32, 64, 128],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=6,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': normalized
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape
        B = x.shape[0]
        # In the initial padding, ensure that the number of frames after the STFT (the length of the T dimension) is even,
        # so that the RFFT operation can be used in the separation network.
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = F.pad(x, (0, padding))

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # separation
        x = self.separation_net(x)

        # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)

        x = x[:, :, :, :-padding]

        return x
