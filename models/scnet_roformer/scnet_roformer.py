import contextlib
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint


def exists(val):
    return val is not None


def autocast_disabled(device):
    if device.type in {"cuda", "cpu", "xpu", "hpu", "mps"}:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


# rotary embedding

class RotaryEmbedding(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self.interpolate_factor = 1.
        self.cache = {}


def _pair_swap(t: torch.Tensor) -> torch.Tensor:
    return t.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)


def exact_rotary_angles(rotary, seq_len: int, device) -> torch.Tensor:
    device = torch.device(device)
    key = f"exact_angles:{seq_len}:{device}"
    cache = getattr(rotary, "cache", None)
    angles = cache.get(key) if isinstance(cache, dict) else None
    if angles is None:
        with torch.autocast(device_type=device.type, enabled=False):
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            positions = positions / rotary.interpolate_factor
            frequencies = rotary.freqs.detach().to(device=device, dtype=torch.float32)
            angles = (positions[:, None] * frequencies[None, :]).repeat_interleave(2, dim=-1)
        if isinstance(cache, dict):
            cache[key] = angles
    return angles


def _exact_rotary_tables(rotary, seq_len: int, device, dtype, seq_dim: int):
    device = torch.device(device)
    cache_key = f"exact_cs:{seq_len}:{dtype}:{device}:{seq_dim}"
    cache = getattr(rotary, "cache", None)
    tables = cache.get(cache_key) if isinstance(cache, dict) else None
    if tables is None:
        angles = exact_rotary_angles(rotary, seq_len, device)
        with torch.autocast(device_type=device.type, enabled=False):
            sign = torch.ones(angles.shape[-1], device=device, dtype=torch.float32)
            sign[0::2] = -1
            c = angles.cos().to(dtype)
            s = (angles.sin() * sign).to(dtype)
        if seq_dim == -3:
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
        tables = (c, s)
        if isinstance(cache, dict):
            cache[cache_key] = tables
    return tables


def rotate_queries_and_keys_exact(rotary, q: torch.Tensor, k: torch.Tensor, seq_dim: int = -2):
    c, s = _exact_rotary_tables(rotary, q.shape[seq_dim], q.device, q.dtype, seq_dim)
    return q * c + _pair_swap(q) * s, k * c + _pair_swap(k) * s


# transformer

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = (x.float() * x.float()).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.gamma


class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        if self.rotary_embed is not None:
            q, k = rotate_queries_and_keys_exact(self.rotary_embed, q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0., scale=self.scale
        )

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                rotary_embed=rotary_embed,
            )

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# separation network

class FeatureConversion(nn.Module):
    """
    Integrates into the adjacent Dual-Path layer.

    Args:
        channels (int): Number of input channels.
        inverse (bool): If True, uses ifft; otherwise, uses rfft.
    """

    def __init__(self, channels, inverse, preserve_amp_dtype=True):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        self.preserve_amp_dtype = preserve_amp_dtype

    def forward(self, x):
        # B, C, F, T = x.shape
        dtype_in = x.dtype
        if self.inverse:
            x = x.float()
            x_r = x[:, :self.channels // 2, :, :]
            x_i = x[:, self.channels // 2:, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)
        if self.preserve_amp_dtype and dtype_in in (torch.float16, torch.bfloat16):
            x = x.to(dtype_in)
        return x


class DualPathTransformer(nn.Module):
    """
    Dual-Path Transformer in Separation Network.

    Args:
        d_model (int): The number of expected features in the input.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        ff_mult (int): Expansion factor of the feed-forward layer.
        freq_rotary_embed (RotaryEmbedding): Rotary embedding of the frequency path.
        time_rotary_embed (RotaryEmbedding): Rotary embedding of the time path.
    """

    def __init__(self, d_model, heads, dim_head, ff_mult, freq_rotary_embed, time_rotary_embed):
        super().__init__()

        self.d_model = d_model
        transformer_kwargs = dict(dim=d_model, depth=1, dim_head=dim_head, heads=heads, ff_mult=ff_mult)
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])
        self.freq_layer = Transformer(rotary_embed=freq_rotary_embed, **transformer_kwargs)
        self.time_layer = Transformer(rotary_embed=time_rotary_embed, **transformer_kwargs)

    def forward(self, x):
        B, C, F, T = x.shape

        # Frequency-path
        original_x = x
        x = self.norm_layers[0](x)
        x = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        x = self.freq_layer(x)
        x = x.view(B, T, F, C).permute(0, 3, 2, 1)
        x = x + original_x

        # Time-path
        original_x = x
        x = self.norm_layers[1](x)
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)
        x = self.time_layer(x)
        x = x.view(B, F, T, C).permute(0, 3, 1, 2)
        x = x + original_x

        return x


class SeparationNet(nn.Module):
    """
    Separation network of alternating dual-path layers and feature conversions.

    Args:
    - channels (int): Number input channels.
    - num_layers (int): Number of dual-path layers.
    - heads (int): Number of attention heads.
    - dim_head (int): Dimension of each attention head.
    - ff_mult (int): Expansion factor of the feed-forward layers.
    - use_torch_checkpoint (bool): Enable activation checkpointing to save VRAM.
    """

    def __init__(
        self,
        channels,
        num_layers=6,
        heads=8,
        dim_head=64,
        ff_mult=4,
        use_torch_checkpoint=False,
        preserve_amp_dtype=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_torch_checkpoint = use_torch_checkpoint

        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        time_rotary_embed = RotaryEmbedding(dim=dim_head)

        self.dp_modules = nn.ModuleList([
            DualPathTransformer(
                channels * (2 if i % 2 == 1 else 1),
                heads,
                dim_head,
                ff_mult,
                freq_rotary_embed,
                time_rotary_embed,
            )
            for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            FeatureConversion(
                channels * 2,
                inverse=False if i % 2 == 0 else True,
                preserve_amp_dtype=preserve_amp_dtype,
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            if self.use_torch_checkpoint and self.training:
                x = checkpoint(self.dp_modules[i], x, use_reentrant=False)
                x = checkpoint(self.feature_conversion[i], x, use_reentrant=False)
            else:
                x = self.dp_modules[i](x)
                x = self.feature_conversion[i](x)
        return x


# encoder / decoder

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block applied along time.

    Args:
        channels (int): input/output channels.
        expansion (int): channel expansion of the pointwise layers.
        kernel (int): kernel size of the depthwise convolution.
    """

    def __init__(self, channels, expansion=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.dwconv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, channels * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(channels * expansion, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.

    Args:
        channels (int): input/output channels.
        depth (int): number of ConvNeXt blocks.
        expansion (int): channel expansion of the pointwise layers.
        kernel (int): kernel size for the depthwise convolutions.
    """

    def __init__(self, channels, depth=3, expansion=4, kernel=3):
        super().__init__()
        self.layers = nn.ModuleList([ConvNeXtBlock(channels, expansion, kernel) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + skip
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class SDlayer(nn.Module):
    """
    Implements a Sparse Down-sample Layer for processing different frequency bands separately.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (list of dict): Configuration for each frequency band, with keys 'SR', 'stride'
                                   and 'kernel' for proportion, stride and kernel size.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()

        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs:
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        rates = [config['SR'] for config in band_configs]
        self.split_rates = [sum(rates[:i + 1]) for i in range(len(rates) - 1)]

    def forward(self, x):
        B, C, Fr, T = x.shape
        edges = [0] + [math.ceil(Fr * rate) for rate in self.split_rates] + [Fr]
        splits = list(zip(edges[:-1], edges[1:]))

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
    Implements a Sparse Up-sample Layer in decoder with sub-pixel convolutions.

    Args:
    - channels_in: The number of input channels.
    - channels_out: The number of output channels.
    - band_configs: Configuration for each frequency band, with keys 'stride' and 'kernel'.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()

        self.channels_out = channels_out
        self.strides = [config['stride'] for config in band_configs]
        self.kernels = [config['kernel'] for config in band_configs]
        self.up_convs = nn.ModuleList([
            nn.Conv2d(channels_in, channels_out * config['stride'], (config['kernel'], 1))
            for config in band_configs
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        edges = [0]
        for length in lengths:
            edges.append(edges[-1] + length)

        outputs = []
        for idx, (conv, stride, kernel) in enumerate(zip(self.up_convs, self.strides, self.kernels)):
            band = x[:, :, edges[idx]:edges[idx + 1], :]
            pad_left = (kernel - 1) // 2
            band = F.pad(band, (0, 0, pad_left, kernel - 1 - pad_left))
            out = conv(band)
            bins = out.shape[2]
            out = out.view(B, self.channels_out, stride, bins, T).transpose(2, 3).reshape(
                B, self.channels_out, bins * stride, T)

            # Trim the output symmetrically to the original length
            dist = abs(origin_lengths[idx] - out.shape[2]) // 2
            outputs.append(out[:, :, dist:dist + origin_lengths[idx], :])

        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_configs (list of dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): Convolution depths for each frequency band.
    """

    def __init__(self, channels_in, channels_out, band_configs, conv_config, depths, kernel_size=3):
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

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


class SCNetRoformer(nn.Module):
    """
    SCNet with dual-path transformers, ConvNeXt encoder blocks and one decoder per source
    predicting complex masks.

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - dims (List[int]): Channel dimensions; dims[0] is the width after the input projection.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band.
    - conv_depths (List[int]): Number of ConvNeXt blocks for each frequency band.
    - conv_expansion (int): Channel expansion of the ConvNeXt blocks.
    - conv_kernel (int): Kernel size of the ConvNeXt depthwise convolutions.
    - num_dplayer (int): Number of dual-path layers.
    - heads (int): Number of attention heads.
    - dim_head (int): Dimension of each attention head.
    - ff_mult (int): Expansion factor of the feed-forward layers.
    - use_torch_checkpoint (bool): Enable activation checkpointing in the separation network.
    """

    def __init__(self,
                 sources=('vocals', 'bass', 'drums', 'guitar', 'piano', 'other'),
                 audio_channels=2,
                 # Main structure
                 dims=(16, 32, 64, 128),
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 # SD/SU layer
                 band_SR=(0.1975, 0.0508, 0.26545, 0.48625),
                 band_stride=(1, 2, 4, 8),
                 band_kernel=(3, 2, 4, 8),
                 # Convolution Module
                 conv_depths=(3, 3, 3, 3),
                 conv_expansion=4,
                 conv_kernel=3,
                 # Dual-path transformer
                 num_dplayer=6,
                 heads=8,
                 dim_head=64,
                 ff_mult=4,
                 use_torch_checkpoint=False,
                 preserve_amp_dtype=True,
                 multi_stft_resolution_loss_weight=1.,
                 multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
                 multi_stft_hop_size=147,
                 multi_stft_normalized=False,
                 multi_stft_n_fft=2048,
                 multi_stft_window_fn=torch.hann_window,
                 ):
        super().__init__()
        self.sources = list(sources)
        self.audio_channels = audio_channels
        self.dims = list(dims)
        self.nfft = nfft
        self.hop_length = hop_size
        self.band_configs = [
            {'SR': sr, 'stride': stride, 'kernel': kernel}
            for sr, stride, kernel in zip(band_SR, band_stride, band_kernel)
        ]
        self.conv_config = {
            'expansion': conv_expansion,
            'kernel': conv_kernel,
        }

        window = 0.5 * (1 - torch.cos(2 * math.pi * (torch.arange(nfft, dtype=torch.float64) + 1) / (nfft + 1)))
        self.register_buffer("window", window.float(), persistent=False)

        self.first_conv = nn.Conv2d(audio_channels * 2, dims[0], 1, bias=False)

        self.encoder = nn.ModuleList()
        for index in range(len(dims) - 1):
            self.encoder.append(SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            ))

        self.decoders = nn.ModuleDict()
        for source in self.sources:
            decoder = nn.ModuleList()
            for index in range(len(dims) - 1):
                decoder.insert(0, nn.Sequential(
                    FusionLayer(channels=dims[index + 1]),
                    SUlayer(
                        channels_in=dims[index + 1],
                        channels_out=dims[index] if index != 0 else audio_channels * 4,
                        band_configs=self.band_configs,
                    )
                ))
            self.decoders[source] = decoder

        self.separation_net = SeparationNet(
            channels=dims[-1],
            num_layers=num_dplayer,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            use_torch_checkpoint=use_torch_checkpoint,
            preserve_amp_dtype=preserve_amp_dtype,
        )

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = multi_stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def stft(self, x, frames):
        B, C, L = x.shape
        pad_left = self.nfft - self.hop_length
        pad_right = (frames - 1) * self.hop_length + self.nfft - pad_left - L
        x = F.pad(x, (pad_left, pad_right)).reshape(B * C, -1)
        x = torch.stft(x, self.nfft, self.hop_length, window=self.window, center=False, return_complex=True)
        return x[:, :-1].reshape(B, C, -1, frames)

    def istft(self, x, length):
        B, S, C, Fr, T = x.shape
        x = F.pad(x.reshape(-1, Fr, T), (0, 0, 0, 1))
        frames = torch.fft.irfft(x, n=self.nfft, dim=1) * self.window[:, None]
        total = (T - 1) * self.hop_length + self.nfft
        fold = dict(output_size=(1, total), kernel_size=(1, self.nfft), stride=(1, self.hop_length))
        y = F.fold(frames, **fold).reshape(B, S, C, total)
        envelope = (self.window ** 2)[None, :, None].expand(1, -1, T)
        envelope = F.fold(envelope, **fold).reshape(total)
        pad_left = self.nfft - self.hop_length
        return y[..., pad_left:pad_left + length] / envelope[pad_left:pad_left + length]

    def forward(self, x, target=None, active_stem_ids=None, return_loss_breakdown=False):
        # B, C, L = x.shape
        device = x.device
        B, _, L = x.shape
        frames = math.ceil(L / self.hop_length) + self.nfft // self.hop_length - 1
        # Keep the number of frames even so that the RFFT operation can be used in the separation network.
        frames += frames % 2

        with autocast_disabled(x.device):
            spec = self.stft(x, frames)
        mix = torch.view_as_real(spec)
        x = mix.permute(0, 1, 4, 2, 3).reshape(B, self.audio_channels * 2, spec.shape[2], frames)

        x = self.first_conv(x)

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

        if active_stem_ids is None:
            stem_ids = list(range(len(self.sources)))
        else:
            stem_ids = active_stem_ids

        # decoders
        masks = []
        for source in [self.sources[i] for i in stem_ids]:
            y = x
            for level, (fusion_layer, su_layer) in enumerate(self.decoders[source]):
                depth = len(save_skip) - 1 - level
                y = fusion_layer(y, save_skip[depth])
                y = su_layer(y, save_lengths[depth], save_original_lengths[depth])
            masks.append(F.glu(y, dim=1))

        # output
        with autocast_disabled(x.device):
            mask = torch.stack(masks, dim=1).to(spec.real.dtype)
            mask = mask.view(B, len(stem_ids), self.audio_channels, 2, *mask.shape[-2:])
            mask = torch.view_as_complex(mask.permute(0, 1, 2, 4, 5, 3).contiguous())
            recon_audio = self.istft(spec.unsqueeze(1) * mask, L)

        if not exists(target):
            return recon_audio

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]

        target_sel = target[:, stem_ids]
        loss = F.l1_loss(recon_audio, target_sel)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, 'b n s t -> (b n s) t'),**res_stft_kwargs)
            target_Y = torch.stft(rearrange(target_sel, 'b n s t -> (b n s) t'),**res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
