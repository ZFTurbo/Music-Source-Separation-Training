import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.model_utils import prefer_target_instrument


class ShortTimeHartleyTransform:
    def __init__(self, *, n_fft: int, hop_length: int, center: bool = True,
                 pad_mode: str = "reflect") -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.window = torch.hamming_window(self.n_fft)

    @staticmethod
    def _hartley_transform(x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft(x)
        return fft.real - fft.imag

    @staticmethod
    def _inverse_hartley_transform(X: torch.Tensor) -> torch.Tensor:
        N = X.size(-1)
        return ShortTimeHartleyTransform._hartley_transform(X) / N

    def transform(self, *, signal: torch.Tensor) -> torch.Tensor:
        assert signal.dim() == 3, "Signal must be a 3D tensor (batch_size, channel, samples)"
        self.window = self.window.to(signal.device)
        batch_size, channels, samples = signal.shape

        # Apply padding if center=True
        if self.center:
            pad_length = self.n_fft // 2
            signal = F.pad(signal, (pad_length, pad_length), mode=self.pad_mode)
        else:
            pad_length = 0

        # print(
        # f"samples={samples}\n"
        # f"self.hop_length={self.hop_length}\n"
        # f"pad_length={pad_length}\n"
        # f"signal_padded={signal.size(2)}"
        # )

        # Compute number of frames
        num_frames = (signal.size(2) - self.n_fft) // self.hop_length + 1

        # Apply window and compute Hartley transform
        window = self.window.to(signal.device, signal.dtype).unsqueeze(0).unsqueeze(0)
        stht_coeffs = []

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = signal[:, :, start:end] * window
            stht_coeffs.append(self._hartley_transform(frame))

        return torch.stack(stht_coeffs, dim=-1)

    def inverse_transform(self, *, stht_coeffs: torch.Tensor, length: int) -> torch.Tensor:
        self.window = self.window.to(stht_coeffs.device)
        # print(stht_coeffs.shape)
        batch_size, channels, n_fft, num_frames = stht_coeffs.shape
        signal_length = length

        # Initialize reconstruction
        reconstructed_signal = torch.zeros((batch_size, channels, signal_length + (self.n_fft if self.center else 0)),
                                           device=stht_coeffs.device, dtype=stht_coeffs.dtype)
        normalization = torch.zeros(signal_length + (self.n_fft if self.center else 0),
                                    device=stht_coeffs.device, dtype=stht_coeffs.dtype)

        window = self.window.to(stht_coeffs.device, stht_coeffs.dtype).unsqueeze(0).unsqueeze(0)

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            # Reconstruct frame and add to signal
            frame = self._inverse_hartley_transform(stht_coeffs[:, :, :, i]) * window
            reconstructed_signal[:, :, start:end] += frame
            normalization[start:end] += (window ** 2).squeeze()

        # Normalize the overlapping regions
        eps = torch.finfo(normalization.dtype).eps
        normalization = torch.clamp(normalization, min=eps)
        reconstructed_signal /= normalization.unsqueeze(0).unsqueeze(0)

        # Remove padding if center=True
        if self.center:
            pad_length = self.n_fft // 2
            reconstructed_signal = reconstructed_signal[:, :, pad_length:-pad_length]

        # Trim to the specified length
        return reconstructed_signal[:, :, :signal_length]


def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        norm = get_norm(norm_type=config.model.norm)
        act = get_act(act_type=config.model.act)

        self.num_target_instruments = len(prefer_target_instrument(config))
        self.num_subbands = config.model.num_subbands

        # dim_c = self.num_subbands * config.audio.num_channels * 2
        dim_c = self.num_subbands * config.audio.num_channels
        n = config.model.num_scales
        scale = config.model.scale
        l = config.model.num_blocks_per_scale
        c = config.model.num_channels
        g = config.model.growth
        bn = config.model.bottleneck_factor
        f = config.audio.dim_f // (self.num_subbands // 2)

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = ShortTimeHartleyTransform(n_fft=config.audio.n_fft, hop_length=config.audio.hop_length)

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, x):
        length = x.shape[-1]
        # print(x.shape)
        x = self.stft.transform(signal=x)
        # print(x.shape)

        mix = x = self.cac2cws(x)

        # print(x.shape)

        first_conv_out = x = self.first_conv(x)

        # print(x.shape)

        x = x.transpose(-1, -2)

        # print(x.shape)

        encoder_outputs = []
        for block in self.encoder_blocks:
            # print(x.shape)
            x = block.tfc_tdf(x)
            # print(x.shape)
            encoder_outputs.append(x)
            x = block.downscale(x)
            # print(x.shape)

        x = self.bottleneck_block(x)
        # print(x.shape)

        for block in self.decoder_blocks:
            # print(x.shape)
            x = block.upscale(x)
            # print(x.shape)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            # print(x.shape)
            x = block.tfc_tdf(x)
            # print(x.shape)

        x = x.transpose(-1, -2)
        # print(x.shape)

        x = x * first_conv_out  # reduce artifacts

        # print(x.shape)

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b * self.num_target_instruments, -1, f, t)
            x = self.stft.inverse_transform(stht_coeffs=x, length=length)
            x = x.reshape(b, self.num_target_instruments, x.shape[-2], x.shape[-1])
        else:
            x = self.stft.inverse_transform(stht_coeffs=x, length=length)
        # print("!!!", x.shape)
        return x
