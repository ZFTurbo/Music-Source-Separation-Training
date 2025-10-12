import torch.nn as nn
import torch

from .modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from .bandsequence import BandSequenceModelModule

from .layers import (get_norm)
from .abstract import DTTNetBase

class DPTDFNet(DTTNetBase):
    def __init__(self, num_blocks, l, g, k, bn, bias, bn_norm, bandsequence, block_type,  **kwargs):

        super().__init__(**kwargs)
        # self.save_hyperparameters()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        scale = (2, 2)

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, audio_channels, num_samples) - Raw audio waveform
        Returns:
            (batch, audio_channels, num_samples) - Separated audio waveform
        '''
        # --- START OF FIX ---
        # 1. Convert raw audio input to a spectrogram
        spec = self.stft(x)
        # --- END OF FIX ---

        # The original forward pass now operates on the spectrogram
        x_spec = self.first_conv(spec)

        x_spec = x_spec.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x_spec = self.encoding_blocks[i](x_spec)
            ds_outputs.append(x_spec)
            x_spec = self.ds[i](x_spec)

        x_spec = self.bottleneck_block1(x_spec)
        x_spec = self.bottleneck_block2(x_spec)

        for i in range(self.n):
            x_spec = self.us[i](x_spec)
            
            # Crop the skip connection tensor to match the size of the upsampled tensor.
            skip_connection = ds_outputs[-i - 1]
            if skip_connection.shape[3] > x_spec.shape[3]:
                skip_connection = skip_connection[:, :, :, :x_spec.shape[3]]
            if skip_connection.shape[2] > x_spec.shape[2]:
                skip_connection = skip_connection[:, :, :x_spec.shape[2], :]
            
            x_spec = x_spec * skip_connection
            x_spec = self.decoding_blocks[i](x_spec)

        x_spec = x_spec.transpose(-1, -2)

        spec_out = self.final_conv(x_spec)

        # --- START OF FIX ---
        # 2. Convert the output spectrogram back to raw audio before returning
        waveform_out = self.istft(spec_out)
        # --- END OF FIX ---

        return waveform_out