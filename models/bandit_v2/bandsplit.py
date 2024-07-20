from typing import List, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from .utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class NormFC(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        bandwidth: int,
        in_channels: int,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        if not treat_channel_as_feature:
            raise NotImplementedError

        self.treat_channel_as_feature = treat_channel_as_feature

        if normalize_channel_independently:
            raise NotImplementedError

        reim = 2

        norm = nn.LayerNorm(in_channels * bandwidth * reim)

        fc_in = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channels
        else:
            assert emb_dim % in_channels == 0
            emb_dim = emb_dim // in_channels

        fc = nn.Linear(fc_in, emb_dim)

        self.combined = nn.Sequential(norm, fc)

    def forward(self, xb):
        return checkpoint_sequential(self.combined, 1, xb, use_reentrant=False)


class BandSplitModule(nn.Module):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        in_channels: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        check_nonzero_bandwidth(band_specs)

        if require_no_gap:
            check_no_gap(band_specs)

        if require_no_overlap:
            check_no_overlap(band_specs)

        self.band_specs = band_specs
        # list of [fstart, fend) in index.
        # Note that fend is exclusive.
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.emb_dim = emb_dim

        try:
            self.norm_fc_modules = nn.ModuleList(
                [  # type: ignore
                    torch.compile(
                        NormFC(
                            emb_dim=emb_dim,
                            bandwidth=bw,
                            in_channels=in_channels,
                            normalize_channel_independently=normalize_channel_independently,
                            treat_channel_as_feature=treat_channel_as_feature,
                        ),
                        disable=True,
                    )
                    for bw in self.band_widths
                ]
            )
        except Exception as e:
            self.norm_fc_modules = nn.ModuleList(
                [  # type: ignore
                    NormFC(
                        emb_dim=emb_dim,
                        bandwidth=bw,
                        in_channels=in_channels,
                        normalize_channel_independently=normalize_channel_independently,
                        treat_channel_as_feature=treat_channel_as_feature,
                    )
                    for bw in self.band_widths
                ]
            )

    def forward(self, x: torch.Tensor):
        # x = complex spectrogram (batch, in_chan, n_freq, n_time)

        batch, in_chan, band_width, n_time = x.shape

        z = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim), device=x.device
        )

        x = torch.permute(x, (0, 3, 1, 2)).contiguous()

        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]
            xb = x[:, :, :, fstart:fend]
            xb = torch.view_as_real(xb)
            xb = torch.reshape(xb, (batch, n_time, -1))
            z[:, i, :, :] = nfm(xb)

        return z
