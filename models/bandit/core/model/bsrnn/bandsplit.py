from typing import List, Tuple

import torch
from torch import nn

from models.bandit.core.model.bsrnn.utils import (
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
            in_channel: int,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        self.treat_channel_as_feature = treat_channel_as_feature

        if normalize_channel_independently:
            raise NotImplementedError

        reim = 2

        self.norm = nn.LayerNorm(in_channel * bandwidth * reim)

        fc_in = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0
            emb_dim = emb_dim // in_channel

        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, xb):
        # xb = (batch, n_time, in_chan, reim * band_width)

        batch, n_time, in_chan, ribw = xb.shape
        xb = self.norm(xb.reshape(batch, n_time, in_chan * ribw))
        # (batch, n_time, in_chan * reim * band_width)

        if not self.treat_channel_as_feature:
            xb = xb.reshape(batch, n_time, in_chan, ribw)
            # (batch, n_time, in_chan, reim * band_width)

        zb = self.fc(xb)
        # (batch, n_time, emb_dim)
        # OR
        # (batch, n_time, in_chan, emb_dim_per_chan)

        if not self.treat_channel_as_feature:
            batch, n_time, in_chan, emb_dim_per_chan = zb.shape
            # (batch, n_time, in_chan, emb_dim_per_chan)
            zb = zb.reshape((batch, n_time, in_chan * emb_dim_per_chan))

        return zb  # (batch, n_time, emb_dim)


class BandSplitModule(nn.Module):
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            in_channel: int,
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

        self.norm_fc_modules = nn.ModuleList(
                [  # type: ignore
                        (
                                NormFC(
                                        emb_dim=emb_dim,
                                        bandwidth=bw,
                                        in_channel=in_channel,
                                        normalize_channel_independently=normalize_channel_independently,
                                        treat_channel_as_feature=treat_channel_as_feature,
                                )
                        )
                        for bw in self.band_widths
                ]
        )

    def forward(self, x: torch.Tensor):
        # x = complex spectrogram (batch, in_chan, n_freq, n_time)

        batch, in_chan, _, n_time = x.shape

        z = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim),
            device=x.device
        )

        xr = torch.view_as_real(x)  # batch, in_chan, n_freq, n_time, 2
        xr = torch.permute(
            xr,
            (0, 3, 1, 4, 2)
            )  # batch, n_time, in_chan, 2, n_freq
        batch, n_time, in_chan, reim, band_width = xr.shape
        for i, nfm in enumerate(self.norm_fc_modules):
            # print(f"bandsplit/band{i:02d}")
            fstart, fend = self.band_specs[i]
            xb = xr[..., fstart:fend]
            # (batch, n_time, in_chan, reim, band_width)
            xb = torch.reshape(xb, (batch, n_time, in_chan, -1))
            # (batch, n_time, in_chan, reim * band_width)
            # z.append(nfm(xb))  # (batch, n_time, emb_dim)
            z[:, i, :, :] = nfm(xb.contiguous())

        # z = torch.stack(z, dim=1)

        return z
