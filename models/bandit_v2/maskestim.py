from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn.modules import activation
from torch.utils.checkpoint import checkpoint_sequential

from .utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class BaseNormMLP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ):
        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=mlp_dim),
            activation.__dict__[hidden_activation](**self.hidden_activation_kwargs),
        )

        self.bandwidth = bandwidth
        self.in_channels = in_channels

        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2


class NormMLP(BaseNormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channels=in_channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

        self.output = nn.Sequential(
            nn.Linear(
                in_features=mlp_dim,
                out_features=bandwidth * in_channels * self.reim * 2,
            ),
            nn.GLU(dim=-1),
        )

        try:
            self.combined = torch.compile(
                nn.Sequential(self.norm, self.hidden, self.output), disable=True
            )
        except Exception as e:
            self.combined = nn.Sequential(self.norm, self.hidden, self.output)

    def reshape_output(self, mb):
        # print(mb.shape)
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(
                batch, n_time, self.in_channels, self.bandwidth, self.reim
            ).contiguous()
            # print(mb.shape)
            mb = torch.view_as_complex(mb)  # (batch, n_time, in_channels, bandwidth)
        else:
            mb = mb.reshape(batch, n_time, self.in_channels, self.bandwidth)

        mb = torch.permute(mb, (0, 2, 3, 1))  # (batch, in_channels, bandwidth, n_time)

        return mb

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)
        # qb = self.norm(qb)  # (batch, n_time, emb_dim)
        # qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        # mb = self.output(qb)  # (batch, n_time, bandwidth * in_channels * reim)

        mb = checkpoint_sequential(self.combined, 2, qb, use_reentrant=False)
        mb = self.reshape_output(mb)  # (batch, in_channels, bandwidth, n_time)

        return mb


class MaskEstimationModuleSuperBase(nn.Module):
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Dict = None,
    ) -> None:
        super().__init__()

        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if norm_mlp_kwargs is None:
            norm_mlp_kwargs = {}

        self.norm_mlp = nn.ModuleList(
            [
                norm_mlp_cls(
                    bandwidth=self.band_widths[b],
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channels=in_channels,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    **norm_mlp_kwargs,
                )
                for b in range(self.n_bands)
            ]
        )

    def compute_masks(self, q):
        batch, n_bands, n_time, emb_dim = q.shape

        masks = []

        for b, nmlp in enumerate(self.norm_mlp):
            # print(f"maskestim/{b:02d}")
            qb = q[:, b, :, :]
            mb = nmlp(qb)
            masks.append(mb)

        return masks

    def compute_mask(self, q, b):
        batch, n_bands, n_time, emb_dim = q.shape
        qb = q[:, b, :, :]
        mb = self.norm_mlp[b](qb)
        return mb


class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
        self,
        in_channels: int,
        band_specs: List[Tuple[float, float]],
        freq_weights: List[torch.Tensor],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Dict = None,
        use_freq_weights: bool = False,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)

        if cond_dim > 0:
            raise NotImplementedError

        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim + cond_dim,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channels = in_channels

        if freq_weights is not None and use_freq_weights:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw)

                self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

    def forward(self, q):
        # q = (batch, n_bands, n_time, emb_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        masks = torch.zeros(
            (batch, self.in_channels, self.n_freq, n_time),
            device=q.device,
            dtype=torch.complex64,
        )

        for im in range(n_bands):
            fstart, fend = self.band_specs[im]

            mask = self.compute_mask(q, im)

            if self.use_freq_weights:
                fw = self.get_buffer(f"freq_weights/{im}")[:, None]
                mask = mask * fw
            masks[:, :, fstart:fend, :] += mask

        return masks


class MaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        **kwargs,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(
            in_channels=in_channels,
            band_specs=band_specs,
            freq_weights=None,
            n_freq=None,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

    def forward(self, q, cond=None):
        # q = (batch, n_bands, n_time, emb_dim)

        masks = self.compute_masks(
            q
        )  # [n_bands  * (batch, in_channels, bandwidth, n_time)]

        # TODO: currently this requires band specs to have no gap and no overlap
        masks = torch.concat(masks, dim=2)  # (batch, in_channels, n_freq, n_time)

        return masks
