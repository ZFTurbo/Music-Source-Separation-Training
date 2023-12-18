import warnings
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn.modules import activation

from models.bandit.core.model.bsrnn.utils import (
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
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True, ):

        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = torch.jit.script(nn.Sequential(
                nn.Linear(in_features=emb_dim, out_features=mlp_dim),
                activation.__dict__[hidden_activation](
                        **self.hidden_activation_kwargs
                ),
        ))

        self.bandwidth = bandwidth
        self.in_channel = in_channel

        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2


class NormMLP(BaseNormMLP):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                bandwidth=bandwidth,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        )

        self.output = torch.jit.script(
                nn.Sequential(
                        nn.Linear(
                                in_features=mlp_dim,
                                out_features=bandwidth * in_channel * self.reim * 2,
                        ),
                        nn.GLU(dim=-1),
                )
        )

    def reshape_output(self, mb):
        # print(mb.shape)
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(
                    batch,
                    n_time,
                    self.in_channel,
                    self.bandwidth,
                    self.reim
            ).contiguous()
            # print(mb.shape)
            mb = torch.view_as_complex(
                    mb
            )  # (batch, n_time, in_channel, bandwidth)
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)

        mb = torch.permute(
                mb,
                (0, 2, 3, 1)
        )  # (batch, in_channel, bandwidth, n_time)

        return mb

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb0")


        qb = self.norm(qb)  # (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb1")

        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb2")
        mb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("mb")
        mb = self.reshape_output(mb)  # (batch, in_channel, bandwidth, n_time)

        return mb


class MultAddNormMLP(NormMLP):
    def __init__(self, emb_dim: int, mlp_dim: int, bandwidth: int, in_channel: "int | None", hidden_activation: str = "Tanh", hidden_activation_kwargs=None, complex_mask: bool = True) -> None:
        super().__init__(emb_dim, mlp_dim, bandwidth, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask)

        self.output2 = torch.jit.script(
                nn.Sequential(
                        nn.Linear(
                                in_features=mlp_dim,
                                out_features=bandwidth * in_channel * self.reim * 2,
                        ),
                        nn.GLU(dim=-1),
                )
        )

    def forward(self, qb):

        qb = self.norm(qb)  # (batch, n_time, emb_dim)
        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        mmb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        mmb = self.reshape_output(mmb)  # (batch, in_channel, bandwidth, n_time)
        amb = self.output2(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        amb = self.reshape_output(amb)  # (batch, in_channel, bandwidth, n_time)

        return mmb, amb


class MaskEstimationModuleSuperBase(nn.Module):
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
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
                        (
                                norm_mlp_cls(
                                        bandwidth=self.band_widths[b],
                                        emb_dim=emb_dim,
                                        mlp_dim=mlp_dim,
                                        in_channel=in_channel,
                                        hidden_activation=hidden_activation,
                                        hidden_activation_kwargs=hidden_activation_kwargs,
                                        complex_mask=complex_mask,
                                        **norm_mlp_kwargs,
                                )
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



class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
            self,
            in_channel: int,
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
            use_freq_weights: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)

        # if cond_dim > 0:
        #     raise NotImplementedError

        super().__init__(
                band_specs=band_specs,
                emb_dim=emb_dim + cond_dim,
                mlp_dim=mlp_dim,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                norm_mlp_cls=norm_mlp_cls,
                norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channel = in_channel

        if freq_weights is not None:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw)

                self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

        self.cond_dim = cond_dim

    def forward(self, q, cond=None):
        # q = (batch, n_bands, n_time, emb_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            print(cond)
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")

            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            cond = torch.ones(
                    (batch, n_bands, n_time, self.cond_dim),
                    device=q.device,
                    dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)
        else:
            pass

        mask_list = self.compute_masks(
                q
        )  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        masks = torch.zeros(
                (batch, self.in_channel, self.n_freq, n_time),
                device=q.device,
                dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
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
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
            **kwargs,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(
                in_channel=in_channel,
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
        )  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        # TODO: currently this requires band specs to have no gap and no overlap
        masks = torch.concat(
                masks,
                dim=2
        )  # (batch, in_channel, n_freq, n_time)

        return masks
