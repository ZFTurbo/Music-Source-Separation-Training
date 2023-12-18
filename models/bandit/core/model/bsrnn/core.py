from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.bandit.core.model.bsrnn import BandsplitCoreBase
from models.bandit.core.model.bsrnn.bandsplit import BandSplitModule
from models.bandit.core.model.bsrnn.maskestim import (
    MaskEstimationModule,
    OverlappingMaskEstimationModule
)
from models.bandit.core.model.bsrnn.tfmodel import (
    ConvolutionalTimeFreqModule,
    SeqBandModellingModule,
    TransformerTimeFreqModule
)


class MultiMaskBandSplitCoreBase(BandsplitCoreBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, cond=None, compute_residual: bool = True):
        # x = complex spectrogram (batch, in_chan, n_freq, n_time)
        # print(x.shape)
        batch, in_chan, n_freq, n_time = x.shape
        x = torch.reshape(x, (-1, 1, n_freq, n_time))

        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)

        # if torch.any(torch.isnan(z)):
        #     raise ValueError("z nan")

        # print(z)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)
        # print(q)


        # if torch.any(torch.isnan(q)):
        #     raise ValueError("q nan")

        out = {}

        for stem, mem in self.mask_estim.items():
            m = mem(q, cond=cond)

            # if torch.any(torch.isnan(m)):
            #     raise ValueError("m nan", stem)

            s = self.mask(x, m)
            s = torch.reshape(s, (batch, in_chan, n_freq, n_time))
            out[stem] = s

        return {"spectrogram": out}

    

    def instantiate_mask_estim(self, 
                               in_channel: int,
                               stems: List[str],
                               band_specs: List[Tuple[float, float]],
                               emb_dim: int,
                               mlp_dim: int,
                               cond_dim: int,
                               hidden_activation: str,
                                
                                hidden_activation_kwargs: Optional[Dict] = None,
                                complex_mask: bool = True,
                                overlapping_band: bool = False,
                                freq_weights: Optional[List[torch.Tensor]] = None,
                                n_freq: Optional[int] = None,
                                use_freq_weights: bool = True,
                                mult_add_mask: bool = False
                                ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if "mne:+" in stems:
            stems = [s for s in stems if s != "mne:+"]

        if overlapping_band:
            assert freq_weights is not None
            assert n_freq is not None

            if mult_add_mask:

                self.mask_estim = nn.ModuleDict(
                        {
                                stem: MultAddMaskEstimationModule(
                                        band_specs=band_specs,
                                        freq_weights=freq_weights,
                                        n_freq=n_freq,
                                        emb_dim=emb_dim,
                                        mlp_dim=mlp_dim,
                                        in_channel=in_channel,
                                        hidden_activation=hidden_activation,
                                        hidden_activation_kwargs=hidden_activation_kwargs,
                                        complex_mask=complex_mask,
                                        use_freq_weights=use_freq_weights,
                                )
                                for stem in stems
                        }
                )
            else:
                self.mask_estim = nn.ModuleDict(
                        {
                                stem: OverlappingMaskEstimationModule(
                                        band_specs=band_specs,
                                        freq_weights=freq_weights,
                                        n_freq=n_freq,
                                        emb_dim=emb_dim,
                                        mlp_dim=mlp_dim,
                                        in_channel=in_channel,
                                        hidden_activation=hidden_activation,
                                        hidden_activation_kwargs=hidden_activation_kwargs,
                                        complex_mask=complex_mask,
                                        use_freq_weights=use_freq_weights,
                                )
                                for stem in stems
                        }
                )
        else:
            self.mask_estim = nn.ModuleDict(
                    {
                            stem: MaskEstimationModule(
                                    band_specs=band_specs,
                                    emb_dim=emb_dim,
                                    mlp_dim=mlp_dim,
                                    cond_dim=cond_dim,
                                    in_channel=in_channel,
                                    hidden_activation=hidden_activation,
                                    hidden_activation_kwargs=hidden_activation_kwargs,
                                    complex_mask=complex_mask,
                            )
                            for stem in stems
                    }
            )

    def instantiate_bandsplit(self, 
                              in_channel: int,
                              band_specs: List[Tuple[float, float]],
                              require_no_overlap: bool = False,
                              require_no_gap: bool = True,
                              normalize_channel_independently: bool = False,
                              treat_channel_as_feature: bool = True,
                              emb_dim: int = 128
                              ):
        self.band_split = BandSplitModule(
                        in_channel=in_channel,
                        band_specs=band_specs,
                        require_no_overlap=require_no_overlap,
                        require_no_gap=require_no_gap,
                        normalize_channel_independently=normalize_channel_independently,
                        treat_channel_as_feature=treat_channel_as_feature,
                        emb_dim=emb_dim,
                )

class SingleMaskBandsplitCoreBase(BandsplitCoreBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        # x = complex spectrogram (batch, in_chan, n_freq, n_time)
        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)
        m = self.mask_estim(q)  # (batch, in_chan, n_freq, n_time)

        s = self.mask(x, m)

        return s


class SingleMaskBandsplitCoreRNN(
        SingleMaskBandsplitCoreBase,
):
    def __init__(
            self,
            in_channel: int,
            band_specs: List[Tuple[float, float]],
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__()
        self.band_split = (BandSplitModule(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim,
        ))
        self.tf_model = (SeqBandModellingModule(
                n_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
        ))
        self.mask_estim = (MaskEstimationModule(
                in_channel=in_channel,
                band_specs=band_specs,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        ))


class SingleMaskBandsplitCoreTransformer(
        SingleMaskBandsplitCoreBase,
):
    def __init__(
            self,
            in_channel: int,
            band_specs: List[Tuple[float, float]],
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            tf_dropout: float = 0.0,
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__()
        self.band_split = BandSplitModule(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim,
        )
        self.tf_model = TransformerTimeFreqModule(
                n_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                dropout=tf_dropout,
        )
        self.mask_estim = MaskEstimationModule(
                in_channel=in_channel,
                band_specs=band_specs,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        )


class MultiSourceMultiMaskBandSplitCoreRNN(MultiMaskBandSplitCoreBase):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: List[Tuple[float, float]],
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            cond_dim: int = 0,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            overlapping_band: bool = False,
            freq_weights: Optional[List[torch.Tensor]] = None,
            n_freq: Optional[int] = None,
            use_freq_weights: bool = True,
            mult_add_mask: bool = False
    ) -> None:

        super().__init__()
        self.instantiate_bandsplit(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim
        )

        
        self.tf_model = (
                SeqBandModellingModule(
                        n_modules=n_sqm_modules,
                        emb_dim=emb_dim,
                        rnn_dim=rnn_dim,
                        bidirectional=bidirectional,
                        rnn_type=rnn_type,
                )
        )

        self.mult_add_mask = mult_add_mask

        self.instantiate_mask_estim(
                in_channel=in_channel,
                stems=stems,
                band_specs=band_specs,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=overlapping_band,
                freq_weights=freq_weights,
                n_freq=n_freq,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )

    @staticmethod
    def _mult_add_mask(x, m):

        assert m.ndim == 5

        mm = m[..., 0]
        am = m[..., 1]

        # print(mm.shape, am.shape, x.shape, m.shape)

        return x * mm + am

    def mask(self, x, m):
        if self.mult_add_mask:

            return self._mult_add_mask(x, m)
        else:
            return super().mask(x, m)


class MultiSourceMultiMaskBandSplitCoreTransformer(
        MultiMaskBandSplitCoreBase,
):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: List[Tuple[float, float]],
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            tf_dropout: float = 0.0,
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            overlapping_band: bool = False,
            freq_weights: Optional[List[torch.Tensor]] = None,
            n_freq: Optional[int] = None,
            use_freq_weights:bool=True,
            rnn_type: str = "LSTM",
            cond_dim: int = 0,
            mult_add_mask: bool = False
    ) -> None:
        super().__init__()
        self.instantiate_bandsplit(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim
        )
        self.tf_model = TransformerTimeFreqModule(
                n_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                dropout=tf_dropout,
        )
        
        self.instantiate_mask_estim(
                in_channel=in_channel,
                stems=stems,
                band_specs=band_specs,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=overlapping_band,
                freq_weights=freq_weights,
                n_freq=n_freq,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )



class MultiSourceMultiMaskBandSplitCoreConv(
        MultiMaskBandSplitCoreBase,
):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: List[Tuple[float, float]],
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            tf_dropout: float = 0.0,
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            overlapping_band: bool = False,
            freq_weights: Optional[List[torch.Tensor]] = None,
            n_freq: Optional[int] = None,
            use_freq_weights:bool=True,
            rnn_type: str = "LSTM",
            cond_dim: int = 0,
            mult_add_mask: bool = False
    ) -> None:
        super().__init__()
        self.instantiate_bandsplit(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim
        )
        self.tf_model = ConvolutionalTimeFreqModule(
                n_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                dropout=tf_dropout,
        )
        
        self.instantiate_mask_estim(
                in_channel=in_channel,
                stems=stems,
                band_specs=band_specs,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=overlapping_band,
                freq_weights=freq_weights,
                n_freq=n_freq,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )


class PatchingMaskBandsplitCoreBase(MultiMaskBandSplitCoreBase):
    def __init__(self) -> None:
        super().__init__()

    def mask(self, x, m):
        # x.shape = (batch, n_channel, n_freq, n_time)
        # m.shape = (kernel_freq, kernel_time, batch, n_channel, n_freq, n_time)

        _, n_channel, kernel_freq, kernel_time, n_freq, n_time = m.shape
        padding = ((kernel_freq - 1) // 2, (kernel_time - 1) // 2)

        xf = F.unfold(
                x,
                kernel_size=(kernel_freq, kernel_time),
                padding=padding,
                stride=(1, 1),
        )

        xf = xf.view(
                -1,
                n_channel,
                kernel_freq,
                kernel_time,
                n_freq,
                n_time,
        )

        sf = xf * m

        sf = sf.view(
                -1,
                n_channel * kernel_freq * kernel_time,
                n_freq * n_time,
        )

        s = F.fold(
                sf,
                output_size=(n_freq, n_time),
                kernel_size=(kernel_freq, kernel_time),
                padding=padding,
                stride=(1, 1),
        ).view(
                -1,
                n_channel,
                n_freq,
                n_time,
        )

        return s

    def old_mask(self, x, m):
        # x.shape = (batch, n_channel, n_freq, n_time)
        # m.shape = (kernel_freq, kernel_time, batch, n_channel, n_freq, n_time)

        s = torch.zeros_like(x)

        _, n_channel, n_freq, n_time = x.shape
        kernel_freq, kernel_time, _, _, _, _ = m.shape

        # print(x.shape, m.shape)

        kernel_freq_half = (kernel_freq - 1) // 2
        kernel_time_half = (kernel_time - 1) // 2

        for ifreq in range(kernel_freq):
            for itime in range(kernel_time):
                df, dt = kernel_freq_half - ifreq, kernel_time_half - itime
                x = x.roll(shifts=(df, dt), dims=(2, 3))

                # if `df` > 0:
                #     x[:, :, :df, :] = 0
                # elif `df` < 0:
                #     x[:, :, df:, :] = 0

                # if `dt` > 0:
                #     x[:, :, :, :dt] = 0
                # elif `dt` < 0:
                #     x[:, :, :, dt:] = 0

                fslice = slice(max(0, df), min(n_freq, n_freq + df))
                tslice = slice(max(0, dt), min(n_time, n_time + dt))

                s[:, :, fslice, tslice] += x[:, :, fslice, tslice] * m[ifreq,
                                                                     itime, :,
                                                                     :, fslice,
                                                                     tslice]

        return s


class MultiSourceMultiPatchingMaskBandSplitCoreRNN(
        PatchingMaskBandsplitCoreBase
):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: List[Tuple[float, float]],
            mask_kernel_freq: int,
            mask_kernel_time: int,
            conv_kernel_freq: int,
            conv_kernel_time: int,
            kernel_norm_mlp_version: int,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            overlapping_band: bool = False,
            freq_weights: Optional[List[torch.Tensor]] = None,
            n_freq: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.band_split = BandSplitModule(
                in_channel=in_channel,
                band_specs=band_specs,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                emb_dim=emb_dim,
        )

        self.tf_model = (
                SeqBandModellingModule(
                        n_modules=n_sqm_modules,
                        emb_dim=emb_dim,
                        rnn_dim=rnn_dim,
                        bidirectional=bidirectional,
                        rnn_type=rnn_type,
                )
        )

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if overlapping_band:
            assert freq_weights is not None
            assert n_freq is not None
            self.mask_estim = nn.ModuleDict(
                    {
                            stem: PatchingMaskEstimationModule(
                                    band_specs=band_specs,
                                    freq_weights=freq_weights,
                                    n_freq=n_freq,
                                    emb_dim=emb_dim,
                                    mlp_dim=mlp_dim,
                                    in_channel=in_channel,
                                    hidden_activation=hidden_activation,
                                    hidden_activation_kwargs=hidden_activation_kwargs,
                                    complex_mask=complex_mask,
                                    mask_kernel_freq=mask_kernel_freq,
                                    mask_kernel_time=mask_kernel_time,
                                    conv_kernel_freq=conv_kernel_freq,
                                    conv_kernel_time=conv_kernel_time,
                                    kernel_norm_mlp_version=kernel_norm_mlp_version
                            )
                            for stem in stems
                    }
            )
        else:
            raise NotImplementedError
