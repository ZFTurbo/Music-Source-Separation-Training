from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from models.bandit.core.model._spectral import _SpectralComponent
from models.bandit.core.model.bsrnn.utils import (
    BarkBandsplitSpecification, BassBandsplitSpecification,
    DrumBandsplitSpecification,
    EquivalentRectangularBandsplitSpecification, MelBandsplitSpecification,
    MusicalBandsplitSpecification, OtherBandsplitSpecification,
    TriangularBarkBandsplitSpecification, VocalBandsplitSpecification,
)
from .core import (
    MultiSourceMultiMaskBandSplitCoreConv,
    MultiSourceMultiMaskBandSplitCoreRNN,
    MultiSourceMultiMaskBandSplitCoreTransformer,
    MultiSourceMultiPatchingMaskBandSplitCoreRNN, SingleMaskBandsplitCoreRNN,
    SingleMaskBandsplitCoreTransformer,
)

import pytorch_lightning as pl

def get_band_specs(band_specs, n_fft, fs, n_bands=None):
    if band_specs in ["dnr:speech", "dnr:vox7", "musdb:vocals", "musdb:vox7"]:
        bsm = VocalBandsplitSpecification(
                nfft=n_fft, fs=fs
        ).get_band_specs()
        freq_weights = None
        overlapping_band = False
    elif "tribark" in band_specs:
        assert n_bands is not None
        specs = TriangularBarkBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "bark" in band_specs:
        assert n_bands is not None
        specs = BarkBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "erb" in band_specs:
        assert n_bands is not None
        specs = EquivalentRectangularBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "musical" in band_specs:
        assert n_bands is not None
        specs = MusicalBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif band_specs == "dnr:mel" or "mel" in band_specs:
        assert n_bands is not None
        specs = MelBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    else:
        raise NameError

    return bsm, freq_weights, overlapping_band


def get_band_specs_map(band_specs_map, n_fft, fs, n_bands=None):
    if band_specs_map == "musdb:all":
        bsm = {
                "vocals": VocalBandsplitSpecification(
                        nfft=n_fft, fs=fs
                ).get_band_specs(),
                "drums": DrumBandsplitSpecification(
                        nfft=n_fft, fs=fs
                ).get_band_specs(),
                "bass": BassBandsplitSpecification(
                        nfft=n_fft, fs=fs
                ).get_band_specs(),
                "other": OtherBandsplitSpecification(
                        nfft=n_fft, fs=fs
                ).get_band_specs(),
        }
        freq_weights = None
        overlapping_band = False
    elif band_specs_map == "dnr:vox7":
        bsm_, freq_weights, overlapping_band = get_band_specs(
                "dnr:speech", n_fft, fs, n_bands
        )
        bsm = {
                "speech": bsm_,
                "music": bsm_,
                "effects": bsm_
        }
    elif "dnr:vox7:" in band_specs_map:
        stem = band_specs_map.split(":")[-1]
        bsm_, freq_weights, overlapping_band = get_band_specs(
                "dnr:speech", n_fft, fs, n_bands
        )
        bsm = {
                stem: bsm_
        }
    else:
        raise NameError

    return bsm, freq_weights, overlapping_band


class BandSplitWrapperBase(pl.LightningModule):
    bsrnn: nn.Module
    
    def __init__(self, **kwargs):
        super().__init__()


class SingleMaskMultiSourceBandSplitBase(
        BandSplitWrapperBase,
        _SpectralComponent
):
    def __init__(
            self,
            band_specs_map: Union[str, Dict[str, List[Tuple[float, float]]]],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
    ) -> None:
        super().__init__(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        if isinstance(band_specs_map, str):
            self.band_specs_map, self.freq_weights, self.overlapping_band = get_band_specs_map(
                band_specs_map,
                n_fft,
                fs,
                    n_bands=n_bands
                )

        self.stems = list(self.band_specs_map.keys())

    def forward(self, batch):
        audio = batch["audio"]

        with torch.no_grad():
            batch["spectrogram"] = {stem: self.stft(audio[stem]) for stem in
                                    audio}

        X = batch["spectrogram"]["mixture"]
        length = batch["audio"]["mixture"].shape[-1]

        output = {"spectrogram": {}, "audio": {}}

        for stem, bsrnn in self.bsrnn.items():
            S = bsrnn(X)
            s = self.istft(S, length)
            output["spectrogram"][stem] = S
            output["audio"][stem] = s

        return batch, output


class MultiMaskMultiSourceBandSplitBase(
        BandSplitWrapperBase,
        _SpectralComponent
):
    def __init__(
            self,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
    ) -> None:
        super().__init__(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        if isinstance(band_specs, str):
            self.band_specs, self.freq_weights, self.overlapping_band = get_band_specs(
                band_specs,
                n_fft,
                fs,
                n_bands
                )

        self.stems = stems

    def forward(self, batch):
        # with torch.no_grad():
        audio = batch["audio"]
        cond = batch.get("condition", None)
        with torch.no_grad():
            batch["spectrogram"] = {stem: self.stft(audio[stem]) for stem in
                                    audio}

        X = batch["spectrogram"]["mixture"]
        length = batch["audio"]["mixture"].shape[-1]

        output = self.bsrnn(X, cond=cond)
        output["audio"] = {}

        for stem, S in output["spectrogram"].items():
            s = self.istft(S, length)
            output["audio"][stem] = s

        return batch, output


class MultiMaskMultiSourceBandSplitBaseSimple(
        BandSplitWrapperBase,
        _SpectralComponent
):
    def __init__(
            self,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
    ) -> None:
        super().__init__(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        if isinstance(band_specs, str):
            self.band_specs, self.freq_weights, self.overlapping_band = get_band_specs(
                band_specs,
                n_fft,
                fs,
                n_bands
                )

        self.stems = stems

    def forward(self, batch):
        with torch.no_grad():
            X = self.stft(batch)
        length = batch.shape[-1]
        output = self.bsrnn(X, cond=None)
        res = []
        for stem, S in output["spectrogram"].items():
            s = self.istft(S, length)
            res.append(s)
        res = torch.stack(res, dim=1)
        return res


class SingleMaskMultiSourceBandSplitRNN(SingleMaskMultiSourceBandSplitBase):
    def __init__(
            self,
            in_channel: int,
            band_specs_map: Union[str, Dict[str, List[Tuple[float, float]]]],
            fs: int = 44100,
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
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                band_specs_map=band_specs_map,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        self.bsrnn = nn.ModuleDict(
                {
                        src: SingleMaskBandsplitCoreRNN(
                                band_specs=specs,
                                in_channel=in_channel,
                                require_no_overlap=require_no_overlap,
                                require_no_gap=require_no_gap,
                                normalize_channel_independently=normalize_channel_independently,
                                treat_channel_as_feature=treat_channel_as_feature,
                                n_sqm_modules=n_sqm_modules,
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                rnn_type=rnn_type,
                                mlp_dim=mlp_dim,
                                hidden_activation=hidden_activation,
                                hidden_activation_kwargs=hidden_activation_kwargs,
                                complex_mask=complex_mask,
                        )
                        for src, specs in self.band_specs_map.items()
                }
        )


class SingleMaskMultiSourceBandSplitTransformer(
        SingleMaskMultiSourceBandSplitBase
):
    def __init__(
            self,
            in_channel: int,
            band_specs_map: Union[str, Dict[str, List[Tuple[float, float]]]],
            fs: int = 44100,
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
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                band_specs_map=band_specs_map,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        self.bsrnn = nn.ModuleDict(
                {
                        src: SingleMaskBandsplitCoreTransformer(
                                band_specs=specs,
                                in_channel=in_channel,
                                require_no_overlap=require_no_overlap,
                                require_no_gap=require_no_gap,
                                normalize_channel_independently=normalize_channel_independently,
                                treat_channel_as_feature=treat_channel_as_feature,
                                n_sqm_modules=n_sqm_modules,
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                tf_dropout=tf_dropout,
                                mlp_dim=mlp_dim,
                                hidden_activation=hidden_activation,
                                hidden_activation_kwargs=hidden_activation_kwargs,
                                complex_mask=complex_mask,
                        )
                        for src, specs in self.band_specs_map.items()
                }
        )


class MultiMaskMultiSourceBandSplitRNN(MultiMaskMultiSourceBandSplitBase):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            cond_dim: int = 0,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
            use_freq_weights: bool = True,
            normalize_input: bool = False,
            mult_add_mask: bool = False,
            freeze_encoder: bool = False,
    ) -> None:
        super().__init__(
                stems=stems,
                band_specs=band_specs,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
                n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreRNN(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )

        self.normalize_input = normalize_input
        self.cond_dim = cond_dim

        if freeze_encoder:
            for param in self.bsrnn.band_split.parameters():
                param.requires_grad = False

            for param in self.bsrnn.tf_model.parameters():
                param.requires_grad = False


class MultiMaskMultiSourceBandSplitRNNSimple(MultiMaskMultiSourceBandSplitBaseSimple):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            cond_dim: int = 0,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
            use_freq_weights: bool = True,
            normalize_input: bool = False,
            mult_add_mask: bool = False,
            freeze_encoder: bool = False,
    ) -> None:
        super().__init__(
                stems=stems,
                band_specs=band_specs,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
                n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreRNN(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )

        self.normalize_input = normalize_input
        self.cond_dim = cond_dim

        if freeze_encoder:
            for param in self.bsrnn.band_split.parameters():
                param.requires_grad = False

            for param in self.bsrnn.tf_model.parameters():
                param.requires_grad = False


class MultiMaskMultiSourceBandSplitTransformer(
        MultiMaskMultiSourceBandSplitBase
):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            cond_dim: int = 0,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
            use_freq_weights: bool = True,
            normalize_input: bool = False,
            mult_add_mask: bool = False
    ) -> None:
        super().__init__(
                stems=stems,
                band_specs=band_specs,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
                n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreTransformer(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )



class MultiMaskMultiSourceBandSplitConv(
        MultiMaskMultiSourceBandSplitBase
):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            fs: int = 44100,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            cond_dim: int = 0,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
            use_freq_weights: bool = True,
            normalize_input: bool = False,
            mult_add_mask: bool = False
    ) -> None:
        super().__init__(
                stems=stems,
                band_specs=band_specs,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
                n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreConv(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )
class PatchingMaskMultiSourceBandSplitRNN(MultiMaskMultiSourceBandSplitBase):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]],
            kernel_norm_mlp_version: int = 1,
            mask_kernel_freq: int = 3,
            mask_kernel_time: int = 3,
            conv_kernel_freq: int = 1,
            conv_kernel_time: int = 1,
            fs: int = 44100,
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
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
    ) -> None:
        super().__init__(
                stems=stems,
                band_specs=band_specs,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
                n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiPatchingMaskBandSplitCoreRNN(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                mask_kernel_freq=mask_kernel_freq,
                mask_kernel_time=mask_kernel_time,
                conv_kernel_freq=conv_kernel_freq,
                conv_kernel_time=conv_kernel_time,
                kernel_norm_mlp_version=kernel_norm_mlp_version,
        )
