from typing import Dict, List, Optional

import torch
import torchaudio as ta
from torch import nn
import pytorch_lightning as pl

from .bandsplit import BandSplitModule
from .maskestim import OverlappingMaskEstimationModule
from .tfmodel import SeqBandModellingModule
from .utils import MusicalBandsplitSpecification



class BaseEndToEndModule(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()


class BaseBandit(BaseEndToEndModule):
    def __init__(
        self,
        in_channels: int,
        fs: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
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
    ):
        super().__init__()

        self.in_channels = in_channels

        self.instantitate_spectral(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

        self.instantiate_bandsplit(
            in_channels=in_channels,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
            n_fft=n_fft,
            fs=fs,
        )

        self.instantiate_tf_modelling(
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )

    def instantitate_spectral(
        self,
        n_fft: int = 2048,
        win_length: Optional[int] = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        power: Optional[int] = None,
        normalized: bool = True,
        center: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
    ):
        assert power is None

        window_fn = torch.__dict__[window_fn]

        self.stft = ta.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

        self.istft = ta.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0,
            window_fn=window_fn,
            wkwargs=wkwargs,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

    def instantiate_bandsplit(
        self,
        in_channels: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        n_fft: int = 2048,
        fs: int = 44100,
    ):
        assert band_type == "musical"

        self.band_specs = MusicalBandsplitSpecification(
            nfft=n_fft, fs=fs, n_bands=n_bands
        )

        self.band_split = BandSplitModule(
            in_channels=in_channels,
            band_specs=self.band_specs.get_band_specs(),
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )

    def instantiate_tf_modelling(
        self,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ):
        try:
            self.tf_model = torch.compile(
                SeqBandModellingModule(
                    n_modules=n_sqm_modules,
                    emb_dim=emb_dim,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                ),
                disable=True,
            )
        except Exception as e:
            self.tf_model = SeqBandModellingModule(
                    n_modules=n_sqm_modules,
                    emb_dim=emb_dim,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                )

    def mask(self, x, m):
        return x * m

    def forward(self, batch, mode="train"):
        # Model takes mono as input we give stereo, so we do process of each channel independently
        init_shape = batch.shape
        if not isinstance(batch, dict):
            mono = batch.view(-1, 1, batch.shape[-1])
            batch = {
                "mixture": {
                    "audio": mono
                }
            }

        with torch.no_grad():
            mixture = batch["mixture"]["audio"]

            x = self.stft(mixture)
            batch["mixture"]["spectrogram"] = x

            if "sources" in batch.keys():
                for stem in batch["sources"].keys():
                    s = batch["sources"][stem]["audio"]
                    s = self.stft(s)
                    batch["sources"][stem]["spectrogram"] = s

        batch = self.separate(batch)

        if 1:
            b = []
            for s in self.stems:
                # We need to obtain stereo again
                r = batch['estimates'][s]['audio'].view(-1, init_shape[1], init_shape[2])
                b.append(r)
            # And we need to return back tensor and not independent stems
            batch = torch.stack(b, dim=1)
        return batch

    def encode(self, batch):
        x = batch["mixture"]["spectrogram"]
        length = batch["mixture"]["audio"].shape[-1]

        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)

        return x, q, length

    def separate(self, batch):
        raise NotImplementedError


class Bandit(BaseBandit):
    def __init__(
        self,
        in_channels: int,
        stems: List[str],
        band_type: str = "musical",
        n_bands: int = 64,
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
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        fs: int = 44100,
        stft_precisions="32",
        bandsplit_precisions="bf16",
        tf_model_precisions="bf16",
        mask_estim_precisions="bf16",
    ):
        super().__init__(
            in_channels=in_channels,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
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
            fs=fs,
        )

        self.stems = stems

        self.instantiate_mask_estim(
            in_channels=in_channels,
            stems=stems,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )

    def instantiate_mask_estim(
        self,
        in_channels: int,
        stems: List[str],
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = False,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = nn.ModuleDict(
            {
                stem: OverlappingMaskEstimationModule(
                    band_specs=self.band_specs.get_band_specs(),
                    freq_weights=self.band_specs.get_freq_weights(),
                    n_freq=n_freq,
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channels=in_channels,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    use_freq_weights=use_freq_weights,
                )
                for stem in stems
            }
        )

    def separate(self, batch):
        batch["estimates"] = {}

        x, q, length = self.encode(batch)

        for stem, mem in self.mask_estim.items():
            m = mem(q)

            s = self.mask(x, m.to(x.dtype))
            s = torch.reshape(s, x.shape)
            batch["estimates"][stem] = {
                "audio": self.istft(s, length),
                "spectrogram": s,
            }

        return batch

