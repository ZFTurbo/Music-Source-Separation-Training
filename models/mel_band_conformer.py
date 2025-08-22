from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from conformer import Conformer
from torch.nn import Module, ModuleList
from librosa import filters
from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype
from einops import rearrange, pack, unpack, reduce, repeat

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        # split along band dimension and run per-band MLP
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        # split input into predefined frequency-band chunks
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        # stack back as (bands) axis
        return torch.stack(outs, dim=-2)


class MelBandConformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_conformer_depth: int = 2,
        freq_conformer_depth: int = 2,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        # Conformer params
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        # STFT
        dim_freqs_in: int = 1025,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        stft_window_fn: Optional[Callable] = None,
        # Loss
        mask_estimator_depth: int = 1,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length: bool = False,

        use_torch_checkpoint: bool = False,
        skip_connection: bool = False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList([])

        # Layers per block: [ time-Conformer, freq-Conformer ]
        conformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )

        for _ in range(depth):
            time_block = Conformer(depth=time_conformer_depth, **conformer_kwargs)
            freq_block = Conformer(depth=freq_conformer_depth, **conformer_kwargs)
            self.layers.append(nn.ModuleList([time_block, freq_block]))

        self.stft_window_fn = partial(stft_window_fn or torch.hann_window, stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        # number of frequency bins produced by STFT (ignoring complex axis)
        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_n_fft),
            return_complex=True
        ).shape[1]

        # build mel filter bank to define band grouping
        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)
        # ensure coverage at the boundaries
        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequency bins must be covered by bands'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            # duplicate indices for stereo by interleaving channels along the freq axis
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        # BandSplit and MaskEstimator — same structure as your original
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([
            MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=4,  # could be exposed as a parameter
            )
            for _ in range(num_stems)
        ])

        # multi-resolution STFT loss setup
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

    def forward(
        self,
        raw_audio: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        return_loss_breakdown: bool = False
    ):
        """
        b - batch
        f - freq
        t - time
        s - audio channel (1 mono / 2 stereo)
        n - stems
        c - complex (2)
        d - feature dim
        """
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'set stereo=True for stereo input (C=2), stereo=False for mono (C=1)'

        # --- STFT ---
        raw_audio_flat, packed_shape = raw_audio.reshape(-1, raw_audio.shape[-1]), raw_audio.shape[:2]
        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)                  # (B*C, F, T, 2)
        stft_repr = stft_repr.view(*packed_shape, *stft_repr.shape[1:])  # (b, s, f, t, c)

        # fold channel into frequency axis (as in your setup)
        stft_repr_fs = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # index frequencies by mel bands
        b_idx = torch.arange(batch, device=device)[..., None]
        x = stft_repr_fs[b_idx, self.freq_indices]                 # (b, sum(freqs_in_bands), t, c)
        x = rearrange(x, 'b f t c -> b t (f c)')                   # flatten complex axis into features

        # --- BandSplit -> (b, t, bands, dim) ---
        if self.use_torch_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # --- Axial Conformer (time, then freq) ---
        store = [None] * len(self.layers)

        for i, (time_conf, freq_conf) in enumerate(self.layers):
            # Time axis: (b, t, bands, d) -> ((b*bands), t, d)
            bsz, tlen, bands, d = x.shape
            x_time = rearrange(x, 'b t f d -> (b f) t d')

            if self.use_torch_checkpoint:
                x_time = torch.utils.checkpoint.checkpoint(time_conf, x_time, use_reentrant=False)
            else:
                x_time = time_conf(x_time)

            x = rearrange(x_time, '(b f) t d -> b t f d', b=bsz, f=bands)

            # Freq axis: (b, t, f, d) -> ((b*t), f, d)
            bsz, tlen, bands, d = x.shape
            x_freq = rearrange(x, 'b t f d -> (b t) f d')

            if self.use_torch_checkpoint:
                x_freq = torch.utils.checkpoint.checkpoint(freq_conf, x_freq, use_reentrant=False)
            else:
                x_freq = freq_conf(x_freq)

            x = rearrange(x_freq, '(b t) f d -> b t f d', b=bsz, t=tlen)

            if self.skip_connection:
                store[i] = x if store[i] is None else store[i] + x

        # --- Mask estimation ---
        # (b, t, f_bands, d) -> per-stem MLP over bands
        if self.use_torch_checkpoint:
            masks = torch.stack([torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
                                 for fn in self.mask_estimators], dim=1)
        else:
            masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        # --- Complex modulation ---
        stft_repr_c = rearrange(stft_repr, 'b s f t c -> b 1 (f s) t c')
        stft_repr_c = torch.view_as_complex(stft_repr_c)          # (b, 1, F*S, T)
        masks_c = torch.view_as_complex(masks)                     # (b, n, F*S, T)

        masks_c = masks_c.type(stft_repr_c.dtype)

        scatter_idx = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=self.num_stems, t=stft_repr_c.shape[-1])
        stft_repr_expanded = repeat(stft_repr_c, 'b 1 ... -> b n ...', n=self.num_stems)

        masks_summed = torch.zeros_like(stft_repr_expanded).scatter_add_(2, scatter_idx, masks_c)
        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=self.audio_channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)
        stft_mod = stft_repr_c * masks_averaged

        # --- iSTFT ---
        stft_mod = rearrange(stft_mod, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        recon_audio = torch.istft(
            stft_mod,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length
        )
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # Loss
        if target is None:
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0
        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss += F.l1_loss(recon_Y, target_Y)

        total_loss = loss + self.multi_stft_resolution_loss_weight * multi_stft_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
