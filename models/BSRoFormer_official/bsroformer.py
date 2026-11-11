"""BSRoFormer — Band-Split RoPE Transformer for music source separation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from rotary_embedding_torch import RotaryEmbedding

from .roformer import RMSNorm, RoFormerBlock


class BandProjection(nn.Module):
    """Per-band norm + linear projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.ff = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.ff(self.norm(x))


class MaskEstimator(nn.Module):
    """Per-band mask estimator: norm -> linear -> tanh -> linear -> GLU."""

    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(feature_dim)
        self.ff_0 = nn.Linear(feature_dim, 4 * feature_dim)
        self.tanh = nn.Tanh()
        self.ff_1 = nn.Linear(4 * feature_dim, 2 * out_dim)
        self.glu = nn.GLU()

    def forward(self, x):
        return self.glu(self.ff_1(self.tanh(self.ff_0(self.norm(x)))))


class BandSplit(nn.Module):
    """Split spectrogram into equal-width subbands and project each to a feature vector.

    Each subspec [lo, hi] is divided into n_band equal bands. Band slices are
    precomputed in the channel-interleaved f*c axis for a single-pass forward.

    Input:  (b, c, t, f)
    Output: (b, t, k, d)   k = total bands across all subspecs
    """
    def __init__(
        self, num_feature, channel_num,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec= [24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.channel_num = channel_num
        self.band_transforms = nn.ModuleList()
        self.band_slices = []  # (lo_ch, hi_ch) in the f*c axis, one per band

        for (lo, hi), n_band in zip(subspec_idxs, n_band_per_subspec):
            n_bins = hi - lo + 1
            assert n_bins % n_band == 0
            bins_per_band = n_bins // n_band
            in_dim = bins_per_band * channel_num
            for i in range(n_band):
                lo_ch = (lo + i * bins_per_band) * channel_num
                self.band_slices.append((lo_ch, lo_ch + in_dim))
                self.band_transforms.append(BandProjection(in_dim, num_feature))

    def forward(self, x):
        # x: (b, c, t, f) -> (b, t, f*c) with f as outer index
        b, c, t, f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, t, f * c)
        outs = [proj(x[:, :, lo:hi]) for (lo, hi), proj in zip(self.band_slices, self.band_transforms)]
        return torch.stack(outs).permute(1, 2, 0, 3)  # (k, b, t, d) -> (b, t, k, d)


class MaskEstimation(nn.Module):
    """Estimate a complex mask per subband; scatter into full frequency grid.

    Mirrors BandSplit: same subspec/band structure, same absolute freq_slices.
    Each band's estimate is placed into its slice; the flat mask is reshaped to (b, c, t, f).

    Input:  (b, t, k, d)
    Output: (b, c, t, f)
    """
    def __init__(
        self, num_feature, channel_num,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec=[24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.channel_num = channel_num
        self.mask_estimators = nn.ModuleList()
        self.freq_slices = []  # (lo_ch, hi_ch) in the f*c axis, one per band

        for (lo, hi), n_band in zip(subspec_idxs, n_band_per_subspec):
            n_bins = hi - lo + 1
            assert n_bins % n_band == 0
            bins_per_band = n_bins // n_band
            out_dim = bins_per_band * channel_num
            for i in range(n_band):
                lo_ch = (lo + i * bins_per_band) * channel_num
                self.freq_slices.append((lo_ch, lo_ch + out_dim))
                self.mask_estimators.append(MaskEstimator(num_feature, out_dim))

        self.num_freq_ch = self.freq_slices[-1][1]

    def forward(self, x):
        # x: (b, t, k, d)  ->  mask: (b, c, t, f)
        b, t, _, _ = x.shape
        mask = torch.zeros(b, t, self.num_freq_ch, dtype=x.dtype, device=x.device) # (b, t, f*c)
        for band_idx, ((lo, hi), estimator) in enumerate(zip(self.freq_slices, self.mask_estimators)):
            mask[:, :, lo:hi] += estimator(x[:, :, band_idx])
        # (b, t, f*c) -> (b, c, t, f)
        f = self.num_freq_ch // self.channel_num
        return mask.view(b, t, f, self.channel_num).permute(0, 3, 1, 2).contiguous()


class BSRoFormer(nn.Module):
    """Band-Split RoPE Transformer for music source separation."""
    def __init__(
        self, *, input_channels, output_channels,
        depth, num_feature, window_size, hop_size,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec=[24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.output_channels = output_channels

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            power=None, window_fn=torch.hann_window, center=True, pad_mode="reflect")
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            window_fn=torch.hann_window, center=True, pad_mode="reflect")

        stft_channels = input_channels * 2  # real + imag
        dim_head = num_feature // 8

        self.multi_band_transform = BandSplit(
            num_feature, stft_channels, subspec_idxs, n_band_per_subspec)
        self.rotary_emb_t = RotaryEmbedding(dim=dim_head)
        self.rotary_emb_k = RotaryEmbedding(dim=dim_head)
        self.transformer_stack = nn.ModuleList([
            RoFormerBlock(num_feature, 8, dim_head, num_feature * 4, 0)
            for _ in range(depth)
        ])
        self.mask_estimation = MaskEstimation(
            num_feature, output_channels * 2, subspec_idxs, n_band_per_subspec)

    def forward(self, mix):
        """mix: (batch, channels, samples) -> separated: (batch, channels, samples)"""
        batch_size = mix.shape[0]

        # STFT -> complex spectrogram; disable autocast (requires float32)
        with torch.autocast(device_type="cuda", enabled=False):
            complex_spec = self.stft(mix.float())
        # (b, c, f, t) -> (b, c, t, f)
        real = complex_spec.real.transpose(2, 3)
        imag = complex_spec.imag.transpose(2, 3)

        # Band split: spectrogram -> per-band feature vectors
        band_input = torch.cat((real, imag), dim=1)[..., :-1]  # drop Nyquist bin
        band_features = self.multi_band_transform(band_input)   # (b, t, k, d)

        # Alternating time/band transformers with rotary embeddings
        b, t, k, d = batch_size, *band_features.shape[1:]
        for block in self.transformer_stack:
            # (b, t, k, d) -> (b*k, t, d): attend across time within each band
            band_features = block.transform_t(
                band_features.permute(0, 2, 1, 3).contiguous().reshape(b * k, t, d),
                self.rotary_emb_t)
            # (b*k, t, d) -> (b*t, k, d): attend across bands within each time step
            band_features = block.transform_k(
                band_features.view(b, k, t, d).permute(0, 2, 1, 3).contiguous().reshape(b * t, k, d),
                self.rotary_emb_k)
            # (b*t, k, d) -> (b, t, k, d)
            band_features = band_features.view(b, t, k, d)

        # Estimate complex mask and apply to input spectrogram
        complex_mask = F.pad(self.mask_estimation(band_features), (0, 1))  # restore Nyquist
        mask_real = complex_mask[:, :self.output_channels]
        mask_imag = complex_mask[:, self.output_channels:]

        with torch.autocast(device_type="cuda", enabled=False):
            separated_real = real.float() * mask_real.float() - imag.float() * mask_imag.float()
            separated_imag = real.float() * mask_imag.float() + imag.float() * mask_real.float()
            # (b, c, t, f) -> (b, c, f, t) for iSTFT
            audio = self.istft(torch.complex(
                separated_real.transpose(2, 3).contiguous(),
                separated_imag.transpose(2, 3).contiguous()), length=mix.shape[-1])

        return audio.to(mix.dtype).unsqueeze(1)


# ── Tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    passed, failed = 0, 0

    def check(name, cond, detail=""):
        global passed, failed
        tag, suffix = ("PASS", "") if cond else ("FAIL", f"  ({detail})" if detail else "")
        print(f"  [{tag}] {name}{suffix}")
        if cond: passed += 1
        else:    failed += 1

    # Small config: two subspecs, 4 bands each, 4 bins per band -> 32 total freq bins
    # window_size=64 -> n_fft/2 + 1 = 33 freq bins; drop Nyquist -> 32 ✓
    SUBSPEC = [[0, 15], [16, 31]]
    N_BAND  = [4, 4]
    C       = 4   # stft_channels (input_channels * 2)
    D       = 64  # num_feature
    K       = sum(N_BAND)                          # total bands = 8
    N_FREQ_CH = (SUBSPEC[-1][1] + 1) * C          # (32) * 4 = 128

    # ── BandSplit ─────────────────────────────────────────────
    print("BandSplit")
    bs = BandSplit(num_feature=D, channel_num=C, subspec_idxs=SUBSPEC, n_band_per_subspec=N_BAND)

    check("total bands == sum(n_band_per_subspec)", len(bs.band_slices) == K)
    check("band_slices start at 0", bs.band_slices[0][0] == 0)
    check("band_slices end at (last_bin+1)*c", bs.band_slices[-1][1] == N_FREQ_CH)

    contiguous = all(bs.band_slices[i][1] == bs.band_slices[i + 1][0]
                     for i in range(len(bs.band_slices) - 1))
    check("band_slices are contiguous (no gaps, no overlaps)", contiguous)

    # Each band's slice must fall within its subspec's freq range
    in_range, idx = True, 0
    for (lo, hi), n_band in zip(SUBSPEC, N_BAND):
        for _ in range(n_band):
            s_lo, s_hi = bs.band_slices[idx][0] // C, (bs.band_slices[idx][1] - 1) // C
            if not (lo <= s_lo and s_hi <= hi):
                in_range = False
            idx += 1
    check("every band slice falls within its subspec freq range", in_range)

    # BandProjection input dim must match its slice width
    dims_ok = all(proj.ff.in_features == hi - lo
                  for proj, (lo, hi) in zip(bs.band_transforms, bs.band_slices))
    check("BandProjection in_dim matches slice width", dims_ok)

    x = torch.randn(2, C, 20, 32)
    out = bs(x)
    check("output shape (b, t, k, d)", out.shape == (2, 20, K, D), f"got {out.shape}")

    # ── MaskEstimation ────────────────────────────────────────
    print("\nMaskEstimation")
    me = MaskEstimation(num_feature=D, channel_num=C, subspec_idxs=SUBSPEC, n_band_per_subspec=N_BAND)

    check("total estimators == total bands", len(me.mask_estimators) == K)
    check("freq_slices identical to BandSplit band_slices", me.freq_slices == bs.band_slices)
    check("num_freq_ch == (last_bin+1)*c", me.num_freq_ch == N_FREQ_CH)

    sizes_ok = all(est.ff_1.out_features == 2 * (hi - lo)
                   for est, (lo, hi) in zip(me.mask_estimators, me.freq_slices))
    check("MaskEstimator output sizes match freq slices", sizes_ok)

    x = torch.randn(2, 20, K, D)
    out = me(x)
    check("output shape (b, c, t, f)", out.shape == (2, C, 20, N_FREQ_CH // C), f"got {out.shape}")

    # Scatter correctness: only band 5 active (zero input -> RMSNorm clamps to 0, no bias leak)
    x_zero = torch.zeros(1, 5, K, D)
    x_zero[:, :, 5, :] = 1.0
    out_flat = me(x_zero).permute(0, 2, 3, 1).contiguous().view(1, 5, N_FREQ_CH)
    lo5, hi5 = me.freq_slices[5]
    check("scatter: outside band 5 slice is zero",
          out_flat[:, :, :lo5].abs().max() == 0 and out_flat[:, :, hi5:].abs().max() == 0)
    check("scatter: inside band 5 slice is non-zero",
          out_flat[:, :, lo5:hi5].abs().max() > 0)

    # ── BSRoFormer forward (eval) ─────────────────────────────
    print("\nBSRoFormer forward (eval)")
    cfg = dict(
        input_channels=2, output_channels=2,
        depth=2, num_feature=D, window_size=2048, hop_size=441,
        # use default subspec_idxs/n_band_per_subspec: covers bins 0-1023
        # which matches window_size=2048 -> 1025 STFT bins, drop Nyquist -> 1024 bins
    )
    model = BSRoFormer(**cfg).to(device).eval()
    # Sample count must be a multiple of hop_size: iSTFT returns floor(L/hop)*hop samples
    L = 44100  # = 100 * hop_size(441)
    mix = torch.randn(2, 2, L).to(device)
    with torch.no_grad():
        out = model(mix)
    check("output shape matches input", out.shape == mix.shape, f"got {out.shape}")
    check("output is finite", out.isfinite().all().item())

    # ── Gradient flow (train) ─────────────────────────────────
    print("\nGradient flow (train)")
    model.train()
    model(torch.randn(1, 2, L).to(device)).mean().backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    check("all parameters receive gradients", all(grads), f"{sum(grads)}/{len(grads)}")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
