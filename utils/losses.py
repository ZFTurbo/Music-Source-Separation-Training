import argparse
from typing import Any, Optional, Callable, Union

import auraloss
import torch.nn.functional as F
import torch
from ml_collections import ConfigDict
from torch import nn
from torch_log_wmse import LogWMSE
import librosa
from torchaudio.transforms import AmplitudeToDB


class BleedFullPenaltyLoss(nn.Module):
    """
    Differentiable penalty for missing content (Fullness) or added artifacts (Bleedless).
    mode: 'fullness' penalizes regions where Reference > Estimate.
    mode: 'bleedless' penalizes regions where Estimate > Reference.
    """

    def __init__(self, mode: str = 'fullness', sr: int = 44100, n_fft: int = 4096, hop_length: int = 1024,
                 n_mels: int = 512):
        super().__init__()
        if mode not in ['fullness', 'bleedless']:
            raise ValueError("mode must be either 'fullness' or 'bleedless'")

        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.window = torch.hann_window(n_fft)

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        self.mel_filter_bank = torch.from_numpy(mel_basis).float()

        self.amplitude_to_db = AmplitudeToDB(stype="magnitude", top_db=80)

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        device = estimate.device
        window = self.window.to(device)
        mel_filter_bank = self.mel_filter_bank.to(device)
        self.amplitude_to_db = self.amplitude_to_db.to(device)

        lenc = estimate.shape[-1]
        est_reshaped = estimate.view(-1, lenc)
        ref_reshaped = reference.view(-1, lenc)

        D_est = torch.abs(torch.stft(est_reshaped, n_fft=self.n_fft, hop_length=self.hop_length,
                                     window=window, return_complex=True, pad_mode="constant"))
        D_ref = torch.abs(torch.stft(ref_reshaped, n_fft=self.n_fft, hop_length=self.hop_length,
                                     window=window, return_complex=True, pad_mode="constant"))

        S_est_mel = torch.matmul(mel_filter_bank, D_est)
        S_ref_mel = torch.matmul(mel_filter_bank, D_ref)

        S_est_db = self.amplitude_to_db(S_est_mel)
        S_ref_db = self.amplitude_to_db(S_ref_mel)

        # Choose what to penalize depending on the mode
        if self.mode == 'fullness':
            penalty = torch.relu(S_ref_db - S_est_db)
        else:  # 'bleedless'
            penalty = torch.relu(S_est_db - S_ref_db)

        # Calculate the mean only where there is an actual loss (as in metrics.py)
        # Use clamp(min=1) to prevent division by zero without triggering CPU-GPU synchronization
        active_elements_count = torch.count_nonzero(penalty)
        loss = torch.sum(penalty) / active_elements_count.clamp(min=1)

        return loss


def multistft_loss(
    y_: torch.Tensor,
    y: torch.Tensor,
    loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    Compute a (multi-resolution) STFT-based loss on waveforms.

    Reshapes inputs to (B, C*T, L) when needed and delegates to a provided
    multi-resolution STFT criterion (e.g., `auraloss.freq.MultiResolutionSTFTLoss`),
    a widely used spectral loss for audio synthesis/enhancement that compares
    magnitudes across multiple STFT settings.
    See: Steinmetz & Reiss, 2020, “auraloss: Audio-focused loss functions in PyTorch”.

    Args:
        y_ (torch.Tensor): Predicted waveform tensor of shape (B, C, T) or (B, S, C, T).
        y (torch.Tensor): Target waveform tensor with a compatible shape.
        loss_multistft (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            A callable implementing the MR-STFT loss.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """

    if len(y_.shape) == 4:
        y1_ = y_.reshape(y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3])
    elif len(y_.shape) == 3:
        y1_ = y_
    if len(y.shape) == 4:
        y1 = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])
    elif len(y_.shape) == 3:
        y1 = y
    if len(y_.shape) not in [3, 4]:
        raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 3 or 4 dimensions.")
    return loss_multistft(y1_, y1)


def masked_loss(
    y_: torch.Tensor,
    y: torch.Tensor,
    q: float,
    coarse: bool = True
) -> torch.Tensor:
    """
    Robust, quantile-masked MSE (“trimmed” MSE).

    Computes an elementwise MSE, optionally averages spatial dims (“coarse”),
    then masks out the largest residuals by keeping values below the `q`-quantile.
    This yields robustness to outliers akin to trimmed/robust regression losses.
    See classical robust estimation: Huber, 1964; Rousseeuw & Leroy, 1987.

    Args:
        y_ (torch.Tensor): Predicted tensor matching `y`'s shape.
        y (torch.Tensor): Ground-truth tensor.
        q (float): Quantile in (0, 1] used to keep low-error elements.
        coarse (bool, optional): If True, average over last two dims before masking.
            Defaults to True.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """

    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = loss.mean(dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    quantile = torch.quantile(loss.detach(), q, interpolation='linear', dim=1, keepdim=True)
    mask = loss < quantile
    return (loss * mask).mean()


def spec_rmse_loss(
    estimate: torch.Tensor,
    sources: torch.Tensor,
    stft_config: dict,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    RMSE in the complex STFT domain.

    Computes STFT for prediction and target, represents complex values as
    real+imag pairs, and applies RMSE (L2) over the spectral representation.
    Spectral-domain L2/RMSE losses are common in speech/music enhancement.
    See, e.g., Steinmetz & Reiss, 2020; Yamamoto et al., 2020 (Parallel WaveGAN).

    Args:
        estimate (torch.Tensor): Predicted time-domain signal(s), e.g., (B, S, C, T).
        sources (torch.Tensor): Target time-domain signal(s), matching shape.
        stft_config (dict): Parameters for `torch.stft` (e.g., n_fft, hop_length, win_length).

    Returns:
        torch.Tensor: Scalar loss tensor.
    """

    lenc = estimate.shape[-1]
    spec_estimate = estimate.view(-1, lenc)
    spec_sources = sources.view(-1, lenc)

    spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)

    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
    spec_estimate = spec_estimate.view(*new_shape)
    spec_sources = spec_sources.view(*new_shape)

    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')

    dims = tuple(range(2, loss.dim()))
    loss = (loss.mean(dims) + eps).sqrt().mean(dim=(0, 1))

    return loss


def spec_masked_loss(
    estimate: torch.Tensor,
    sources: torch.Tensor,
    stft_config: dict,
    q: float = 0.9,
    coarse: bool = True
) -> torch.Tensor:
    """
    Quantile-masked MSE in the complex STFT domain.

    Computes a complex STFT for prediction and target, forms an elementwise MSE
    in the spectral domain, optionally averages spatial/frequency dims (“coarse”),
    and masks out the highest-error elements using the `q`-quantile threshold for
    robustness to outliers. Related to trimmed/robust spectral losses.
    See: Huber, 1964; Rousseeuw & Leroy, 1987; spectral losses as in Steinmetz & Reiss, 2020.

    Args:
        estimate (torch.Tensor): Predicted time-domain signal(s), e.g., (B, S, C, T).
        sources (torch.Tensor): Target time-domain signal(s), matching shape.
        stft_config (dict): Parameters for `torch.stft`.
        q (float, optional): Quantile in (0, 1] to keep low-error elements. Defaults to 0.9.
        coarse (bool, optional): If True, average over spectral dims before masking. Defaults to True.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """

    lenc = estimate.shape[-1]
    spec_estimate = estimate.view(-1, lenc)
    spec_sources = sources.view(-1, lenc)

    spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)

    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
    spec_estimate = spec_estimate.view(*new_shape)
    spec_sources = spec_sources.view(*new_shape)

    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')

    if coarse:
        loss = loss.mean(dim=(-3, -2))

    loss = loss.reshape(loss.shape[0], -1)

    quantile = torch.quantile(
        loss.detach(),
        q,
        interpolation='linear',
        dim=1,
        keepdim=True
    )

    mask = loss < quantile

    masked_loss = (loss * mask).mean()

    return masked_loss


def l1_snr_loss(y_: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L1-SNR loss in time domain.

    L1-based signal-to-noise ratio loss (without additional regularization).
    From torch-l1-snr package.

    Args:
        y_ (torch.Tensor): Predicted waveform tensor.
        y (torch.Tensor): Target waveform tensor.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """
    from torch_l1_snr import L1SNRLoss
    loss_fn = L1SNRLoss(name="l1_snr")
    return loss_fn(y_, y)


def l1_snr_db_loss(y_: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L1-SNR loss with dB-scale level regularization.

    Extends L1-SNR with adaptive level-matching regularization in dB scale.
    From torch-l1-snr package.

    Args:
        y_ (torch.Tensor): Predicted waveform tensor.
        y (torch.Tensor): Target waveform tensor.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """
    from torch_l1_snr import L1SNRDBLoss
    loss_fn = L1SNRDBLoss(name="l1_snr_db")
    return loss_fn(y_, y)


def stft_l1_snr_db_loss(y_: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L1-SNR loss in multi-resolution STFT domain.

    Applies L1-SNR to complex STFT (real/imaginary) across multiple resolutions.
    From torch-l1-snr package.

    Args:
        y_ (torch.Tensor): Predicted waveform tensor.
        y (torch.Tensor): Target waveform tensor.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """
    from torch_l1_snr import STFTL1SNRDBLoss
    loss_fn = STFTL1SNRDBLoss(name="stft_l1_snr_db")
    return loss_fn(y_, y)


def multi_l1_snr_db_loss(y_: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Combined time + STFT domain L1-SNR loss.

    Balances time-domain and spectral-domain L1-SNR with optional regularization.
    This is the recommended loss from torch-l1-snr for most use cases.
    From torch-l1-snr package.

    Args:
        y_ (torch.Tensor): Predicted waveform tensor.
        y (torch.Tensor): Target waveform tensor.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """
    from torch_l1_snr import MultiL1SNRDBLoss
    loss_fn = MultiL1SNRDBLoss(name="multi_l1_snr_db")
    return loss_fn(y_, y)


def choice_loss(
    args: argparse.Namespace,
    config: ConfigDict
) -> Callable[[Any, Any, Union[Any, None]], torch.Tensor]:
    """
    Build a composite loss from CLI/config options.

    Returns a callable that sums enabled terms (with per-term coefficients):
    - `masked_loss`: robust, quantile-masked MSE (trimmed MSE; Huber, 1964; Rousseeuw & Leroy, 1987).
    - `mse_loss`: standard mean squared error.
    - `l1_loss`: mean absolute error.
    - `multistft_loss`: multi-resolution STFT magnitude loss (Steinmetz & Reiss, 2020).
    - `log_wmse_loss`: weighted MSE operating in a log/spectral perceptual space (log-weighted MSE).
    - `l1_snr_loss`: L1-SNR loss in time domain (Watcharasupat et al., 2023).
    - `l1_snr_db_loss`: L1-SNR with dB-scale level regularization.
    - `stft_l1_snr_db_loss`: L1-SNR in multi-resolution STFT domain.
    - `multi_l1_snr_db_loss`: combined time + STFT domain L1-SNR (recommended).
    - `spec_rmse_loss`: RMSE in complex STFT domain.
    - `spec_masked_loss`: quantile-masked spectral MSE (robust spectral loss).

    Args:
        args (argparse.Namespace): Parsed arguments specifying which losses are active
            and their coefficients.
        config (ConfigDict): Configuration with loss hyperparameters (e.g., STFT settings,
            quantile `q`, coarse masking flag).

    Returns:
        Callable[[Any, Any, Optional[Any]], torch.Tensor]: A function `loss(y_pred, y_true, x=None)`
        that computes the weighted sum of the selected loss terms.
    """

    loss_fns = []

    if 'masked_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None:
            masked_loss(y_pred, y_true,
                        q=config['training']['q'],
                        coarse=config['training']['coarse_loss_clip'])
            * args.masked_loss_coef
        )

    if 'mse_loss' in args.loss:
        mse = nn.MSELoss()
        loss_fns.append(
            lambda y_pred, y_true, x=None: mse(y_pred, y_true) * args.mse_loss_coef
        )

    if 'l1_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None: F.l1_loss(y_pred, y_true) * args.l1_loss_coef
        )

    if 'multistft_loss' in args.loss:
        loss_options = dict(config.get('loss_multistft', {}))
        stft_loss = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)
        loss_fns.append(
            lambda y_pred, y_true, x=None: multistft_loss(y_pred, y_true, stft_loss)
                                           * args.multistft_loss_coef
        )

    if 'log_wmse_loss' in args.loss:
        log_wmse = LogWMSE(
            audio_length=int(getattr(config.audio, 'chunk_size', 485100))
                         // int(getattr(config.audio, 'sample_rate', 44100)),
            sample_rate=int(getattr(config.audio, 'sample_rate', 44100)),
            return_as_loss=True,
            bypass_filter=getattr(config.training, 'bypass_filter', False),
        )
        loss_fns.append(
            lambda y_pred, y_true, x: log_wmse(x, y_pred, y_true)
                                           * args.log_wmse_loss_coef
        )

    if 'l1_snr_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None: l1_snr_loss(y_pred, y_true) * args.l1_snr_loss_coef
        )

    if 'l1_snr_db_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None: l1_snr_db_loss(y_pred, y_true) * args.l1_snr_db_loss_coef
        )

    if 'stft_l1_snr_db_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None: stft_l1_snr_db_loss(y_pred, y_true) * args.stft_l1_snr_db_loss_coef
        )

    if 'multi_l1_snr_db_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x=None: multi_l1_snr_db_loss(y_pred, y_true) * args.multi_l1_snr_db_loss_coef
        )

    if 'spec_rmse_loss' in args.loss:
        stft_config = {
            'n_fft': getattr(config.model, 'nfft', 4096),
            'hop_length': getattr(config.model, 'hop_size', 1024),
            'win_length': getattr(config.model, 'win_size', 4096),
            'center': True,
            'normalized': getattr(config.model, 'normalized', True)
        }
        loss_fns.append(
            lambda y_pred, y_true, x=None: spec_rmse_loss(y_pred, y_true, stft_config) *
                                           args.spec_rmse_loss_coef)

    if 'spec_masked_loss' in args.loss:
        stft_config = {
            'n_fft': getattr(config.model, 'nfft', 4096),
            'hop_length': getattr(config.model, 'hop_size', 1024),
            'win_length': getattr(config.model, 'win_size', 4096),
            'center': True,
            'normalized': getattr(config.model, 'normalized', True)
        }
        loss_fns.append(
            lambda y_pred, y_true, x=None: spec_masked_loss(y_pred, y_true,
                                                            stft_config,
                                                            q=config['training']['q'],
                                                            coarse=config['training']['coarse_loss_clip'])
                                           * args.spec_masked_loss_coef
        )

    if 'fullness_penalty_loss' in args.loss:
        sr = int(getattr(config.audio, 'sample_rate', 44100))
        n_fft = getattr(config.model, 'nfft', 4096)
        hop_length = getattr(config.model, 'hop_size', 1024)

        fullness_loss_module = BleedFullPenaltyLoss(
            mode='fullness',
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        loss_fns.append(
            lambda y_pred, y_true, x=None: fullness_loss_module(y_pred, y_true) * args.fullness_penalty_loss_coef
        )

    if 'bleedless_penalty_loss' in args.loss:
        sr = int(getattr(config.audio, 'sample_rate', 44100))
        n_fft = getattr(config.model, 'nfft', 4096)
        hop_length = getattr(config.model, 'hop_size', 1024)

        bleedless_loss_module = BleedFullPenaltyLoss(
            mode='bleedless',
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        loss_fns.append(
            lambda y_pred, y_true, x=None: bleedless_loss_module(y_pred, y_true) * args.bleedless_penalty_loss_coef
        )

    def multi_loss(y_pred: Any, y_true: Any, x: Optional[Any] = None) -> torch.Tensor:
        total = 0
        for fn in loss_fns:
            total = total + fn(y_pred, y_true, x)
        return total

    return multi_loss
