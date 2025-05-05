import argparse
from typing import Any, Optional, Callable

import auraloss
import torch.nn.functional as F
import torch
from ml_collections import ConfigDict
from torch import nn
from torch_log_wmse import LogWMSE


def multistft_loss(y_: torch.Tensor, y: torch.Tensor,
                   loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
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


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = loss.mean(dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    quantile = torch.quantile(loss.detach(), q, interpolation='linear', dim=1, keepdim=True)
    mask = loss < quantile
    return (loss * mask).mean()


def spec_rmse_loss(estimate, sources, stft_config):
    _, _, _, lenc = estimate.shape
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
    loss = loss.mean(dims).sqrt().mean(dim=(0, 1))

    return loss


def spec_masked_loss(estimate, sources, stft_config, q: float = 0.9, coarse: bool = True):
    _, _, _, lenc = estimate.shape
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


def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Callable[..., torch.Tensor]:
    """
    Select and return the appropriate loss function based on the configuration and arguments.

    Args:
        args: Parsed command-line arguments containing flags for different loss functions.
        config: Configuration object containing loss settings and parameters.

    Returns:
        A loss function that can be applied to the predicted and ground truth tensors.
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

    def multi_loss(y_pred: Any, y_true: Any, x: Optional[Any] = None) -> torch.Tensor:
        total = 0
        for fn in loss_fns:
            total = total + fn(y_pred, y_true, x)
        return total

    return multi_loss