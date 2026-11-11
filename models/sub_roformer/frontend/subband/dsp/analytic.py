import torch
from torch import Tensor


def real_to_analytic(x: Tensor) -> Tensor:
    r"""Convert real signal into analytic signal.

    Args:
        x: (any, l), real signal

    Returns:
        out: (any, l), complex signal
    """
    L = x.shape[-1]
    x = torch.fft.fft(x)
    x[..., 1 : L // 2] *= 2
    x[..., L // 2 + 1 :] = 0
    return torch.fft.ifft(x)


def analytic_to_real(x: Tensor) -> Tensor:
    r"""Convert analytic singal into real signal.

    Args:
        x: (any,), complex signal

    Returns:
        out: (any,), real signal
    """
    return x.real
