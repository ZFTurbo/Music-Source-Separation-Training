import math

import torch
from torch import Tensor


def fix_length(x: Tensor, start: int, length: int) -> Tensor:
    end = start + length
    return x[..., start : end]


def shift_frequency(x: Tensor, f: Tensor, sr: int, n=1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)
    """
    if f.dtype in {torch.float16, torch.bfloat16}:
        f = f.float()
    omega = f / (sr / 2) * math.pi  # (k,)
    t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
    a = torch.exp(1.j * omega[:, None] * t[None, :])
    out = x * a
    return out
