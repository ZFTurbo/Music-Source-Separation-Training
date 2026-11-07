import torch
from einops import rearrange
from torch import Tensor

from .utils import fix_length


def fftconvolve(x: Tensor, h: Tensor, use_complex_fft: bool, mode="same") -> Tensor:
    """Convolution with FFT.

    Args:
        x: (b, i, l), signal
        h: (o, i, l), signal

    Returns:
        out: (b, o, l)
    """

    if x.dtype in {torch.float16, torch.bfloat16}:
        x = x.float()
    if h.dtype in {torch.float16, torch.bfloat16}:
        h = h.float()
    L1 = x.shape[-1]
    L2 = h.shape[-1]
    L = L1 + L2 - 1  # full convolution length

    if use_complex_fft:
        fft = torch.fft.fft
        ifft = torch.fft.ifft
    else:
        fft = torch.fft.rfft
        ifft = torch.fft.irfft

    # pad both sequences to length L
    X = fft(torch.nn.functional.pad(x, (0, L - L1)))
    H = fft(torch.nn.functional.pad(h, (0, L - L2)))
    
    # multiply in frequency domain and IFFT
    Y = torch.einsum('bil,oil->bol', X, H)  # (b, o, l)
    y = ifft(Y, n=L)

    if mode == "full":
        return y

    elif mode == "same":
        start = (L2 - 1) // 2    
        end = start + L1
        return y[:, :, start:end]  # (b, o, l)

    else:
        raise ValueError(mode)


def polyphase_fftconvolve(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Strided convolution with polylphase decomposition and FFT.

    b: batch_size
    k: n_banks
    l: audio_samples
    n: filter_len

    Args:
        x: (b, l_up)
        w: (k, n)
        stride: int

    Returns:
        out: (b, k, l_down)
    """
    assert torch.all(w[:, 0] == 0)
    x = rearrange(x, 'b (t1 t2) -> b t2 t1', t2=stride)  # (b, t2, t1)
    w = rearrange(w, 'k (n1 n2) -> k n2 n1', n2=stride)  # (b, n2, n1)
    
    L = x.shape[-1]
    N = w.shape[-1]
    assert N % 2 == 0

    w = torch.flip(w, dims=[2])  # (b, n2, n1)
    x = fftconvolve(x, w, use_complex_fft=True, mode="full")  # (b, k, l_down)
    out = fix_length(x, N // 2 - 1, L)  # (b, k, l_down)

    return out
