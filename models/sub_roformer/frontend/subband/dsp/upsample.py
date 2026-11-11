from einops import rearrange
from torch import Tensor

from .convolve import fftconvolve
from .utils import fix_length


def polyphase_fftupsample(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Convolution with polylphase.

    b: batch_size
    l: audio_samples
    n: filter_len

    Args:
        x: (b, l_down)
        w: (n,)

    Returns:
        out: (b, l_up)
    """

    L = x.shape[-1] * stride
    N = w.shape[-1]
    x = rearrange(x, 'b l -> b 1 l')
    w = rearrange(w, '(n1 n2) -> n2 1 n1', n2=stride)  # (b, n2, n1)

    x = fftconvolve(x, w, use_complex_fft=True, mode="full")  # (b, k, l_down)
    x = rearrange(x, 'b t2 t1 -> b (t1 t2)')
    out = fix_length(x, N // 2, L)
    
    return out
