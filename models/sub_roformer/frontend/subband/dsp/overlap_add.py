from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def overlap_add(x: Tensor, hop_length: int, window: Tensor | None) -> Tensor:
    r"""Overlap-add.

    b: batch_size
    t: n_frames
    n: frame_samples
    l: audio_samples

    Args:
        x: (b, t, n)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length
    
    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        ).squeeze(dim=(0, 1, 2))  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


if __name__ == "__main__":
    r"""
    b: batch_size
    t: n_frames
    n: window_size
    l: audio_samples
    """

    T = 100  # n_frames
    N = 2048  # window_size
    hop_length = 480

    x = torch.ones((4, T, N))  # (b, t, n)
    out = overlap_add(x, hop_length, torch.hann_window(N))  # (b, l)
    print(out.shape)
