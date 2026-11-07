import math
from functools import partial

import torch
from torch import Tensor


_PHASE_CHUNK_ELEMENTS = 1 << 20


def fix_length(x: Tensor, start: int, length: int) -> Tensor:
    end = start + length
    return x[..., start : end]


def _phase_steps(x: Tensor, f: Tensor, sr: int) -> Tensor:
    phase_scale = 1 << 32
    phase_half = phase_scale >> 1
    phase_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    frequencies = f.to(device=x.device, dtype=phase_dtype)
    scaled_frequencies = frequencies.abs().mul(phase_scale).round().to(torch.int64)
    steps = torch.div(scaled_frequencies + sr // 2, sr, rounding_mode="floor")
    steps.mul_(frequencies.sign().to(torch.int64))
    return steps.add_(phase_half).bitwise_and_(phase_scale - 1).sub_(phase_half)


def _shift_frequency_reference(x: Tensor, f: Tensor, sr: int, n: int = 1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)

    Frequencies are fixed, non-learnable values quantized to sr / 2**32 Hz.
    Integer phase wraps before conversion to radians, so float32 phase
    evaluation stays within [-pi, pi) regardless of signal length. Phase
    workspace is tiled; float32/complex64 inputs require no float64 tensors.
    """
    phase_scale = 1 << 32
    phase_half = phase_scale >> 1
    phase_mask = phase_scale - 1
    phase_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    oscillator_dtype = torch.complex128 if phase_dtype == torch.float64 else torch.complex64
    phase_steps = _phase_steps(x, f, sr)

    length = x.shape[-1]
    oscillator = torch.empty((f.numel(), length), dtype=oscillator_dtype, device=x.device)
    chunk_length = max(1, _PHASE_CHUNK_ELEMENTS // max(1, f.numel()))
    for start in range(0, length, chunk_length):
        end = min(start + chunk_length, length)
        sample_indices = torch.arange(start, end, dtype=torch.int64, device=x.device)
        sample_indices.mul_(n).bitwise_and_(phase_mask)
        phase_words = phase_steps[:, None] * sample_indices[None, :]
        phase_words.add_(phase_half).bitwise_and_(phase_mask).sub_(phase_half)
        phase = phase_words.to(phase_dtype).mul_(2 * math.pi / phase_scale)
        torch.polar(torch.ones_like(phase), phase, out=oscillator[:, start:end])
        del sample_indices, phase_words, phase

    return x * oscillator


class _ShiftFrequency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, steps, stride, kernel, conjugate):
        ctx.save_for_backward(steps)
        ctx.stride = stride
        ctx.kernel = kernel
        ctx.conjugate = conjugate
        signal = signal.resolve_conj().resolve_neg()
        if not signal.is_contiguous() and not (signal.ndim in (2, 3) and signal.stride(-1) == 1):
            signal = signal.contiguous()
        return kernel(signal, steps, stride, conjugate)

    @staticmethod
    def backward(ctx, gradient):
        steps, = ctx.saved_tensors
        result = _ShiftFrequency.apply(
            gradient, steps, ctx.stride, ctx.kernel, not ctx.conjugate,
        )
        return result, None, None, None, None


def shift_frequency(x: Tensor, f: Tensor, sr: int, n: int = 1) -> Tensor:
    """Fixed-point modulation with fused CPU/CUDA/MPS kernels when available.

    Frequencies are non-learnable. Other devices, real inputs, and broadcast
    band shapes retain the original PyTorch implementation.
    """
    if x.dtype not in (torch.complex64, torch.complex128) or x.ndim < 2 or f.ndim != 1 or x.shape[-2] != f.numel():
        return _shift_frequency_reference(x, f, sr, n)
    if x.device.type == "cpu":
        from .cpu_kernel import is_available, shift_frequency_cpu

        kernel = shift_frequency_cpu
    elif x.device.type == "cuda":
        from .cuda_kernel import is_available, shift_frequency_cuda

        kernel = partial(shift_frequency_cuda, sample_rate=sr)
    elif x.device.type == "mps" and x.dtype == torch.complex64:
        from .mps_kernel import is_available, shift_frequency_mps

        kernel = partial(shift_frequency_mps, sample_rate=sr)
    else:
        return _shift_frequency_reference(x, f, sr, n)
    if not is_available():
        return _shift_frequency_reference(x, f, sr, n)
    if x.device.type == "cuda":
        values = f.to(device=x.device).contiguous()
    elif x.device.type == "mps":
        values = f.to(device=x.device, dtype=torch.float32).contiguous()
    else:
        values = _phase_steps(x, f, sr)
    return _ShiftFrequency.apply(x, values, n % (1 << 32), kernel, False)
