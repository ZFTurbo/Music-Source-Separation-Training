"""Fused CUDA implementation of fixed-point frequency modulation."""

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None


def is_available() -> bool:
    """Return whether the optional Triton implementation can be imported."""
    return triton is not None


if triton is not None:

    @triton.jit
    def _shift_frequency_kernel(
        source,
        steps,
        target,
        elements: tl.constexpr,
        length: tl.constexpr,
        bands: tl.constexpr,
        band_stride: tl.constexpr,
        batch_stride: tl.constexpr,
        stride: tl.constexpr,
        double: tl.constexpr,
        conjugate: tl.constexpr,
        sample_rate: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        valid = offsets < elements
        band_indices = (offsets // length) % bands
        sample_indices = (offsets % length).to(tl.uint32)
        values = tl.load(steps + band_indices, valid, other=0)
        if sample_rate is not None:
            if double:
                frequencies = values.to(tl.float64)
            else:
                frequencies = values.to(tl.float32)
            scaled = libdevice.nearbyint(tl.abs(frequencies) * 4294967296.0).to(tl.int64)
            magnitude = (scaled + sample_rate // 2) // sample_rate
            phase_steps = tl.where(frequencies < 0, -magnitude, magnitude).to(tl.uint32)
        else:
            phase_steps = values.to(tl.uint32)
        phase_words = (phase_steps * sample_indices * stride).to(tl.int32)
        # 2 * pi / 2**32 as a literal: torch.compile (inductor) re-generates user triton kernels
        # without the module globals, so `math.pi` here fails with "NameError: math is not defined"
        if double:
            phase = phase_words.to(tl.float64) * tl.full((), 1.4629180792671596e-09, tl.float64)
        else:
            phase = phase_words.to(tl.float32) * 1.4629180792671596e-09
        if double:
            real_oscillator = libdevice.cos(phase)
            imag_oscillator = libdevice.sin(phase)
        else:
            real_oscillator = tl.cos(phase)
            imag_oscillator = tl.sin(phase)
        if conjugate:
            imag_oscillator = -imag_oscillator
        source_offsets = (offsets // (length * bands)) * batch_stride + band_indices * band_stride + offsets % length
        real_source = tl.load(source + 2 * source_offsets, valid, other=0)
        imag_source = tl.load(source + 2 * source_offsets + 1, valid, other=0)
        real_target = real_source * real_oscillator - imag_source * imag_oscillator
        imag_target = real_source * imag_oscillator + imag_source * real_oscillator
        tl.store(target + 2 * offsets, real_target, valid)
        tl.store(target + 2 * offsets + 1, imag_target, valid)


def shift_frequency_cuda(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False,
    *, sample_rate: int | None = None,
) -> Tensor:
    """Modulate complex CUDA input from phase steps or frequencies in Hz.

    With sample_rate set, quantization is fused into the modulation kernel.
    """
    if not is_available():
        raise RuntimeError("The DSP3 CUDA kernel requires Triton.")
    if x.device.type != "cuda" or x.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Expected a complex64 or complex128 CUDA input.")
    if x.ndim < 2 or not (x.is_contiguous() or (x.ndim <= 3 and x.stride(-1) == 1)):
        raise ValueError("Expected a contiguous input with at least two dimensions.")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Expected a positive integer sample rate.")
    if (
        phase_steps.device != x.device
        or (sample_rate is None and phase_steps.dtype != torch.int64)
        or phase_steps.ndim != 1
        or not phase_steps.is_contiguous()
        or phase_steps.numel() != x.shape[-2]
    ):
        raise ValueError("Expected contiguous phase steps (int64) or frequencies matching the input bands and device.")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return result
    block = 256
    with torch.cuda.device(x.device):
        _shift_frequency_kernel[(triton.cdiv(x.numel(), block),)](
            torch.view_as_real(x),
            phase_steps,
            torch.view_as_real(result),
            x.numel(),
            x.shape[-1],
            x.shape[-2],
            x.stride(-2),
            x.stride(-3) if x.ndim == 3 else x.shape[-2] * x.shape[-1],
            n & ((1 << 32) - 1),
            x.dtype == torch.complex128,
            conjugate,
            sample_rate,
            block,
            enable_fp_fusion=False,
        )
    return result
