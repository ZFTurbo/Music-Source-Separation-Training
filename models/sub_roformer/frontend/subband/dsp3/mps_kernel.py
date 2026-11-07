"""Lazy Metal implementation of fixed-point frequency modulation."""

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


def is_available() -> bool:
    """Return whether this runtime supports compiling MPS shaders."""
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


@lru_cache(maxsize=1)
def _load_library():
    source = Path(__file__).parent / "csrc" / "shift_frequency_mps.metal"
    return torch.mps.compile_shader(source.read_text())


def shift_frequency_mps(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False,
    *, sample_rate: int | None = None,
) -> Tensor:
    """Modulate row-contiguous complex64 MPS input from steps or float32 Hz."""
    if not is_available():
        raise RuntimeError("The DSP3 MPS kernel requires torch.mps.compile_shader and MPS.")
    if x.device.type != "mps" or x.dtype != torch.complex64:
        raise ValueError("Expected a complex64 MPS input.")
    if x.ndim < 2 or not (x.is_contiguous() or (x.ndim <= 3 and x.stride(-1) == 1)):
        raise ValueError("Expected a contiguous input with at least two dimensions.")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Expected a positive integer sample rate.")
    if (
        phase_steps.device != x.device
        or phase_steps.dtype != (torch.int64 if sample_rate is None else torch.float32)
        or phase_steps.ndim != 1
        or not phase_steps.is_contiguous()
        or phase_steps.numel() != x.shape[-2]
    ):
        raise ValueError("Expected contiguous phase steps (int64) or frequencies (float32) matching the input bands and device.")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return result
    library = _load_library()
    if sample_rate is not None:
        steps = torch.empty_like(phase_steps, dtype=torch.int64)
        library.quantize_frequencies(phase_steps, steps, sample_rate, threads=phase_steps.numel())
        phase_steps = steps
    library.shift_frequency(
        torch.view_as_real(x), phase_steps, torch.view_as_real(result),
        x.shape[-1], x.shape[-2], x.stride(-2),
        x.stride(-3) if x.ndim == 3 else x.shape[-2] * x.shape[-1],
        n % (1 << 32), int(conjugate),
        threads=(x.shape[-1], x.shape[-2], x.numel() // (x.shape[-1] * x.shape[-2])),
        group_size=(256, 1, 1),
    )
    return result
