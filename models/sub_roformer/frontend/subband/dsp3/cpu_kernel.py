"""Lazy native CPU implementation of fixed-point frequency modulation."""

from functools import lru_cache
from pathlib import Path
import platform
import sys

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _load_extension():
    from torch.utils.cpp_extension import load

    flags = ["-O3", "-ffp-contract=off"]
    if sys.platform == "linux" and torch.backends.openmp.is_available():
        flags.append("-fopenmp")
    if platform.machine() in ("x86_64", "AMD64") and torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512"):
        flags.extend(["-mavx2", "-mfma", "-DCPU_CAPABILITY_AVX2", "-DCPU_CAPABILITY=AVX2"])
    try:
        return load(
            name="subroformer_dsp3_native_cpu",
            sources=[str(Path(__file__).parent / "csrc" / "shift_frequency_cpu.cpp")],
            extra_cflags=flags,
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "The DSP3 CPU kernel requires a working C++ compiler and Ninja. "
            "Select the reference backend if native compilation is unavailable."
        ) from error


@lru_cache(maxsize=1)
def is_available() -> bool:
    """Check whether the native extension can be loaded, compiling if needed."""
    try:
        _load_extension()
    except RuntimeError:
        return False
    return True


def shift_frequency_cpu(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False
) -> Tensor:
    """Modulate contiguous or row-contiguous CPU input from signed phase steps."""
    import torch

    _load_extension()
    return torch.ops.subroformer_dsp3_native.shift_frequency_cpu(
        x, phase_steps, n % (1 << 32), conjugate
    )
