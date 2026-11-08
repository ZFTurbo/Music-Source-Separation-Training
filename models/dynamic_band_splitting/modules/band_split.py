"""Frequency-region configuration and static band partitions."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def region_configs(
    edges: Sequence[int],
    max_chunks: Sequence[int],
    target_ratios: Sequence[float] | None = None,
    **extra: Any,
) -> list[dict[str, Any]]:
    """Build DBS coarse-region configs from bin edges ``[0, e1, ..., F]``.

    ``target_ratios`` defaults to ``width / max_chunks`` so that the hard
    Top-K selects exactly ``max_chunks`` tokens per region. Any extra keyword
    is copied into every region (e.g. ``use_energy=True`` for TIGER).
    """
    edges = [int(e) for e in edges]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"edges must be strictly increasing, got {edges}")
    if len(max_chunks) != len(edges) - 1:
        raise ValueError("max_chunks needs one entry per region")
    configs = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        chunks = int(max_chunks[index])
        ratio = (
            float(target_ratios[index])
            if target_ratios is not None
            else (end - start) / float(chunks)
        )
        configs.append(dict(start=start, end=end, target_ratio=ratio, max_chunks=chunks, **extra))
    return configs


def desired_token_count(width: int, target_ratio: float, max_chunks: int) -> int:
    """Number of tokens the hard Top-K keeps in one region."""
    return max(1, min(int(max_chunks), int(round(width / max(float(target_ratio), 1.01)))))


def proportional_band_widths(n_fft: int, num_bands: int) -> tuple[int, ...]:
    """Equal-fraction static partition of the ``n_fft // 2 + 1`` STFT bins."""
    n_fft, num_bands = int(n_fft), int(num_bands)
    freqs = n_fft // 2 + 1
    if not 2 <= num_bands <= freqs:
        raise ValueError(f"num_bands must be within [2, {freqs}], got {num_bands}")
    edges = [0]
    for index in range(1, num_bands):
        raw = int(round(index / num_bands * freqs))
        edges.append(max(edges[-1] + 1, min(raw, freqs - (num_bands - index))))
    edges.append(freqs)
    return tuple(b - a for a, b in zip(edges[:-1], edges[1:]))


def validate_region_configs(configs: Sequence[Mapping[str, Any]], num_freqs: int) -> None:
    """Regions must tile ``[0, num_freqs)`` contiguously."""
    cursor = 0
    for cfg in configs:
        start, end = int(cfg["start"]), int(cfg["end"])
        if start != cursor or end <= start:
            raise ValueError(f"regions must tile [0, {num_freqs}) contiguously: {list(configs)}")
        cursor = end
    if cursor != num_freqs:
        raise ValueError(f"regions end at {cursor}, expected {num_freqs}")
