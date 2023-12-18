from typing import Any, Callable

import numpy as np
import torch
import torchmetrics as tm
from torch._C import _LinAlgError
from torchmetrics import functional as tmF


class SafeSignalDistortionRatio(tm.SignalDistortionRatio):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def update(self, *args, **kwargs) -> Any:
        try:
            super().update(*args, **kwargs)
        except:
            pass

    def compute(self) -> Any:
        if self.total == 0:
            return torch.tensor(torch.nan)
        return super().compute()


class BaseChunkMedianSignalRatio(tm.Metric):
    def __init__(
            self,
            func: Callable,
            window_size: int,
            hop_size: int = None,
            zero_mean: bool = False,
    ) -> None:
        super().__init__()

        # self.zero_mean = zero_mean
        self.func = func
        self.window_size = window_size
        if hop_size is None:
            hop_size = window_size
        self.hop_size = hop_size

        self.add_state(
            "sum_snr",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
            )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        n_samples = target.shape[-1]

        n_chunks = int(
            np.ceil((n_samples - self.window_size) / self.hop_size) + 1
            )

        snr_chunk = []

        for i in range(n_chunks):
            start = i * self.hop_size

            if n_samples - start < self.window_size:
                continue

            end = start + self.window_size

            try:
                chunk_snr = self.func(
                        preds[..., start:end],
                        target[..., start:end]
                        )

                # print(preds.shape, chunk_snr.shape)

                if torch.all(torch.isfinite(chunk_snr)):
                    snr_chunk.append(chunk_snr)
            except _LinAlgError:
                pass

        snr_chunk = torch.stack(snr_chunk, dim=-1)
        snr_batch, _ = torch.nanmedian(snr_chunk, dim=-1)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Any:
        return self.sum_snr / self.total


class ChunkMedianSignalNoiseRatio(BaseChunkMedianSignalRatio):
    def __init__(
            self,
            window_size: int,
            hop_size: int = None,
            zero_mean: bool = False
    ) -> None:
        super().__init__(
                func=tmF.signal_noise_ratio,
                window_size=window_size,
                hop_size=hop_size,
                zero_mean=zero_mean,
        )


class ChunkMedianScaleInvariantSignalNoiseRatio(BaseChunkMedianSignalRatio):
    def __init__(
            self,
            window_size: int,
            hop_size: int = None,
            zero_mean: bool = False
    ) -> None:
        super().__init__(
                func=tmF.scale_invariant_signal_noise_ratio,
                window_size=window_size,
                hop_size=hop_size,
                zero_mean=zero_mean,
        )


class ChunkMedianSignalDistortionRatio(BaseChunkMedianSignalRatio):
    def __init__(
            self,
            window_size: int,
            hop_size: int = None,
            zero_mean: bool = False
    ) -> None:
        super().__init__(
                func=tmF.signal_distortion_ratio,
                window_size=window_size,
                hop_size=hop_size,
                zero_mean=zero_mean,
        )


class ChunkMedianScaleInvariantSignalDistortionRatio(
        BaseChunkMedianSignalRatio
        ):
    def __init__(
            self,
            window_size: int,
            hop_size: int = None,
            zero_mean: bool = False
    ) -> None:
        super().__init__(
                func=tmF.scale_invariant_signal_distortion_ratio,
                window_size=window_size,
                hop_size=hop_size,
                zero_mean=zero_mean,
        )
