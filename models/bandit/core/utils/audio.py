from collections import defaultdict

from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.jit.script
def merge(
        combined: torch.Tensor,
        original_batch_size: int,
        n_channel: int,
        n_chunks: int,
        chunk_size: int, ):
    combined = torch.reshape(
            combined,
            (original_batch_size, n_chunks, n_channel, chunk_size)
    )
    combined = torch.permute(combined, (0, 2, 3, 1)).reshape(
            original_batch_size * n_channel,
            chunk_size,
            n_chunks
    )

    return combined


@torch.jit.script
def unfold(
        padded_audio: torch.Tensor,
        original_batch_size: int,
        n_channel: int,
        chunk_size: int,
        hop_size: int
        ) -> torch.Tensor:

    unfolded_input = F.unfold(
            padded_audio[:, :, None, :],
            kernel_size=(1, chunk_size),
            stride=(1, hop_size)
    )

    _, _, n_chunks = unfolded_input.shape
    unfolded_input = unfolded_input.view(
            original_batch_size,
            n_channel,
            chunk_size,
            n_chunks
    )
    unfolded_input = torch.permute(
            unfolded_input,
            (0, 3, 1, 2)
    ).reshape(
            original_batch_size * n_chunks,
            n_channel,
            chunk_size
    )

    return unfolded_input


@torch.jit.script
# @torch.compile
def merge_chunks_all(
        combined: torch.Tensor,
        original_batch_size: int,
        n_channel: int,
        n_samples: int,
        n_padded_samples: int,
        n_chunks: int,
        chunk_size: int,
        hop_size: int,
        edge_frame_pad_sizes: Tuple[int, int],
        standard_window: torch.Tensor,
        first_window: torch.Tensor,
        last_window: torch.Tensor
):
    combined = merge(
        combined,
        original_batch_size,
        n_channel,
        n_chunks,
        chunk_size
        )

    combined = combined * standard_window[:, None].to(combined.device)

    combined = F.fold(
            combined.to(torch.float32), output_size=(1, n_padded_samples),
            kernel_size=(1, chunk_size),
            stride=(1, hop_size)
    )

    combined = combined.view(
            original_batch_size,
            n_channel,
            n_padded_samples
    )

    pad_front, pad_back = edge_frame_pad_sizes
    combined = combined[..., pad_front:-pad_back]

    combined = combined[..., :n_samples]

    return combined

    # @torch.jit.script


def merge_chunks_edge(
        combined: torch.Tensor,
        original_batch_size: int,
        n_channel: int,
        n_samples: int,
        n_padded_samples: int,
        n_chunks: int,
        chunk_size: int,
        hop_size: int,
        edge_frame_pad_sizes: Tuple[int, int],
        standard_window: torch.Tensor,
        first_window: torch.Tensor,
        last_window: torch.Tensor
):
    combined = merge(
        combined,
        original_batch_size,
        n_channel,
        n_chunks,
        chunk_size
        )

    combined[..., 0] = combined[..., 0] * first_window
    combined[..., -1] = combined[..., -1] * last_window
    combined[..., 1:-1] = combined[...,
                          1:-1] * standard_window[:, None]

    combined = F.fold(
            combined, output_size=(1, n_padded_samples),
            kernel_size=(1, chunk_size),
            stride=(1, hop_size)
    )

    combined = combined.view(
            original_batch_size,
            n_channel,
            n_padded_samples
    )

    combined = combined[..., :n_samples]

    return combined


class BaseFader(nn.Module):
    def __init__(
            self,
            chunk_size_second: float,
            hop_size_second: float,
            fs: int,
            fade_edge_frames: bool,
            batch_size: int,
    ) -> None:
        super().__init__()

        self.chunk_size = int(chunk_size_second * fs)
        self.hop_size = int(hop_size_second * fs)
        self.overlap_size = self.chunk_size - self.hop_size
        self.fade_edge_frames = fade_edge_frames
        self.batch_size = batch_size

    # @torch.jit.script
    def prepare(self, audio):

        if self.fade_edge_frames:
            audio = F.pad(audio, self.edge_frame_pad_sizes, mode="reflect")

        n_samples = audio.shape[-1]
        n_chunks = int(
                np.ceil((n_samples - self.chunk_size) / self.hop_size) + 1
        )

        padded_size = (n_chunks - 1) * self.hop_size + self.chunk_size
        pad_size = padded_size - n_samples

        padded_audio = F.pad(audio, (0, pad_size))

        return padded_audio, n_chunks

    def forward(
            self,
            audio: torch.Tensor,
            model_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
    ):

        original_dtype = audio.dtype
        original_device = audio.device

        audio = audio.to("cpu")

        original_batch_size, n_channel, n_samples = audio.shape
        padded_audio, n_chunks = self.prepare(audio)
        del audio
        n_padded_samples = padded_audio.shape[-1]

        if n_channel > 1:
            padded_audio = padded_audio.view(
                    original_batch_size * n_channel, 1, n_padded_samples
            )

        unfolded_input = unfold(
                padded_audio,
                original_batch_size,
                n_channel,
                self.chunk_size, self.hop_size
        )

        n_total_chunks, n_channel, chunk_size = unfolded_input.shape

        n_batch = np.ceil(n_total_chunks / self.batch_size).astype(int)

        chunks_in = [
                unfolded_input[
                b * self.batch_size:(b + 1) * self.batch_size, ...].clone()
                for b in range(n_batch)
        ]

        all_chunks_out = defaultdict(
                lambda: torch.zeros_like(
                        unfolded_input, device="cpu"
                )
        )

        # for b, cin in enumerate(tqdm(chunks_in)):
        for b, cin in enumerate(chunks_in):
            if torch.allclose(cin, torch.tensor(0.0)):
                del cin
                continue

            chunks_out = model_fn(cin.to(original_device))
            del cin
            for s, c in chunks_out.items():
                all_chunks_out[s][b * self.batch_size:(b + 1) * self.batch_size,
                ...] = c.cpu()
            del chunks_out

        del unfolded_input
        del padded_audio

        if self.fade_edge_frames:
            fn = merge_chunks_all
        else:
            fn = merge_chunks_edge
        outputs = {}

        torch.cuda.empty_cache()

        for s, c in all_chunks_out.items():
            combined: torch.Tensor = fn(
                    c,
                    original_batch_size,
                    n_channel,
                    n_samples,
                    n_padded_samples,
                    n_chunks,
                    self.chunk_size,
                    self.hop_size,
                    self.edge_frame_pad_sizes,
                    self.standard_window,
                    self.__dict__.get("first_window", self.standard_window),
                    self.__dict__.get("last_window", self.standard_window)
            )

            outputs[s] = combined.to(
                dtype=original_dtype,
                device=original_device
                )

        return {
                "audio": outputs
        }
    #
    # def old_forward(
    #         self,
    #         audio: torch.Tensor,
    #         model_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
    # ):
    #
    #     n_samples = audio.shape[-1]
    #     original_batch_size = audio.shape[0]
    #
    #     padded_audio, n_chunks = self.prepare(audio)
    #
    #     ndim = padded_audio.ndim
    #     broadcaster = [1 for _ in range(ndim - 1)] + [self.chunk_size]
    #
    #     outputs = defaultdict(
    #             lambda: torch.zeros_like(
    #                     padded_audio, device=audio.device, dtype=torch.float64
    #             )
    #     )
    #
    #     all_chunks_out = []
    #     len_chunks_in = []
    #
    #     batch_size_ = int(self.batch_size // original_batch_size)
    #     for b in range(int(np.ceil(n_chunks / batch_size_))):
    #         chunks_in = []
    #         for j in range(batch_size_):
    #             i = b * batch_size_ + j
    #             if i == n_chunks:
    #                 break
    #
    #             start = i * hop_size
    #             end = start + self.chunk_size
    #             chunk_in = padded_audio[..., start:end]
    #             chunks_in.append(chunk_in)
    #
    #         chunks_in = torch.concat(chunks_in, dim=0)
    #         chunks_out = model_fn(chunks_in)
    #         all_chunks_out.append(chunks_out)
    #         len_chunks_in.append(len(chunks_in))
    #
    #     for b, (chunks_out, lci) in enumerate(
    #             zip(all_chunks_out, len_chunks_in)
    #     ):
    #         for stem in chunks_out:
    #             for j in range(lci // original_batch_size):
    #                 i = b * batch_size_ + j
    #
    #                 if self.fade_edge_frames:
    #                     window = self.standard_window
    #                 else:
    #                     if i == 0:
    #                         window = self.first_window
    #                     elif i == n_chunks - 1:
    #                         window = self.last_window
    #                     else:
    #                         window = self.standard_window
    #
    #                 start = i * hop_size
    #                 end = start + self.chunk_size
    #
    #                 chunk_out = chunks_out[stem][j * original_batch_size: (j + 1) * original_batch_size,
    #                             ...]
    #                 contrib = window.view(*broadcaster) * chunk_out
    #                 outputs[stem][..., start:end] = (
    #                         outputs[stem][..., start:end] + contrib
    #                 )
    #
    #     if self.fade_edge_frames:
    #         pad_front, pad_back = self.edge_frame_pad_sizes
    #         outputs = {k: v[..., pad_front:-pad_back] for k, v in
    #                    outputs.items()}
    #
    #     outputs = {k: v[..., :n_samples].to(audio.dtype) for k, v in
    #                outputs.items()}
    #
    #     return {
    #             "audio": outputs
    #     }


class LinearFader(BaseFader):
    def __init__(
            self,
            chunk_size_second: float,
            hop_size_second: float,
            fs: int,
            fade_edge_frames: bool = False,
            batch_size: int = 1,
    ) -> None:

        assert hop_size_second >= chunk_size_second / 2

        super().__init__(
                chunk_size_second=chunk_size_second,
                hop_size_second=hop_size_second,
                fs=fs,
                fade_edge_frames=fade_edge_frames,
                batch_size=batch_size,
        )

        in_fade = torch.linspace(0.0, 1.0, self.overlap_size + 1)[:-1]
        out_fade = torch.linspace(1.0, 0.0, self.overlap_size + 1)[1:]
        center_ones = torch.ones(self.chunk_size - 2 * self.overlap_size)
        inout_ones = torch.ones(self.overlap_size)

        # using nn.Parameters allows lightning to take care of devices for us
        self.register_buffer(
                "standard_window",
                torch.concat([in_fade, center_ones, out_fade])
        )

        self.fade_edge_frames = fade_edge_frames
        self.edge_frame_pad_size = (self.overlap_size, self.overlap_size)

        if not self.fade_edge_frames:
            self.first_window = nn.Parameter(
                    torch.concat([inout_ones, center_ones, out_fade]),
                    requires_grad=False
            )
            self.last_window = nn.Parameter(
                    torch.concat([in_fade, center_ones, inout_ones]),
                    requires_grad=False
            )


class OverlapAddFader(BaseFader):
    def __init__(
            self,
            window_type: str,
            chunk_size_second: float,
            hop_size_second: float,
            fs: int,
            batch_size: int = 1,
    ) -> None:
        assert (chunk_size_second / hop_size_second) % 2 == 0
        assert int(chunk_size_second * fs) % 2 == 0

        super().__init__(
            chunk_size_second=chunk_size_second,
            hop_size_second=hop_size_second,
            fs=fs,
            fade_edge_frames=True,
            batch_size=batch_size,
        )

        self.hop_multiplier = self.chunk_size / (2 * self.hop_size)
        # print(f"hop multiplier: {self.hop_multiplier}")

        self.edge_frame_pad_sizes = (
            2 * self.overlap_size,
            2 * self.overlap_size
        )

        self.register_buffer(
            "standard_window", torch.windows.__dict__[window_type](
                    self.chunk_size, sym=False,  # dtype=torch.float64
            ) / self.hop_multiplier
        )


if __name__ == "__main__":
    import torchaudio as ta
    fs = 44100
    ola = OverlapAddFader(
        "hann",
        6.0,
        1.0,
        fs,
        batch_size=16
    )
    audio_, _ = ta.load(
            "$DATA_ROOT/MUSDB18/HQ/canonical/test/BKS - Too "
            "Much/vocals.wav"
    )
    audio_ = audio_[None, ...]
    out = ola(audio_, lambda x: {"stem": x})["audio"]["stem"]
    print(torch.allclose(out, audio_))
