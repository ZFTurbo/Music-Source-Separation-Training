import glob
import os

import numpy as np
import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F
from tqdm.contrib.concurrent import process_map

from core.data._types import DataDict
from core.data.musdb.dataset import MUSDB18FullTrackDataset
import pyloudnorm as pyln

class SourceActivityDetector(nn.Module):
    def __init__(
            self,
            analysis_stem: str,
            output_path: str,
            fs: int = 44100,
            segment_length_second: float = 6.0,
            hop_length_second: float = 3.0,
            n_chunks: int = 10,
            chunk_epsilon: float = 1e-5,
            energy_threshold_quantile: float = 0.15,
            segment_epsilon: float = 1e-3,
            salient_proportion_threshold: float = 0.5,
            target_lufs: float = -24
    ) -> None:
        super().__init__()

        self.fs = fs
        self.segment_length = int(segment_length_second * self.fs)
        self.hop_length = int(hop_length_second * self.fs)
        self.n_chunks = n_chunks
        assert self.segment_length % self.n_chunks == 0
        self.chunk_size = self.segment_length // self.n_chunks
        self.chunk_epsilon = chunk_epsilon
        self.energy_threshold_quantile = energy_threshold_quantile
        self.segment_epsilon = segment_epsilon
        self.salient_proportion_threshold = salient_proportion_threshold
        self.analysis_stem = analysis_stem

        self.meter = pyln.Meter(self.fs)
        self.target_lufs = target_lufs

        self.output_path = output_path

    def forward(self, data: DataDict) -> None:

        stem_ = self.analysis_stem if (
                    self.analysis_stem != "none") else "mixture"

        x = data["audio"][stem_]

        xnp = x.numpy()
        loudness = self.meter.integrated_loudness(xnp.T)

        for stem in data["audio"]:
            s = data["audio"][stem]
            s = pyln.normalize.loudness(s.numpy().T, loudness, self.target_lufs).T
            s = torch.as_tensor(s)
            data["audio"][stem] = s

        if x.ndim == 3:
            assert x.shape[0] == 1
            x = x[0]

        n_chan, n_samples = x.shape

        n_segments = (
                int(
                    np.ceil((n_samples - self.segment_length) / self.hop_length)
                    ) + 1
        )

        segments = torch.zeros((n_segments, n_chan, self.segment_length))
        for i in range(n_segments):
            start = i * self.hop_length
            end = start + self.segment_length
            end = min(end, n_samples)

            xseg = x[:, start:end]

            if end - start < self.segment_length:
                xseg = F.pad(
                        xseg,
                        pad=(0, self.segment_length - (end - start)),
                        value=torch.nan
                )

            segments[i, :, :] = xseg

        chunks = segments.reshape(
                (n_segments, n_chan, self.n_chunks, self.chunk_size)
                )

        if self.analysis_stem != "none":
            chunk_energies = torch.mean(torch.square(chunks), dim=(1, 3))
            chunk_energies = torch.nan_to_num(chunk_energies, nan=0)
            chunk_energies[chunk_energies == 0] = self.chunk_epsilon

            energy_threshold = torch.nanquantile(
                    chunk_energies, q=self.energy_threshold_quantile
            )

            if energy_threshold < self.segment_epsilon:
                energy_threshold = self.segment_epsilon  # type: ignore[assignment]

            chunks_above_threshold = chunk_energies > energy_threshold
            n_chunks_above_threshold = torch.mean(
                    chunks_above_threshold.to(torch.float), dim=-1
            )

            segment_above_threshold = (
                    n_chunks_above_threshold > self.salient_proportion_threshold
            )

            if torch.sum(segment_above_threshold) == 0:
                return

        else:
            segment_above_threshold = torch.ones((n_segments,))

        for i in range(n_segments):
            if not segment_above_threshold[i]:
                continue

            outpath = os.path.join(
                    self.output_path,
                    self.analysis_stem,
                    f"{data['track']} - {self.analysis_stem}{i:03d}",
            )
            os.makedirs(outpath, exist_ok=True)

            for stem in data["audio"]:
                if stem == self.analysis_stem:
                    segment = torch.nan_to_num(segments[i, :, :], nan=0)
                else:
                    start = i * self.hop_length
                    end = start + self.segment_length
                    end = min(n_samples, end)

                    segment = data["audio"][stem][:, start:end]

                    if end - start < self.segment_length:
                        segment = F.pad(
                                segment,
                                (0, self.segment_length - (end - start))
                        )

                assert segment.shape[-1] == self.segment_length, segment.shape

                # ta.save(os.path.join(outpath, f"{stem}.wav"), segment, self.fs)

                np.save(os.path.join(outpath, f"{stem}.wav"), segment)


def preprocess(
        analysis_stem: str,
        output_path: str = "/data/MUSDB18/HQ/saded-np",
        fs: int = 44100,
        segment_length_second: float = 6.0,
        hop_length_second: float = 3.0,
        n_chunks: int = 10,
        chunk_epsilon: float = 1e-5,
        energy_threshold_quantile: float = 0.15,
        segment_epsilon: float = 1e-3,
        salient_proportion_threshold: float = 0.5,
) -> None:

    sad = SourceActivityDetector(
            analysis_stem=analysis_stem,
            output_path=output_path,
            fs=fs,
            segment_length_second=segment_length_second,
            hop_length_second=hop_length_second,
            n_chunks=n_chunks,
            chunk_epsilon=chunk_epsilon,
            energy_threshold_quantile=energy_threshold_quantile,
            segment_epsilon=segment_epsilon,
            salient_proportion_threshold=salient_proportion_threshold,
    )

    for split in ["train", "val", "test"]:
        ds = MUSDB18FullTrackDataset(
                data_root="/data/MUSDB18/HQ/canonical",
                split=split,
        )

        tracks = []
        for i, track in enumerate(tqdm(ds, total=len(ds))):
            if i % 32 == 0 and tracks:
                process_map(sad, tracks, max_workers=8)
                tracks = []
            tracks.append(track)
        process_map(sad, tracks, max_workers=8)

def loudness_norm_one(
        inputs
):
    infile, outfile, target_lufs = inputs

    audio, fs = ta.load(infile)
    audio = audio.mean(dim=0, keepdim=True).numpy().T

    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    np.save(outfile, audio.T)

def loudness_norm(
        data_path: str,
        # output_path: str,
        target_lufs = -17.0,
):
    files = glob.glob(
            os.path.join(data_path, "**", "*.wav"), recursive=True
    )

    outfiles = [
            f.replace(".wav", ".npy").replace("saded", "saded-np") for f in files
    ]

    files = [(f, o, target_lufs) for f, o in zip(files, outfiles)]

    process_map(loudness_norm_one, files, chunksize=2)



if __name__ == "__main__":

    from tqdm import tqdm
    import fire

    fire.Fire()
