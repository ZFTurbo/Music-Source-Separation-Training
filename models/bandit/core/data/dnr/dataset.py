import os
from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np
import pedalboard as pb
import torch
import torchaudio as ta
from torch.utils import data

from models.bandit.core.data._types import AudioDict, DataDict
from models.bandit.core.data.base import BaseSourceSeparationDataset


class DivideAndRemasterBaseDataset(BaseSourceSeparationDataset, ABC):
    ALLOWED_STEMS = ["mixture", "speech", "music", "effects", "mne"]
    STEM_NAME_MAP = {
            "mixture": "mix",
            "speech": "speech",
            "music": "music",
            "effects": "sfx",
    }
    SPLIT_NAME_MAP = {"train": "tr", "val": "cv", "test": "tt"}

    FULL_TRACK_LENGTH_SECOND = 60
    FULL_TRACK_LENGTH_SAMPLES = FULL_TRACK_LENGTH_SECOND * 44100

    def __init__(
            self,
            split: str,
            stems: List[str],
            files: List[str],
            data_path: str,
            fs: int = 44100,
            npy_memmap: bool = True,
            recompute_mixture: bool = False,
    ) -> None:
        super().__init__(
                split=split,
                stems=stems,
                files=files,
                data_path=data_path,
                fs=fs,
                npy_memmap=npy_memmap,
                recompute_mixture=recompute_mixture
        )

    def get_stem(
            self,
            *,
            stem: str,
            identifier: Dict[str, Any]
            ) -> torch.Tensor:
        
        if stem == "mne":
            return self.get_stem(
                stem="music",
                identifier=identifier) + self.get_stem(
                stem="effects",
                identifier=identifier)

        track = identifier["track"]
        path = os.path.join(self.data_path, track)

        if self.npy_memmap:
            audio = np.load(
                    os.path.join(path, f"{self.STEM_NAME_MAP[stem]}.npy"),
                    mmap_mode="r"
            )
        else:
            # noinspection PyUnresolvedReferences
            audio, _ = ta.load(
                    os.path.join(path, f"{self.STEM_NAME_MAP[stem]}.wav")
            )

        return audio

    def get_identifier(self, index):
        return dict(track=self.files[index])

    def __getitem__(self, index: int) -> DataDict:
        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier)

        return {"audio": audio, "track": f"{self.split}/{identifier['track']}"}


class DivideAndRemasterDataset(DivideAndRemasterBaseDataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            stems: Optional[List[str]] = None,
            fs: int = 44100,
            npy_memmap: bool = True,
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS
        self.stems = stems

        data_path = os.path.join(data_root, self.SPLIT_NAME_MAP[split])

        files = sorted(os.listdir(data_path))
        files = [
                f
                for f in files
                if (not f.startswith(".")) and os.path.isdir(
                        os.path.join(data_path, f)
                )
        ]
        # pprint(list(enumerate(files)))
        if split == "train":
            assert len(files) == 3406, len(files)
        elif split == "val":
            assert len(files) == 487, len(files)
        elif split == "test":
            assert len(files) == 973, len(files)

        self.n_tracks = len(files)

        super().__init__(
                data_path=data_path,
                split=split,
                stems=stems,
                files=files,
                fs=fs,
                npy_memmap=npy_memmap,
        )

    def __len__(self) -> int:
        return self.n_tracks


class DivideAndRemasterRandomChunkDataset(DivideAndRemasterBaseDataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            target_length: int,
            chunk_size_second: float,
            stems: Optional[List[str]] = None,
            fs: int = 44100,
            npy_memmap: bool = True,
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS
        self.stems = stems

        data_path = os.path.join(data_root, self.SPLIT_NAME_MAP[split])

        files = sorted(os.listdir(data_path))
        files = [
                f
                for f in files
                if (not f.startswith(".")) and os.path.isdir(
                        os.path.join(data_path, f)
                )
        ]

        if split == "train":
            assert len(files) == 3406, len(files)
        elif split == "val":
            assert len(files) == 487, len(files)
        elif split == "test":
            assert len(files) == 973, len(files)

        self.n_tracks = len(files)

        self.target_length = target_length
        self.chunk_size = int(chunk_size_second * fs)

        super().__init__(
                data_path=data_path,
                split=split,
                stems=stems,
                files=files,
                fs=fs,
                npy_memmap=npy_memmap,
        )

    def __len__(self) -> int:
        return self.target_length

    def get_identifier(self, index):
        return super().get_identifier(index % self.n_tracks)

    def get_stem(
            self,
            *,
            stem: str,
            identifier: Dict[str, Any],
            chunk_here: bool = False,
            ) -> torch.Tensor:

        stem = super().get_stem(
                stem=stem,
                identifier=identifier
        )

        if chunk_here:
            start = np.random.randint(
                    0,
                    self.FULL_TRACK_LENGTH_SAMPLES - self.chunk_size
            )
            end = start + self.chunk_size

            stem = stem[:, start:end]

        return stem

    def __getitem__(self, index: int) -> DataDict:
        identifier = self.get_identifier(index)
        # self.index_lock = index
        audio = self.get_audio(identifier)
        # self.index_lock = None

        start = np.random.randint(
                0,
                self.FULL_TRACK_LENGTH_SAMPLES - self.chunk_size
        )
        end = start + self.chunk_size

        audio = {
                k: v[:, start:end] for k, v in audio.items()
        }

        return {"audio": audio, "track": f"{self.split}/{identifier['track']}"}


class DivideAndRemasterDeterministicChunkDataset(DivideAndRemasterBaseDataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            chunk_size_second: float,
            hop_size_second: float,
            stems: Optional[List[str]] = None,
            fs: int = 44100,
            npy_memmap: bool = True,
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS
        self.stems = stems

        data_path = os.path.join(data_root, self.SPLIT_NAME_MAP[split])

        files = sorted(os.listdir(data_path))
        files = [
                f
                for f in files
                if (not f.startswith(".")) and os.path.isdir(
                        os.path.join(data_path, f)
                )
        ]
        # pprint(list(enumerate(files)))
        if split == "train":
            assert len(files) == 3406, len(files)
        elif split == "val":
            assert len(files) == 487, len(files)
        elif split == "test":
            assert len(files) == 973, len(files)

        self.n_tracks = len(files)

        self.chunk_size = int(chunk_size_second * fs)
        self.hop_size = int(hop_size_second * fs)
        self.n_chunks_per_track = int(
                (
                        self.FULL_TRACK_LENGTH_SECOND - chunk_size_second) / hop_size_second
        )

        self.length = self.n_tracks * self.n_chunks_per_track

        super().__init__(
                data_path=data_path,
                split=split,
                stems=stems,
                files=files,
                fs=fs,
                npy_memmap=npy_memmap,
        )

    def get_identifier(self, index):
        return super().get_identifier(index % self.n_tracks)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> DataDict:

        index = item % self.n_tracks
        chunk = item // self.n_tracks

        data_ = super().__getitem__(index)

        audio = data_["audio"]

        start = chunk * self.hop_size
        end = start + self.chunk_size

        for stem in self.stems:
            data_["audio"][stem] = audio[stem][:, start:end]

        return data_


class DivideAndRemasterRandomChunkDatasetWithSpeechReverb(
        DivideAndRemasterRandomChunkDataset
):
    def __init__(
            self,
            data_root: str,
            split: str,
            target_length: int,
            chunk_size_second: float,
            stems: Optional[List[str]] = None,
            fs: int = 44100,
            npy_memmap: bool = True,
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS

        stems_no_mixture = [s for s in stems if s != "mixture"]

        super().__init__(
                data_root=data_root,
                split=split,
                target_length=target_length,
                chunk_size_second=chunk_size_second,
                stems=stems_no_mixture,
                fs=fs,
                npy_memmap=npy_memmap,
        )

        self.stems = stems
        self.stems_no_mixture = stems_no_mixture

    def __getitem__(self, index: int) -> DataDict:

        data_ = super().__getitem__(index)

        dry = data_["audio"]["speech"][:]
        n_samples = dry.shape[-1]

        wet_level = np.random.rand()

        speech = pb.Reverb(
                room_size=np.random.rand(),
                damping=np.random.rand(),
                wet_level=wet_level,
                dry_level=(1 - wet_level),
                width=np.random.rand()
        ).process(dry, self.fs, buffer_size=8192 * 4)[..., :n_samples]

        data_["audio"]["speech"] = speech

        data_["audio"]["mixture"] = sum(
                [data_["audio"][s] for s in self.stems_no_mixture]
        )

        return data_

    def __len__(self) -> int:
        return super().__len__()


if __name__ == "__main__":

    from pprint import pprint
    from tqdm.auto import tqdm

    for split_ in ["train", "val", "test"]:
        ds = DivideAndRemasterRandomChunkDatasetWithSpeechReverb(
                data_root="$DATA_ROOT/DnR/v2np",
                split=split_,
                target_length=100,
                chunk_size_second=6.0
        )

        print(split_, len(ds))

        for track_ in tqdm(ds):  # type: ignore
            pprint(track_)
            track_["audio"] = {k: v.shape for k, v in track_["audio"].items()}
            pprint(track_)
            # break

        break
