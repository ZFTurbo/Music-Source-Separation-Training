import os
from abc import ABC
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio as ta
from torch.utils import data

from models.bandit.core.data._types import AudioDict, DataDict
from models.bandit.core.data.base import BaseSourceSeparationDataset


class MUSDB18BaseDataset(BaseSourceSeparationDataset, ABC):

    ALLOWED_STEMS = ["mixture", "vocals", "bass", "drums", "other"]

    def __init__(
            self,
            split: str,
            stems: List[str],
            files: List[str],
            data_path: str,
            fs: int = 44100,
            npy_memmap=False,
    ) -> None:
        super().__init__(
                split=split,
                stems=stems,
                files=files,
                data_path=data_path,
                fs=fs,
                npy_memmap=npy_memmap,
                recompute_mixture=False
        )

    def get_stem(self, *, stem: str, identifier) -> torch.Tensor:
        track = identifier["track"]
        path = os.path.join(self.data_path, track)
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            audio = np.load(os.path.join(path, f"{stem}.wav.npy"), mmap_mode="r")
        else:
            audio, _ = ta.load(os.path.join(path, f"{stem}.wav"))

        return audio

    def get_identifier(self, index):
        return dict(track=self.files[index])

    def __getitem__(self, index: int) -> DataDict:
        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier)

        return {"audio": audio, "track": f"{self.split}/{identifier['track']}"}


class MUSDB18FullTrackDataset(MUSDB18BaseDataset):

    N_TRAIN_TRACKS = 100
    N_TEST_TRACKS = 50
    VALIDATION_FILES = [
            "Actions - One Minute Smile",
            "Clara Berry And Wooldog - Waltz For My Victims",
            "Johnny Lokke - Promises & Lies",
            "Patrick Talbot - A Reason To Leave",
            "Triviul - Angelsaint",
            "Alexander Ross - Goodbye Bolero",
            "Fergessen - Nos Palpitants",
            "Leaf - Summerghost",
            "Skelpolu - Human Mistakes",
            "Young Griffo - Pennies",
            "ANiMAL - Rockshow",
            "James May - On The Line",
            "Meaxic - Take A Step",
            "Traffic Experiment - Sirens",
    ]

    def __init__(
            self, data_root: str, split: str, stems: Optional[List[
                str]] = None
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS
        self.stems = stems

        if split == "test":
            subset = "test"
        elif split in ["train", "val"]:
            subset = "train"
        else:
            raise NameError

        data_path = os.path.join(data_root, subset)

        files = sorted(os.listdir(data_path))
        files = [f for f in files if not f.startswith(".")]
        # pprint(list(enumerate(files)))
        if subset == "train":
            assert len(files) == 100, len(files)
            if split == "train":
                files = [f for f in files if f not in self.VALIDATION_FILES]
                assert len(files) == 100 - len(self.VALIDATION_FILES)
            else:
                files = [f for f in files if f in self.VALIDATION_FILES]
                assert len(files) == len(self.VALIDATION_FILES)
        else:
            split = "test"
            assert len(files) == 50

        self.n_tracks = len(files)

        super().__init__(
                data_path=data_path,
                split=split,
                stems=stems,
                files=files
        )

    def __len__(self) -> int:
        return self.n_tracks

class MUSDB18SadDataset(MUSDB18BaseDataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            target_stem: str,
            stems: Optional[List[str]] = None,
            target_length: Optional[int] = None,
            npy_memmap=False,
    ) -> None:

        if stems is None:
            stems = self.ALLOWED_STEMS

        data_path = os.path.join(data_root, target_stem, split)

        files = sorted(os.listdir(data_path))
        files = [f for f in files if not f.startswith(".")]

        super().__init__(
                data_path=data_path,
                split=split,
                stems=stems,
                files=files,
                npy_memmap=npy_memmap
        )
        self.n_segments = len(files)
        self.target_stem = target_stem
        self.target_length = (
                target_length if target_length is not None else self.n_segments
        )

    def __len__(self) -> int:
        return self.target_length

    def __getitem__(self, index: int) -> DataDict:

        index = index % self.n_segments

        return super().__getitem__(index)

    def get_identifier(self, index):
        return super().get_identifier(index % self.n_segments)


class MUSDB18SadOnTheFlyAugmentedDataset(MUSDB18SadDataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            target_stem: str,
            stems: Optional[List[str]] = None,
            target_length: int = 20000,
            apply_probability: Optional[float] = None,
            chunk_size_second: float = 3.0,
            random_scale_range_db: Tuple[float, float] = (-10, 10),
            drop_probability: float = 0.1,
            rescale: bool = True,
    ) -> None:
        super().__init__(data_root, split, target_stem, stems)

        if apply_probability is None:
            apply_probability = (
                                        target_length - self.n_segments) / target_length

        self.apply_probability = apply_probability
        self.drop_probability = drop_probability
        self.chunk_size_second = chunk_size_second
        self.random_scale_range_db = random_scale_range_db
        self.rescale = rescale

        self.chunk_size_sample = int(self.chunk_size_second * self.fs)
        self.target_length = target_length

    def __len__(self) -> int:
        return self.target_length

    def __getitem__(self, index: int) -> DataDict:

        index = index % self.n_segments

        # if np.random.rand() > self.apply_probability:
        #     return super().__getitem__(index)

        audio = {}
        identifier = self.get_identifier(index)

        # assert self.target_stem in self.stems_no_mixture
        for stem in self.stems_no_mixture:
            if stem == self.target_stem:
                identifier_ = identifier
            else:
                if np.random.rand() < self.apply_probability:
                    index_ = np.random.randint(self.n_segments)
                    identifier_ = self.get_identifier(index_)
                else:
                    identifier_ = identifier

            audio[stem] = self.get_stem(stem=stem, identifier=identifier_)

            # if stem == self.target_stem:

            if self.chunk_size_sample < audio[stem].shape[-1]:
                chunk_start = np.random.randint(
                        audio[stem].shape[-1] - self.chunk_size_sample
                )
            else:
                chunk_start = 0

            if np.random.rand() < self.drop_probability:
                # db_scale = "-inf"
                linear_scale = 0.0
            else:
                db_scale = np.random.uniform(*self.random_scale_range_db)
                linear_scale = np.power(10, db_scale / 20)
                # db_scale = f"{db_scale:+2.1f}"
            # print(linear_scale)
            audio[stem][...,
            chunk_start: chunk_start + self.chunk_size_sample] = (
                    linear_scale
                    * audio[stem][...,
                      chunk_start: chunk_start + self.chunk_size_sample]
            )

        audio["mixture"] = self.compute_mixture(audio)

        if self.rescale:
            max_abs_val = max(
                    [torch.max(torch.abs(audio[stem])) for stem in self.stems]
            )  # type: ignore[type-var]
            if max_abs_val > 1:
                audio = {k: v / max_abs_val for k, v in audio.items()}

        track = identifier["track"]

        return {"audio": audio, "track": f"{self.split}/{track}"}

# if __name__ == "__main__":
#
#     from pprint import pprint
#     from tqdm.auto import tqdm
#
#     for split_ in ["train", "val", "test"]:
#         ds = MUSDB18SadOnTheFlyAugmentedDataset(
#             data_root="$DATA_ROOT/MUSDB18/HQ/saded",
#             split=split_,
#             target_stem="vocals"
#         )
#
#         print(split_, len(ds))
#
#         for track_ in tqdm(ds):
#             track_["audio"] = {
#                 k: v.shape for k, v in track_["audio"].items()
#             }
#         pprint(track_)
