import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pedalboard as pb
import torch
import torchaudio as ta
from torch.utils import data

from models.bandit.core.data._types import AudioDict, DataDict


class BaseSourceSeparationDataset(data.Dataset, ABC):
    def __init__(
            self, split: str,
            stems: List[str],
            files: List[str],
            data_path: str,
            fs: int,
            npy_memmap: bool,
            recompute_mixture: bool
            ):
        self.split = split
        self.stems = stems
        self.stems_no_mixture = [s for s in stems if s != "mixture"]
        self.files = files
        self.data_path = data_path
        self.fs = fs
        self.npy_memmap = npy_memmap
        self.recompute_mixture = recompute_mixture

    @abstractmethod
    def get_stem(
            self,
            *,
            stem: str,
            identifier: Dict[str, Any]
            ) -> torch.Tensor:
        raise NotImplementedError

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}
        for stem in stems:
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_audio(self, identifier: Dict[str, Any]) -> AudioDict:

        if self.recompute_mixture:
            audio = self._get_audio(
                self.stems_no_mixture,
                identifier=identifier
                )
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            return self._get_audio(self.stems, identifier=identifier)

    @abstractmethod
    def get_identifier(self, index: int) -> Dict[str, Any]:
        pass

    def compute_mixture(self, audio: AudioDict) -> torch.Tensor:

        return sum(
                audio[stem] for stem in audio if stem != "mixture"
        )
