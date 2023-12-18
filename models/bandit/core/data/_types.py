from typing import Dict, Sequence, TypedDict

import torch

AudioDict = Dict[str, torch.Tensor]

DataDict = TypedDict('DataDict', {'audio': AudioDict, 'track': str})

BatchedDataDict = TypedDict(
        'BatchedDataDict',
        {'audio': AudioDict, 'track': Sequence[str]}
)


class DataDictWithLanguage(TypedDict):
    audio: AudioDict
    track: str
    language: str
