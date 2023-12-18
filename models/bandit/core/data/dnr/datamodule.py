import os
from typing import Mapping, Optional

import pytorch_lightning as pl

from .dataset import (
    DivideAndRemasterDataset,
    DivideAndRemasterDeterministicChunkDataset,
    DivideAndRemasterRandomChunkDataset,
    DivideAndRemasterRandomChunkDatasetWithSpeechReverb
)


def DivideAndRemasterDataModule(
        data_root: str = "$DATA_ROOT/DnR/v2",
        batch_size: int = 2,
        num_workers: int = 8,
        train_kwargs: Optional[Mapping] = None,
        val_kwargs: Optional[Mapping] = None,
        test_kwargs: Optional[Mapping] = None,
        datamodule_kwargs: Optional[Mapping] = None,
        use_speech_reverb: bool = False
        # augmentor=None
) -> pl.LightningDataModule:
    if train_kwargs is None:
        train_kwargs = {}

    if val_kwargs is None:
        val_kwargs = {}

    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}

    if num_workers is None:
        num_workers = os.cpu_count()

        if num_workers is None:
            num_workers = 32

        num_workers = min(num_workers, 64)

    if use_speech_reverb:
        train_cls = DivideAndRemasterRandomChunkDatasetWithSpeechReverb
    else:
        train_cls = DivideAndRemasterRandomChunkDataset

    train_dataset = train_cls(
            data_root, "train", **train_kwargs
    )

    # if augmentor is not None:
    #     train_dataset = AugmentedDataset(train_dataset, augmentor)

    datamodule = pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=DivideAndRemasterDeterministicChunkDataset(
                    data_root, "val", **val_kwargs
            ),
            test_dataset=DivideAndRemasterDataset(
                data_root,
                "test",
                **test_kwargs
                ),
            batch_size=batch_size,
            num_workers=num_workers,
            **datamodule_kwargs
    )

    datamodule.predict_dataloader = datamodule.test_dataloader  # type: ignore[method-assign]

    return datamodule
