import os.path
from typing import Mapping, Optional

import pytorch_lightning as pl

from models.bandit.core.data.musdb.dataset import (
    MUSDB18BaseDataset,
    MUSDB18FullTrackDataset,
    MUSDB18SadDataset,
    MUSDB18SadOnTheFlyAugmentedDataset
)


def MUSDB18DataModule(
        data_root: str = "$DATA_ROOT/MUSDB18/HQ",
        target_stem: str = "vocals",
        batch_size: int = 2,
        num_workers: int = 8,
        train_kwargs: Optional[Mapping] = None,
        val_kwargs: Optional[Mapping] = None,
        test_kwargs: Optional[Mapping] = None,
        datamodule_kwargs: Optional[Mapping] = None,
        use_on_the_fly: bool = True,
        npy_memmap: bool = True
) -> pl.LightningDataModule:
    if train_kwargs is None:
        train_kwargs = {}

    if val_kwargs is None:
        val_kwargs = {}

    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}

    train_dataset: MUSDB18BaseDataset

    if use_on_the_fly:
        train_dataset = MUSDB18SadOnTheFlyAugmentedDataset(
                data_root=os.path.join(data_root, "saded-np"),
                split="train",
                target_stem=target_stem,
                **train_kwargs
        )
    else:
        train_dataset = MUSDB18SadDataset(
                data_root=os.path.join(data_root, "saded-np"),
                split="train",
                target_stem=target_stem,
                **train_kwargs
        )

    datamodule = pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=MUSDB18SadDataset(
                    data_root=os.path.join(data_root, "saded-np"),
                    split="val",
                    target_stem=target_stem,
                    **val_kwargs
            ),
            test_dataset=MUSDB18FullTrackDataset(
                    data_root=os.path.join(data_root, "canonical"),
                    split="test",
                    **test_kwargs
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            **datamodule_kwargs
    )

    datamodule.predict_dataloader = (  # type: ignore[method-assign]
            datamodule.test_dataloader
    )

    return datamodule
