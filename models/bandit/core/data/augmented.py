import warnings
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.utils import data


class AugmentedDataset(data.Dataset):
    def __init__(
            self,
            dataset: data.Dataset,
            augmentation: nn.Module = nn.Identity(),
            target_length: Optional[int] = None,
    ) -> None:
        warnings.warn(
                "This class is no longer used. Attach augmentation to "
                "the LightningSystem instead.",
                DeprecationWarning,
        )

        self.dataset = dataset
        self.augmentation = augmentation

        self.ds_length: int = len(dataset)  # type: ignore[arg-type]
        self.length = target_length if target_length is not None else self.ds_length

    def __getitem__(self, index: int) -> Dict[str, Union[str, Dict[str,
    torch.Tensor]]]:
        item = self.dataset[index % self.ds_length]
        item = self.augmentation(item)
        return item

    def __len__(self) -> int:
        return self.length
