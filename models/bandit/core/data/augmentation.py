from abc import ABC
from typing import Any, Dict, Union

import torch
import torch_audiomentations as tam
from torch import nn

from models.bandit.core.data._types import BatchedDataDict, DataDict


class BaseAugmentor(nn.Module, ABC):
    def forward(self, item: Union[DataDict, BatchedDataDict]) -> Union[
        DataDict, BatchedDataDict]:
        raise NotImplementedError


class StemAugmentor(BaseAugmentor):
    def __init__(
            self,
            audiomentations: Dict[str, Dict[str, Any]],
            fix_clipping: bool = True,
            scaler_margin: float = 0.5,
            apply_both_default_and_common: bool = False,
    ) -> None:
        super().__init__()

        augmentations = {}

        self.has_default = "[default]" in audiomentations
        self.has_common = "[common]" in audiomentations
        self.apply_both_default_and_common = apply_both_default_and_common

        for stem in audiomentations:
            if audiomentations[stem]["name"] == "Compose":
                augmentations[stem] = getattr(
                        tam,
                        audiomentations[stem]["name"]
                )(
                        [
                                getattr(tam, aug["name"])(**aug["kwargs"])
                                for aug in
                                audiomentations[stem]["kwargs"]["transforms"]
                        ],
                        **audiomentations[stem]["kwargs"]["kwargs"],
                )
            else:
                augmentations[stem] = getattr(
                        tam,
                        audiomentations[stem]["name"]
                )(
                        **audiomentations[stem]["kwargs"]
                )

        self.augmentations = nn.ModuleDict(augmentations)
        self.fix_clipping = fix_clipping
        self.scaler_margin = scaler_margin

    def check_and_fix_clipping(
            self, item: Union[DataDict, BatchedDataDict]
    ) -> Union[DataDict, BatchedDataDict]:
        max_abs = []

        for stem in item["audio"]:
            max_abs.append(item["audio"][stem].abs().max().item())

        if max(max_abs) > 1.0:
            scaler = 1.0 / (max(max_abs) + torch.rand(
                    (1,),
                    device=item["audio"]["mixture"].device
            ) * self.scaler_margin)

            for stem in item["audio"]:
                item["audio"][stem] *= scaler

        return item

    def forward(self, item: Union[DataDict, BatchedDataDict]) -> Union[
        DataDict, BatchedDataDict]:

        for stem in item["audio"]:
            if stem == "mixture":
                continue

            if self.has_common:
                item["audio"][stem] = self.augmentations["[common]"](
                        item["audio"][stem]
                ).samples

            if stem in self.augmentations:
                item["audio"][stem] = self.augmentations[stem](
                        item["audio"][stem]
                ).samples
            elif self.has_default:
                if not self.has_common or self.apply_both_default_and_common:
                    item["audio"][stem] = self.augmentations["[default]"](
                            item["audio"][stem]
                    ).samples

        item["audio"]["mixture"] = sum(
                [item["audio"][stem] for stem in item["audio"]
                 if stem != "mixture"]
        )  # type: ignore[call-overload, assignment]

        if self.fix_clipping:
            item = self.check_and_fix_clipping(item)

        return item
