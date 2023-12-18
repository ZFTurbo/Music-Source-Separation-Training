from typing import Any, Dict

import torch
from asteroid import losses as asteroid_losses
from torch import nn
from torch.nn.modules.loss import _Loss

from . import snr


def parse_loss(name: str, kwargs: Dict[str, Any]) -> _Loss:

    for module in [nn.modules.loss, snr, asteroid_losses, asteroid_losses.sdr]:
        if name in module.__dict__:
            return module.__dict__[name](**kwargs)

    raise NameError


class MultiStemWrapper(_Loss):
    def __init__(self, module: _Loss, modality: str = "audio") -> None:
        super().__init__()
        self.loss = module
        self.modality = modality

    def forward(
            self,
            preds: Dict[str, Dict[str, torch.Tensor]],
            target: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        loss = {
                stem: self.loss(
                        preds[self.modality][stem],
                        target[self.modality][stem]
                )
                for stem in preds[self.modality] if stem in target[self.modality]
        }

        return sum(list(loss.values()))


class MultiStemWrapperFromConfig(MultiStemWrapper):
    def __init__(self, name: str, kwargs: Any, modality: str = "audio") -> None:
        loss = parse_loss(name, kwargs)
        super().__init__(module=loss, modality=modality)
