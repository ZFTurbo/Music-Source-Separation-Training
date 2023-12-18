from typing import Any

import torch
from torch import nn
from torch.nn.modules import loss as _loss
from torch.nn.modules.loss import _Loss


class ReImLossWrapper(_Loss):
    def __init__(self, module: _Loss) -> None:
        super().__init__()
        self.module = module

    def forward(
            self,
            preds: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:
        return self.module(
            torch.view_as_real(preds),
            torch.view_as_real(target)
            )


class ReImL1Loss(ReImLossWrapper):
    def __init__(self, **kwargs: Any) -> None:
        l1_loss = _loss.L1Loss(**kwargs)
        super().__init__(module=(l1_loss))


class ReImL2Loss(ReImLossWrapper):
    def __init__(self, **kwargs: Any) -> None:
        l2_loss = _loss.MSELoss(**kwargs)
        super().__init__(module=(l2_loss))
