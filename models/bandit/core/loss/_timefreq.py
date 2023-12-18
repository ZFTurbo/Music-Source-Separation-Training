from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from models.bandit.core.loss._multistem import MultiStemWrapper
from models.bandit.core.loss._complex import ReImL1Loss, ReImL2Loss, ReImLossWrapper
from models.bandit.core.loss.snr import SignalNoisePNormRatio

class TimeFreqWrapper(_Loss):
    def __init__(
            self,
            time_module: _Loss,
            freq_module: Optional[_Loss] = None,
            time_weight: float = 1.0,
            freq_weight: float = 1.0,
            multistem: bool = True,
    ) -> None:
        super().__init__()

        if freq_module is None:
            freq_module = time_module

        if multistem:
            time_module = MultiStemWrapper(time_module, modality="audio")
            freq_module = MultiStemWrapper(freq_module, modality="spectrogram")

        self.time_module = time_module
        self.freq_module = freq_module

        self.time_weight = time_weight
        self.freq_weight = freq_weight

    # TODO: add better type hints
    def forward(self, preds: Any, target: Any) -> torch.Tensor:

        return self.time_weight * self.time_module(
                preds, target
        ) + self.freq_weight * self.freq_module(preds, target)


class TimeFreqL1Loss(TimeFreqWrapper):
    def __init__(
            self,
            time_weight: float = 1.0,
            freq_weight: float = 1.0,
            tkwargs: Optional[Dict[str, Any]] = None,
            fkwargs: Optional[Dict[str, Any]] = None,
            multistem: bool = True,
    ) -> None:
        if tkwargs is None:
            tkwargs = {}
        if fkwargs is None:
            fkwargs = {}
        time_module = (nn.L1Loss(**tkwargs))
        freq_module = ReImL1Loss(**fkwargs)
        super().__init__(
                time_module,
                freq_module,
                time_weight,
                freq_weight,
                multistem
        )


class TimeFreqL2Loss(TimeFreqWrapper):
    def __init__(
            self,
            time_weight: float = 1.0,
            freq_weight: float = 1.0,
            tkwargs: Optional[Dict[str, Any]] = None,
            fkwargs: Optional[Dict[str, Any]] = None,
            multistem: bool = True,
    ) -> None:
        if tkwargs is None:
            tkwargs = {}
        if fkwargs is None:
            fkwargs = {}
        time_module = nn.MSELoss(**tkwargs)
        freq_module = ReImL2Loss(**fkwargs)
        super().__init__(
                time_module,
                freq_module,
                time_weight,
                freq_weight,
                multistem
        )



class TimeFreqSignalNoisePNormRatioLoss(TimeFreqWrapper):
    def __init__(
            self,
            time_weight: float = 1.0,
            freq_weight: float = 1.0,
            tkwargs: Optional[Dict[str, Any]] = None,
            fkwargs: Optional[Dict[str, Any]] = None,
            multistem: bool = True,
    ) -> None:
        if tkwargs is None:
            tkwargs = {}
        if fkwargs is None:
            fkwargs = {}
        time_module = SignalNoisePNormRatio(**tkwargs)
        freq_module = SignalNoisePNormRatio(**fkwargs)
        super().__init__(
                time_module,
                freq_module,
                time_weight,
                freq_weight,
                multistem
        )
