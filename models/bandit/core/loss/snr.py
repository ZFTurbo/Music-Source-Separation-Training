import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class SignalNoisePNormRatio(_Loss):
    def __init__(
            self,
            p: float = 1.0,
            scale_invariant: bool = False,
            zero_mean: bool = False,
            take_log: bool = True,
            reduction: str = "mean",
            EPS: float = 1e-3,
    ) -> None:
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)
        assert not zero_mean

        self.p = p

        self.EPS = EPS
        self.take_log = take_log

        self.scale_invariant = scale_invariant

    def forward(
            self,
            est_target: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:

        target_ = target
        if self.scale_invariant:
            ndim = target.ndim
            dot = torch.sum(est_target * torch.conj(target), dim=-1, keepdim=True)
            s_target_energy = (
                    torch.sum(target * torch.conj(target), dim=-1, keepdim=True)
            )

            if ndim > 2:
                dot = torch.sum(dot, dim=list(range(1, ndim)), keepdim=True)
                s_target_energy = torch.sum(s_target_energy, dim=list(range(1, ndim)), keepdim=True)

            target_scaler = (dot + 1e-8) / (s_target_energy + 1e-8)
            target = target_ * target_scaler

        if torch.is_complex(est_target):
            est_target = torch.view_as_real(est_target)
            target = torch.view_as_real(target)


        batch_size = est_target.shape[0]
        est_target = est_target.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        # target_ = target_.reshape(batch_size, -1)

        if self.p == 1:
            e_error = torch.abs(est_target-target).mean(dim=-1)
            e_target = torch.abs(target).mean(dim=-1)
        elif self.p == 2:
            e_error = torch.square(est_target-target).mean(dim=-1)
            e_target = torch.square(target).mean(dim=-1)
        else:
            raise NotImplementedError
        
        if self.take_log:
            loss = 10*(torch.log10(e_error + self.EPS) - torch.log10(e_target + self.EPS))
        else:
            loss = (e_error + self.EPS)/(e_target + self.EPS)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

        

class MultichannelSingleSrcNegSDR(_Loss):
    def __init__(
            self,
            sdr_type: str,
            p: float = 2.0,
            zero_mean: bool = True,
            take_log: bool = True,
            reduction: str = "mean",
            EPS: float = 1e-8,
    ) -> None:
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

        self.p = p

    def forward(
            self,
            est_target: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:
        if target.size() != est_target.size() or target.ndim != 3:
            raise TypeError(
                    f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=[1, 2], keepdim=True)
            mean_estimate = torch.mean(est_target, dim=[1, 2], keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=[1, 2], keepdim=True)
            # [batch, 1]
            s_target_energy = (
                    torch.sum(target ** 2, dim=[1, 2], keepdim=True) + self.EPS
            )
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]

        if self.p == 2.0:
            losses = torch.sum(scaled_target ** 2, dim=[1, 2]) / (
                    torch.sum(e_noise ** 2, dim=[1, 2]) + self.EPS
            )
        else:
            losses = torch.norm(scaled_target, p=self.p, dim=[1, 2]) / (
                    torch.linalg.vector_norm(e_noise, p=self.p, dim=[1, 2]) + self.EPS
            )
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses
