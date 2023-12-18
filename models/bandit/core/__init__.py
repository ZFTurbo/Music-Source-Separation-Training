import os.path
from collections import defaultdict
from itertools import chain, combinations
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping, Optional,
    Tuple, Type,
    TypedDict
)

import pytorch_lightning as pl
import torch
import torchaudio as ta
import torchmetrics as tm
from asteroid import losses as asteroid_losses
# from deepspeed.ops.adam import DeepSpeedCPUAdam
# from geoopt import optim as gooptim
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from models.bandit.core import loss, metrics as metrics_, model
from models.bandit.core.data._types import BatchedDataDict
from models.bandit.core.data.augmentation import BaseAugmentor, StemAugmentor
from models.bandit.core.utils import audio as audio_
from models.bandit.core.utils.audio import BaseFader

# from pandas.io.json._normalize import nested_to_record

ConfigDict = TypedDict('ConfigDict', {'name': str, 'kwargs': Dict[str, Any]})


class SchedulerConfigDict(ConfigDict):
    monitor: str


OptimizerSchedulerConfigDict = TypedDict(
        'OptimizerSchedulerConfigDict',
        {"optimizer": ConfigDict, "scheduler": SchedulerConfigDict},
        total=False
)


class LRSchedulerReturnDict(TypedDict, total=False):
    scheduler: LRScheduler
    monitor: str


class ConfigureOptimizerReturnDict(TypedDict, total=False):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerReturnDict


OutputType = Dict[str, Any]
MetricsType = Dict[str, torch.Tensor]


def get_optimizer_class(name: str) -> Type[optim.Optimizer]:

    if name == "DeepSpeedCPUAdam":
        return DeepSpeedCPUAdam

    for module in [optim, gooptim]:
        if name in module.__dict__:
            return module.__dict__[name]

    raise NameError


def parse_optimizer_config(
        config: OptimizerSchedulerConfigDict,
        parameters: Iterator[nn.Parameter]
) -> ConfigureOptimizerReturnDict:
    optim_class = get_optimizer_class(config["optimizer"]["name"])
    optimizer = optim_class(parameters, **config["optimizer"]["kwargs"])

    optim_dict: ConfigureOptimizerReturnDict = {
            "optimizer": optimizer,
    }

    if "scheduler" in config:

        lr_scheduler_class_ = config["scheduler"]["name"]
        lr_scheduler_class = lr_scheduler.__dict__[lr_scheduler_class_]
        lr_scheduler_dict: LRSchedulerReturnDict = {
                "scheduler": lr_scheduler_class(
                        optimizer,
                        **config["scheduler"]["kwargs"]
                )
        }

        if lr_scheduler_class_ == "ReduceLROnPlateau":
            lr_scheduler_dict["monitor"] = config["scheduler"]["monitor"]

        optim_dict["lr_scheduler"] = lr_scheduler_dict

    return optim_dict


def parse_model_config(config: ConfigDict) -> Any:
    name = config["name"]

    for module in [model]:
        if name in module.__dict__:
            return module.__dict__[name](**config["kwargs"])

    raise NameError


_LEGACY_LOSS_NAMES = ["HybridL1Loss"]


def _parse_legacy_loss_config(config: ConfigDict) -> nn.Module:
    name = config["name"]

    if name == "HybridL1Loss":
        return loss.TimeFreqL1Loss(**config["kwargs"])

    raise NameError


def parse_loss_config(config: ConfigDict) -> nn.Module:
    name = config["name"]

    if name in _LEGACY_LOSS_NAMES:
        return _parse_legacy_loss_config(config)

    for module in [loss, nn.modules.loss, asteroid_losses]:
        if name in module.__dict__:
            # print(config["kwargs"])
            return module.__dict__[name](**config["kwargs"])

    raise NameError


def get_metric(config: ConfigDict) -> tm.Metric:
    name = config["name"]

    for module in [tm, metrics_]:
        if name in module.__dict__:
            return module.__dict__[name](**config["kwargs"])
    raise NameError


def parse_metric_config(config: Dict[str, ConfigDict]) -> tm.MetricCollection:
    metrics = {}

    for metric in config:
        metrics[metric] = get_metric(config[metric])

    return tm.MetricCollection(metrics)


def parse_fader_config(config: ConfigDict) -> BaseFader:
    name = config["name"]

    for module in [audio_]:
        if name in module.__dict__:
            return module.__dict__[name](**config["kwargs"])

    raise NameError


class LightningSystem(pl.LightningModule):
    _VOX_STEMS = ["speech", "vocals"]
    _BG_STEMS = ["background", "effects", "mne"]

    def __init__(
            self,
            config: Dict,
            loss_adjustment: float = 1.0,
            attach_fader: bool = False
            ) -> None:
        super().__init__()
        self.optimizer_config = config["optimizer"]
        self.model = parse_model_config(config["model"])
        self.loss = parse_loss_config(config["loss"])
        self.metrics = nn.ModuleDict(
                {
                        stem: parse_metric_config(config["metrics"]["dev"])
                        for stem in self.model.stems
                }
        )

        self.metrics.disallow_fsdp = True

        self.test_metrics = nn.ModuleDict(
                {
                        stem: parse_metric_config(config["metrics"]["test"])
                        for stem in self.model.stems
                }
        )

        self.test_metrics.disallow_fsdp = True

        self.fs = config["model"]["kwargs"]["fs"]

        self.fader_config = config["inference"]["fader"]
        if attach_fader:
            self.fader = parse_fader_config(config["inference"]["fader"])
        else:
            self.fader = None

        self.augmentation: Optional[BaseAugmentor]
        if config.get("augmentation", None) is not None:
            self.augmentation = StemAugmentor(**config["augmentation"])
        else:
            self.augmentation = None

        self.predict_output_path: Optional[str] = None
        self.loss_adjustment = loss_adjustment

        self.val_prefix = None
        self.test_prefix = None


    def configure_optimizers(self) -> Any:
        return parse_optimizer_config(
            self.optimizer_config,
            self.trainer.model.parameters()
            )

    def compute_loss(self, batch: BatchedDataDict, output: OutputType) -> Dict[
        str, torch.Tensor]:
        return {"loss": self.loss(output, batch)}

    def update_metrics(
            self,
            batch: BatchedDataDict,
            output: OutputType,
            mode: str
    ) -> None:

        if mode == "test":
            metrics = self.test_metrics
        else:
            metrics = self.metrics

        for stem, metric in metrics.items():

            if stem == "mne:+":
                stem = "mne"

            # print(f"matching for {stem}")
            if mode == "train":
                metric.update(
                    output["audio"][stem],#.cpu(),
                    batch["audio"][stem],#.cpu()
                    )
            else:
                if stem not in batch["audio"]:
                    matched = False
                    if stem in self._VOX_STEMS:
                        for bstem in self._VOX_STEMS:
                            if bstem in batch["audio"]:
                                batch["audio"][stem] = batch["audio"][bstem]
                                matched = True
                                break
                    elif stem in self._BG_STEMS:
                        for bstem in self._BG_STEMS:
                            if bstem in batch["audio"]:
                                batch["audio"][stem] = batch["audio"][bstem]
                                matched = True
                                break
                else:
                    matched = True

                # print(batch["audio"].keys())

                if matched:
                    # print(f"matched {stem}!")
                    if stem == "mne" and "mne" not in output["audio"]:
                        output["audio"]["mne"] = output["audio"]["music"] + output["audio"]["effects"]
                    
                    metric.update(
                        output["audio"][stem],#.cpu(),
                        batch["audio"][stem],#.cpu(),
                    )

                    # print(metric.compute())
    def compute_metrics(self, mode: str="dev") -> Dict[
        str, torch.Tensor]:

        if mode == "test":
            metrics = self.test_metrics
        else:
            metrics = self.metrics

        metric_dict = {}

        for stem, metric in metrics.items():
            md = metric.compute()
            metric_dict.update(
                    {f"{stem}/{k}": v for k, v in md.items()}
                    )
            
        self.log_dict(metric_dict, prog_bar=True, logger=False)

        return metric_dict

    def reset_metrics(self, test_mode: bool = False) -> None:

        if test_mode:
            metrics = self.test_metrics
        else:
            metrics = self.metrics

        for _, metric in metrics.items():
            metric.reset()


    def forward(self, batch: BatchedDataDict) -> Any:
        batch, output = self.model(batch)
        

        return batch, output

    def common_step(self, batch: BatchedDataDict, mode: str) -> Any:
        batch, output = self.forward(batch)
        # print(batch)
        # print(output)
        loss_dict = self.compute_loss(batch, output)

        with torch.no_grad():
            self.update_metrics(batch, output, mode=mode)

        if mode == "train":
            self.log("loss", loss_dict["loss"], prog_bar=True)

        return output, loss_dict


    def training_step(self, batch: BatchedDataDict) -> Dict[str, Any]:

        if self.augmentation is not None:
            with torch.no_grad():
                batch = self.augmentation(batch)

        _, loss_dict = self.common_step(batch, mode="train")

        with torch.inference_mode():
            self.log_dict_with_prefix(
                    loss_dict,
                    "train",
                    batch_size=batch["audio"]["mixture"].shape[0]
            )

        loss_dict["loss"] *= self.loss_adjustment

        return loss_dict

    def on_train_batch_end(
            self, outputs: STEP_OUTPUT, batch: BatchedDataDict, batch_idx: int
    ) -> None:

        metric_dict = self.compute_metrics()
        self.log_dict_with_prefix(metric_dict, "train")
        self.reset_metrics()

    def validation_step(
            self,
            batch: BatchedDataDict,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Dict[str, Any]:

        with torch.inference_mode():
            curr_val_prefix = f"val{dataloader_idx}" if dataloader_idx > 0 else "val"

            if curr_val_prefix != self.val_prefix:
                # print(f"Switching to validation dataloader {dataloader_idx}")
                if self.val_prefix is not None:
                    self._on_validation_epoch_end()
                self.val_prefix = curr_val_prefix
            _, loss_dict = self.common_step(batch, mode="val")

            self.log_dict_with_prefix(
                    loss_dict,
                    self.val_prefix,
                    batch_size=batch["audio"]["mixture"].shape[0],
                    prog_bar=True,
                    add_dataloader_idx=False
            )

        return loss_dict

    def on_validation_epoch_end(self) -> None:
        self._on_validation_epoch_end()

    def _on_validation_epoch_end(self) -> None:
        metric_dict = self.compute_metrics()
        self.log_dict_with_prefix(metric_dict, self.val_prefix, prog_bar=True,
                    add_dataloader_idx=False)
        # self.logger.save()
        # print(self.val_prefix, "Validation metrics:", metric_dict)
        self.reset_metrics()


    def old_predtest_step(
            self,
            batch: BatchedDataDict,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Tuple[BatchedDataDict, OutputType]:

        audio_batch = batch["audio"]["mixture"]
        track_batch = batch.get("track", ["" for _ in range(len(audio_batch))])

        output_list_of_dicts = [
                self.fader(
                        audio[None, ...],
                        lambda a: self.test_forward(a, track)
                )
                for audio, track in zip(audio_batch, track_batch)
        ]

        output_dict_of_lists = defaultdict(list)

        for output_dict in output_list_of_dicts:
            for stem, audio in output_dict.items():
                output_dict_of_lists[stem].append(audio)

        output = {
                "audio": {
                        stem: torch.concat(output_list, dim=0)
                        for stem, output_list in output_dict_of_lists.items()
                }
        }

        return batch, output

    def predtest_step(
            self,
            batch: BatchedDataDict,
            batch_idx: int = -1,
            dataloader_idx: int = 0
    ) -> Tuple[BatchedDataDict, OutputType]:

        if getattr(self.model, "bypass_fader", False):
            batch, output = self.model(batch)
        else:
            audio_batch = batch["audio"]["mixture"]
            output = self.fader(
                audio_batch,
                lambda a: self.test_forward(a, "", batch=batch)
            )

        return batch, output

    def test_forward(
            self,
            audio: torch.Tensor,
            track: str = "",
            batch: BatchedDataDict = None
    ) -> torch.Tensor:

        if self.fader is None:
            self.attach_fader()

        cond = batch.get("condition", None)

        if cond is not None and cond.shape[0] == 1:
            cond = cond.repeat(audio.shape[0], 1)

        _, output = self.forward(
                {"audio": {"mixture": audio},
                 "track": track,
                 "condition": cond,
                }
        )  # TODO: support track properly

        return output["audio"]

    def on_test_epoch_start(self) -> None:
        self.attach_fader(force_reattach=True)

    def test_step(
            self,
            batch: BatchedDataDict,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        curr_test_prefix = f"test{dataloader_idx}"

        # print(batch["audio"].keys())

        if curr_test_prefix != self.test_prefix:
            # print(f"Switching to test dataloader {dataloader_idx}")
            if self.test_prefix is not None:
                self._on_test_epoch_end()
            self.test_prefix = curr_test_prefix

        with torch.inference_mode():
            _, output = self.predtest_step(batch, batch_idx, dataloader_idx)
            # print(output)
            self.update_metrics(batch, output, mode="test")

        return output

    def on_test_epoch_end(self) -> None:
        self._on_test_epoch_end()

    def _on_test_epoch_end(self) -> None:
        metric_dict = self.compute_metrics(mode="test")
        self.log_dict_with_prefix(metric_dict, self.test_prefix, prog_bar=True,
                    add_dataloader_idx=False)
        # self.logger.save()
        # print(self.test_prefix, "Test metrics:", metric_dict)
        self.reset_metrics()

    def predict_step(
            self,
            batch: BatchedDataDict,
            batch_idx: int = 0,
            dataloader_idx: int = 0,
            include_track_name: Optional[bool] = None,
            get_no_vox_combinations: bool = True,
            get_residual: bool = False,
            treat_batch_as_channels: bool = False,
            fs: Optional[int] = None,
    ) -> Any:
        assert self.predict_output_path is not None

        batch_size = batch["audio"]["mixture"].shape[0]

        if include_track_name is None:
            include_track_name = batch_size > 1

        with torch.inference_mode():
            batch, output = self.predtest_step(batch, batch_idx, dataloader_idx)
        print('Pred test finished...')
        torch.cuda.empty_cache()
        metric_dict = {}

        if get_residual:
            mixture = batch["audio"]["mixture"]
            extracted = sum([output["audio"][stem] for stem in output["audio"]])
            residual = mixture - extracted
            print(extracted.shape, mixture.shape, residual.shape)

            output["audio"]["residual"] = residual

        if get_no_vox_combinations:
            no_vox_stems = [
                    stem for stem in output["audio"] if
                    stem not in self._VOX_STEMS
            ]
            no_vox_combinations = chain.from_iterable(
                    combinations(no_vox_stems, r) for r in
                    range(2, len(no_vox_stems) + 1)
            )

            for combination in no_vox_combinations:
                combination_ = list(combination)
                output["audio"]["+".join(combination_)] = sum(
                        [output["audio"][stem] for stem in combination_]
                )

        if treat_batch_as_channels:
            for stem in output["audio"]:
                output["audio"][stem] = output["audio"][stem].reshape(
                        1, -1, output["audio"][stem].shape[-1]
                )
            batch_size = 1

        for b in range(batch_size):
            print("!!", b)
            for stem in output["audio"]:
                print(f"Saving audio for {stem} to {self.predict_output_path}")
                track_name = batch["track"][b].split("/")[-1]

                if batch.get("audio", {}).get(stem, None) is not None:
                    self.test_metrics[stem].reset()
                    metrics = self.test_metrics[stem](
                            batch["audio"][stem][[b], ...],
                            output["audio"][stem][[b], ...]
                    )
                    snr = metrics["snr"]
                    sisnr = metrics["sisnr"]
                    sdr = metrics["sdr"]
                    metric_dict[stem] = metrics
                    print(
                            track_name,
                            f"snr={snr:2.2f} dB",
                            f"sisnr={sisnr:2.2f}",
                            f"sdr={sdr:2.2f} dB",
                    )
                    filename = f"{stem} - snr={snr:2.2f}dB - sdr={sdr:2.2f}dB.wav"
                else:
                    filename = f"{stem}.wav"

                if include_track_name:
                    output_dir = os.path.join(
                            self.predict_output_path,
                            track_name
                    )
                else:
                    output_dir = self.predict_output_path

                os.makedirs(output_dir, exist_ok=True)

                if fs is None:
                    fs = self.fs

                ta.save(
                        os.path.join(output_dir, filename),
                        output["audio"][stem][b, ...].cpu(),
                        fs,
                )

        return metric_dict

    def get_stems(
            self,
            batch: BatchedDataDict,
            batch_idx: int = 0,
            dataloader_idx: int = 0,
            include_track_name: Optional[bool] = None,
            get_no_vox_combinations: bool = True,
            get_residual: bool = False,
            treat_batch_as_channels: bool = False,
            fs: Optional[int] = None,
    ) -> Any:
        assert self.predict_output_path is not None

        batch_size = batch["audio"]["mixture"].shape[0]

        if include_track_name is None:
            include_track_name = batch_size > 1

        with torch.inference_mode():
            batch, output = self.predtest_step(batch, batch_idx, dataloader_idx)
        torch.cuda.empty_cache()
        metric_dict = {}

        if get_residual:
            mixture = batch["audio"]["mixture"]
            extracted = sum([output["audio"][stem] for stem in output["audio"]])
            residual = mixture - extracted
            # print(extracted.shape, mixture.shape, residual.shape)

            output["audio"]["residual"] = residual

        if get_no_vox_combinations:
            no_vox_stems = [
                    stem for stem in output["audio"] if
                    stem not in self._VOX_STEMS
            ]
            no_vox_combinations = chain.from_iterable(
                    combinations(no_vox_stems, r) for r in
                    range(2, len(no_vox_stems) + 1)
            )

            for combination in no_vox_combinations:
                combination_ = list(combination)
                output["audio"]["+".join(combination_)] = sum(
                        [output["audio"][stem] for stem in combination_]
                )

        if treat_batch_as_channels:
            for stem in output["audio"]:
                output["audio"][stem] = output["audio"][stem].reshape(
                        1, -1, output["audio"][stem].shape[-1]
                )
            batch_size = 1

        result = {}
        for b in range(batch_size):
            for stem in output["audio"]:
                track_name = batch["track"][b].split("/")[-1]

                if batch.get("audio", {}).get(stem, None) is not None:
                    self.test_metrics[stem].reset()
                    metrics = self.test_metrics[stem](
                            batch["audio"][stem][[b], ...],
                            output["audio"][stem][[b], ...]
                    )
                    snr = metrics["snr"]
                    sisnr = metrics["sisnr"]
                    sdr = metrics["sdr"]
                    metric_dict[stem] = metrics
                    print(
                            track_name,
                            f"snr={snr:2.2f} dB",
                            f"sisnr={sisnr:2.2f}",
                            f"sdr={sdr:2.2f} dB",
                    )
                    filename = f"{stem} - snr={snr:2.2f}dB - sdr={sdr:2.2f}dB.wav"
                else:
                    filename = f"{stem}.wav"

                if include_track_name:
                    output_dir = os.path.join(
                            self.predict_output_path,
                            track_name
                    )
                else:
                    output_dir = self.predict_output_path

                os.makedirs(output_dir, exist_ok=True)

                if fs is None:
                    fs = self.fs

                result[stem] = output["audio"][stem][b, ...].cpu().numpy()

        return result

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = False
    ) -> Any:

        return super().load_state_dict(state_dict, strict=False)


    def set_predict_output_path(self, path: str) -> None:
        self.predict_output_path = path
        os.makedirs(self.predict_output_path, exist_ok=True)

        self.attach_fader()

    def attach_fader(self, force_reattach=False) -> None:
        if self.fader is None or force_reattach:
            self.fader = parse_fader_config(self.fader_config)
            self.fader.to(self.device)


    def log_dict_with_prefix(
            self,
            dict_: Dict[str, torch.Tensor],
            prefix: str,
            batch_size: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        self.log_dict(
                {f"{prefix}/{k}": v for k, v in dict_.items()},
                batch_size=batch_size,
                logger=True,
                sync_dist=True,
                **kwargs,
        )