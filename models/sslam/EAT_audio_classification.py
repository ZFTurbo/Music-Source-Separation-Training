# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass,field
from enum import Enum, auto
from typing import Any, Optional
from omegaconf import MISSING # II, open_dict removed as they are fairseq specific
# from fairseq import checkpoint_utils, tasks # Fairseq specific
# from fairseq.dataclass import FairseqDataclass # Fairseq specific
# from fairseq.models import BaseFairseqModel, register_model # Fairseq specific
# from fairseq.tasks import FairseqTask # Fairseq specific

# placeholder for mae imports, adjust if needed
from .mae import interpolate_pos_embed 
from .mae import get_2d_sincos_pos_embed_flexible

logger = logging.getLogger(__name__)


# EAT utilize cls token for prediction in most downstream tasks
class PredictionMode(Enum):
    MEAN_POOLING = auto()
    CLS_TOKEN = auto()
    LIN_SOFTMAX = auto()

# we follow the work of data2vec 2.0 on image modality and Audio-MAE in EAT 
@dataclass
class MaeImageClassificationConfig: # Removed FairseqDataclass
    model_path: Optional[str] = None # Changed MISSING to None
    no_pretrained_weights: bool = False
    linear_classifier: bool = False
    num_classes: int = 1000
    mixup: float = 0.0
    cutmix: float = 0.0
    label_smoothing: float = 0.0

    drop_path_rate: float = 0.1
    layer_decay: float = 0.65

    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.0
    mixup_mode: str = "batch"

    pretrained_model_args: Any = None # This will likely need refactoring
    data: Optional[str] = None # Removed II("task.data")

    norm_eps: Optional[float] = None

    remove_alibi: bool = False

    # regularization overwrites
    encoder_dropout: float = 0
    post_mlp_drop: float = 0
    attention_dropout: float = 0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0

    prenet_layerdrop: float = 0
    prenet_dropout: float = 0

    use_fc_norm: bool = True
    prediction_mode: PredictionMode = PredictionMode.CLS_TOKEN

    no_decay_blocks: bool = True

    # settings for specific downstream task
    audio_mae: bool = field(default=False, metadata={"help": "if true, the task is to realize audio classification"})
    esc50_eval: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on esc50 dataset"})
    spcv2_eval: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on speech command v2 dataset"})
    target_length: int = field(default=1024,metadata={"help": "This setting will pad the input sequence will zeros."})

    # specaug for specific downstream task
    specaug: bool = field(default=False, metadata={"help": "if true, use the specaug technique (frame and frequency masked 30%)"})
    freqm: int = field(default=25, metadata={"help": "the mask ratio of frequency dimension in audio spectrogram by default"})
    timem: int = field(default=200, metadata={"help": "the mask ratio of time dimension in audio spectrogram by default"})
    mask_ratio: float = field(default=0.0, metadata={"help": "the mask ratio of both time and freq "})
    
# This function might be needed if layer decay is kept, but its context from BEiT/Fairseq needs review
def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


# @register_model("mae_image_classification", dataclass=MaeImageClassificationConfig) # Removed Fairseq decorator
class MaeImageClassificationModel(nn.Module): # Changed BaseFairseqModel to nn.Module
    def __init__(self, cfg: MaeImageClassificationConfig, pretrained_model_path: Optional[str] = None): # Modified signature
        super().__init__()
        self.cfg = cfg
        
        # --- Start of commented out Fairseq-dependent initialization ---
        # self.audio_mae = self.cfg.audio_mae
        # self.esc50_eval = self.cfg.esc50_eval
        # self.spcv2_eval = self.cfg.spcv2_eval
        # self.target_length = self.cfg.target_length

        # if cfg.pretrained_model_args is None:
        #     # This part is heavily Fairseq dependent for checkpoint and config loading
        #     # state = checkpoint_utils.load_checkpoint_to_cpu(cfg.model_path, {}) 
        #     # pretrained_args = state.get("cfg", None)
        #     # ... (rest of Fairseq config manipulation) ...
        #     pass # Placeholder
        # else:
        #     # state = None
        #     # pretrained_args = cfg.pretrained_model_args
        #     pass # Placeholder

        # # task = tasks.setup_task(pretrained_args.task) # Fairseq specific
        # # model = task.build_model(pretrained_args.model, from_checkpoint=True) # Fairseq specific
        
        # self.model = None # Placeholder for the actual underlying model (e.g., Data2Vec)
        # logger.warning("MaeImageClassificationModel: Underlying model initialization is currently stubbed out due to Fairseq dependencies.")


        # self.d2v_multi = "data2vec_multi" # in pretrained_args.model._name # Needs actual model name
        # self.linear_classifier = cfg.linear_classifier

        # # if state is not None and not cfg.no_pretrained_weights:
        #     # interpolate_pos_embed(model, state) # Fairseq specific checkpoint handling
        #     # model.load_state_dict(state["model"], strict=True) # Standard PyTorch, but state dict keys might differ
        #     # logger.info(f"Loaded pretrained weights from {cfg.model_path}")


        # # if self.d2v_multi:
        # #     self.model.remove_pretraining_modules(modality="image") # Model specific method
        # # else:
        # #     self.model.remove_pretraining_modules() # Model specific method

        # # if self.linear_classifier:
        # #     self.model.requires_grad_(False)

        # self.fc_norm = None
        # # if self.cfg.use_fc_norm and hasattr(pretrained_args.model, 'embed_dim'): # Check if embed_dim is available
        # #     self.fc_norm = nn.LayerNorm(pretrained_args.model.embed_dim, eps=1e-6)
        # #     nn.init.constant_(self.fc_norm.bias, 0)
        # #     nn.init.constant_(self.fc_norm.weight, 1.0)
        # # else:
        # #     logger.warning("fc_norm cannot be created without embed_dim from pretrained_args.model")


        # # if hasattr(pretrained_args.model, 'embed_dim'):
        # #    self.head = nn.Linear(pretrained_args.model.embed_dim, cfg.num_classes)
        # #    nn.init.trunc_normal_(self.head.weight, std=0.02)
        # #    nn.init.constant_(self.head.bias, 0)
        # # else:
        # self.head = None # Placeholder
        # logger.warning("Head cannot be created without embed_dim from pretrained_args.model")


        # self.mixup_fn = None
        # self.specaug = cfg.specaug
        # self.mask_ratio = cfg.mask_ratio

        # # if cfg.mixup > 0 or cfg.cutmix > 0:
        #     # from ..utils.mixup import Mixup # This import might need adjustment based on project structure
        #     # self.mixup_fn = Mixup(...)
        #     # logger.info("Mixup enabled.")
            
        # if self.specaug:
        #     self.freqm = cfg.freqm
        #     self.timem = cfg.timem
            
        #     if self.mask_ratio != 0.0:
        #         self.freqm = 128 * self.mask_ratio # Assuming 128 is a relevant dimension (e.g. n_mels)
        #         self.timem = self.target_length * self.mask_ratio
        #     logger.info(f"SpecAug enabled: freqm={self.freqm}, timem={self.timem}")
        # --- End of commented out Fairseq-dependent initialization ---
        
        # Minimal placeholder initialization
        self.model = nn.Identity() # Placeholder
        self.head = nn.Identity()  # Placeholder
        self.fc_norm = None
        self.linear_classifier = cfg.linear_classifier
        self.mixup_fn = None # Add mixup import and instantiation if needed
        self.specaug = cfg.specaug
        if self.specaug:
            self.freqm = cfg.freqm
            self.timem = cfg.timem
            self.mask_ratio = cfg.mask_ratio
            if self.mask_ratio != 0.0 and hasattr(cfg, 'target_length'): # Assuming target_length is available
                # This might need more context on how freqm/timem are derived
                # For now, just copying logic but it might not be correct without full context
                self.freqm = int(128 * self.mask_ratio) 
                self.timem = int(cfg.target_length * self.mask_ratio)


    # @classmethod
    # def build_model(cls, cfg: MaeImageClassificationConfig, task=FairseqTask): # Removed FairseqTask
    #     """Build a new model instance."""
    #     # assert hasattr(task, "labels"), f"Task {task} must have an attribute 'labels'" # Fairseq specific
    #     # return cls(cfg) # Simplified instantiation, task no longer needed here
    #     pass


    def forward(
        self,
        imgs,
        label=None,
    ):
        # labels = label
        # if self.training and self.mixup_fn is not None and labels is not None: 
        #     imgs, labels = self.mixup_fn(imgs, labels)
            
        # if self.training and self.specaug:
        #     imgs = self.spectrogram_augment(imgs)

        # if self.linear_classifier:
        #     with torch.no_grad():
        #         x = self.model_forward(imgs)
        # else:
        #     x = self.model_forward(imgs)

        # # different prediction mode
        # if self.cfg.prediction_mode == PredictionMode.MEAN_POOLING:
        #     x = x.mean(dim=1)
        # elif self.cfg.prediction_mode == PredictionMode.CLS_TOKEN:
        #     x = x[:, 0]
        # elif self.cfg.prediction_mode == PredictionMode.LIN_SOFTMAX:
        #     dtype = x.dtype
        #     x = F.logsigmoid(x.float())
        #     x = torch.logsumexp(x + x, dim=1) - torch.logsumexp(x + 1e-6, dim=1)
        #     x = x.clamp(max=0)
        #     x = x - torch.log(-(torch.expm1(x)))
        #     x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
        #     x = x.to(dtype=dtype)
        # else:
        #     raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")

        # # layer norm and project
        # if self.fc_norm is not None:
        #     x = self.fc_norm(x)

        # x = self.head(x)

        # if labels is None:
        #     return x
        
        # x = torch.nan_to_num(x)
        
        # # logs for different downstream task    ESC-50 && SPC-2 -> single label    AS (AS2M,AS20K) -> multilabel
        # if not self.cfg.audio_mae or (self.cfg.audio_mae and (self.cfg.esc50_eval or self.cfg.spcv2_eval )): # Adjusted cfg access
        #     if self.training and self.mixup_fn is not None and not self.cfg.spcv2_eval: # Adjusted cfg access
        #         loss = -labels * F.log_softmax(x.float(), dim=-1)
                
        #     elif self.mixup_fn is not None and self.cfg.spcv2_eval: # Adjusted cfg access
        #         loss = F.binary_cross_entropy_with_logits(
        #             x, labels.float(), reduction="none"
        #         )
                
        #     else:
        #         loss = F.cross_entropy(
        #             x.float(),
        #             labels,
        #             label_smoothing=self.cfg.label_smoothing if self.training else 0,
        #             reduction="none",
        #         )

        #     result = {
        #         "losses": {"regression": loss},
        #         "sample_size": imgs.size(0),
        #     }

        #     if not self.training:
        #         with torch.no_grad():
        #             pred = x.argmax(-1)
        #             labels = labels.argmax(-1)
        #             correct = (pred == labels).sum()
        #             result["correct"] = correct
                    
        # else:
        #     loss = F.binary_cross_entropy_with_logits(
        #         x, labels.float(), reduction="none"
        #     )

        #     result = {
        #         "losses": {
        #             "main": loss,
        #         },
        #         "sample_size": labels.sum(),
        #     }

        #     if not self.training:
        #         result["_predictions"] = torch.sigmoid(x) 
        #         result["_targets"] = labels


        # return result
        pass # Body commented out as it depends on self.model and self.head being properly initialized

    def model_forward(self, imgs):
        # if self.d2v_multi: # Relies on self.model being the D2V model
        #     x = self.model.extract_features(
        #         imgs,
        #         mode="IMAGE",
        #         mask=False,
        #         remove_extra_tokens=(
        #             self.cfg.prediction_mode != PredictionMode.CLS_TOKEN
        #         ),
        #     )["x"]
        # else: # Relies on self.model being a non-D2V MAE model
        #     x = self.model(imgs, predictions_only=True)
        #     if (
        #         "no_cls" not in self.model.cfg or not self.model.cfg.no_cls # model.cfg access
        #     ) and not self.cfg.prediction_mode == PredictionMode.CLS_TOKEN:
        #         x = x[:, 1:]
        # return x
        pass # Body commented out

    # specaug
    def spectrogram_augment(self,spec):
        freq_masking = torchaudio.transforms.FrequencyMasking(self.freqm,iid_masks=True)
        time_masking = torchaudio.transforms.TimeMasking(self.timem,iid_masks=True)
        spec_ = spec.transpose(2,3)
        input_with_freq_mask = freq_masking(spec_)
        input_with_time_freq_mask = time_masking(input_with_freq_mask)
        input_with_time_freq_mask = torch.transpose(input_with_time_freq_mask, 2, 3)
        return input_with_time_freq_mask