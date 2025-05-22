# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
from omegaconf import II # Will be removed from field defaults
from enum import Enum, auto
# from fairseq.modules import EMAModule, EMAModuleConfig # Fairseq specific
# from fairseq.dataclass import FairseqDataclass # Fairseq specific
# from fairseq.models import BaseFairseqModel, register_model # Fairseq specific

from .base import (
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder, 
    get_annealed_rate,
)

from .modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from .images import (
    D2vImageConfig,
    ImageEncoder,
)

logger = logging.getLogger(__name__)

# we follow the work of data2vec 2.0 on image modality and Audio-MAE in EAT 
class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()

@dataclass
class D2vModalitiesConfig: # Removed FairseqDataclass
    image: D2vImageConfig = D2vImageConfig() # Assuming D2vImageConfig is not Fairseq specific or will be refactored
    
@dataclass
class Data2VecMultiConfig: # Removed FairseqDataclass

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 12
    
    # standard vision Transformer
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    # EAT averages all Transformer block output (12 layers in total) 
    average_top_k_layers: int = field(
        default=12, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    # clone batch for multi-mask strategy
    clone_batch: int = 16

    # normalization for teacher Transformer layer output
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    # EMA settings
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    ema_anneal_end_step: int = 0 # Was II("optimization.max_update"), placeholder
    # ema_anneal_end_step: int = field(default=II("optimization.max_update")) # Corrected previous edit

    # In EAT, the Transformer encoder and the CNN encoder are both EMA updated
    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = 0 # Was II("optimization.max_update"), placeholder
    # max_update: int = field(default=II("optimization.max_update")) # Corrected previous edit


    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = 0 # Was II("common.seed"), placeholder
    # seed: int = field(default=II("common.seed")) # Corrected previous edit

    skip_ema: bool = False

    # d2v_loss is the frame-level loss while cls_loss is the utterance-level loss
    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False

    # the experiment of using dino loss instead of direct utterance loss (not included in our paper)
    utterance_level: bool = field(default=False, metadata={"help": "if true, we will add utterance-level loss to the total loss"})
    init_center_token_zero: bool = field(default=False, metadata={"help": "if true, we will initialize the centor token with zero vertors"})
    center_exp: float = field(default=0.9, metadata={"help": "this value control the exponent decay of center value's coefficient"})
    softmax_temperature_student: float = field(default=0.1, metadata={"help": "this value control the temperature of softmax function of student output in the dino loss"})
    softmax_temperature_teacher: float = field(default=0.05, metadata={"help": "this value control the temperature of softmax function in teacher output the dino loss"})


# @register_model("data2vec_multi", dataclass=Data2VecMultiConfig) # Removed Fairseq decorator
class Data2VecMultiModel(nn.Module): # Changed BaseFairseqModel to nn.Module
    # def make_modality_encoder( # This method is Fairseq specific due to `task`
    #     self,
    #     cfg: D2vModalityConfig,
    #     embed_dim: int,
    #     make_block: Callable[[float], nn.ModuleList],
    #     norm_layer: Callable[[int], nn.LayerNorm],
    #     layer_norm_first: bool,
    #     alibi_biases,
    #     task, # Fairseq task
    # ) -> ModalitySpecificEncoder:
    #     if cfg.type.value == Modality.IMAGE.value:
    #         enc_cls = ImageEncoder
    #     else:
    #         raise Exception(f"unsupported modality {cfg.type}")
    #     return enc_cls(
    #         cfg,
    #         embed_dim,
    #         make_block,
    #         norm_layer,
    #         layer_norm_first,
    #         alibi_biases,
    #         task, # Fairseq task
    #     )

    def __init__(self, cfg: Data2VecMultiConfig): # Simplified signature, removed modalities, skip_ema, task
        super().__init__()
        self.cfg = cfg
        # --- Start of commented out Fairseq-dependent initialization ---
        # self.modalities = modalities # Needs to be handled based on cfg
        # self.task = task # Fairseq specific

        # make_layer_norm = partial(
        #     nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        # )

        # def make_block(drop_path, dim=None, heads=None):
        #     return AltBlock(
        #         cfg.embed_dim if dim is None else dim,
        #         cfg.num_heads if heads is None else heads,
        #         cfg.mlp_ratio,
        #         qkv_bias=True,
        #         drop=cfg.encoder_dropout,
        #         attn_drop=cfg.attention_dropout,
        #         mlp_drop=cfg.activation_dropout,
        #         post_mlp_drop=cfg.post_mlp_drop,
        #         drop_path=drop_path,
        #         norm_layer=make_layer_norm,
        #         layer_norm_first=cfg.layer_norm_first,
        #         ffn_targets=not cfg.end_of_block_targets,
        #     )

        # self.alibi_biases = {}
        # self.modality_encoders = nn.ModuleDict()
        
        # # This loop depends on self.modalities and make_modality_encoder (which needs task)
        # # for mod in self.modalities:
        # #     mod_cfg = getattr(cfg.modalities, mod.name.lower())
        # #     enc = self.make_modality_encoder(
        # #         mod_cfg,
        # #         cfg.embed_dim,
        # #         make_block,
        # #         make_layer_norm,
        # #         cfg.layer_norm_first,
        # #         self.alibi_biases,
        # #         task, # Fairseq task
        # #     )
        # #     self.modality_encoders[mod.name] = enc

        # self.ema = None # Fairseq EMA module

        # self.average_top_k_layers = cfg.average_top_k_layers
        # self.loss_beta = cfg.loss_beta
        # self.loss_scale = cfg.loss_scale
        # self.utterance_level = cfg.utterance_level

        # self.dropout_input = nn.Dropout(cfg.dropout_input)

        # dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)
        # self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])


        # self.norm = None
        # if cfg.layer_norm_first:
        #     self.norm = make_layer_norm(cfg.embed_dim)

        # if self.cfg.mae_init:
        #     self.apply(self._init_weights)
        # else:
        #     # from fairseq.modules.transformer_sentence_encoder import init_bert_params # Fairseq specific
        #     # self.apply(init_bert_params) # Fairseq specific
        #     pass # Placeholder for non-Fairseq initialization

        # # for mod_enc in self.modality_encoders.values():
        # #     mod_enc.reset_parameters() # Depends on modality_encoders

        # # if not skip_ema: # skip_ema was a __init__ param
        #     # self.ema = self.make_ema_teacher(cfg.ema_decay) # Fairseq specific
        #     # self.shared_decoder = (
        #     #     Decoder1d(cfg.shared_decoder, cfg.embed_dim)
        #     #     if self.cfg.shared_decoder is not None
        #     #     else None
        #     # )
        #     # if self.shared_decoder is not None:
        #     #     self.shared_decoder.apply(self._init_weights)
        #     # self.recon_proj = None
        #     # if cfg.recon_loss > 0:
        #     #     self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim//3)
        #     # self.cls_proj = None
        #     # if cfg.utterance_level:
        #     #     self.cls_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        #     pass


        # # Parameter specific optimizer settings - this might be Fairseq specific or need careful review
        # # for pn, p in self.named_parameters():
        # #     if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
        # #         p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
        # #     if cfg.decoder_group and "decoder" in pn:
        # #         p.param_group = "decoder"
        
        # self.center = None
        # # if self.utterance_level:
        # #     self.center_exp = cfg.center_exp
        # #     self.soft_tem_s = cfg.softmax_temperature_student
        # #     self.soft_tem_t = cfg.softmax_temperature_teacher
        # #     self.center = nn.Parameter(
        # #             torch.zeros(1, 1, cfg.embed_dim, requires_grad=False)
        # #         )
        # #     if not cfg.init_center_token_zero:
        # #         nn.init.normal_(self.center)
        # #     elif self.center.size(1) > 1:
        # #         nn.init.normal_(self.center[:, 1:])

        # self.num_updates = 0
        # --- End of commented out Fairseq-dependent initialization ---
        
        # Placeholder attributes
        self.modality_encoders = nn.ModuleDict() # Needs to be populated based on cfg.modalities
        self.blocks = nn.ModuleList() # Needs to be populated
        self.norm = None
        self.dropout_input = None
        self.ema = None
        self.shared_decoder = None
        self.recon_proj = None
        self.cls_proj = None
        self.center = None
        self.num_updates = 0
        
        logger.warning("Data2VecMultiModel: Initialization is currently stubbed out due to Fairseq dependencies.")


    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # @torch.no_grad()
    # def make_ema_teacher(self, ema_decay): # Fairseq EMA specific
    #     # ema_config = EMAModuleConfig(
    #     #     ema_decay=ema_decay,
    #     #     ema_fp32=True,
    #     #     log_norms=self.cfg.log_norms,
    #     #     add_missing_params=False,
    #     # )
    #     # model_copy = self.make_target_model()
    #     # return EMAModule(
    #     #     model_copy,
    #     #     ema_config,
    #     #     copy_model=False,
    #     # )
    #     pass

    # def make_target_model(self): # Fairseq EMA specific
    #     # logger.info("making target model")
    #     # model_copy = Data2VecMultiModel(
    #     #     self.cfg, self.modalities, skip_ema=True, task=self.task # self.modalities and self.task are problematic
    #     # )
    #     # if self.cfg.ema_encoder_only:
    #     #     model_copy = model_copy.blocks
    #     #     for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
    #     #         p_t.data.copy_(p_s.data)
    #     # else:
    #     #     for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
    #     #         p_t.data.copy_(p_s.data)
    #     #     for mod_enc in model_copy.modality_encoders.values():
    #     #         mod_enc.decoder = None
    #     #         if not mod_enc.modality_cfg.ema_local_encoder:
    #     #             mod_enc.local_encoder = None
    #     #             mod_enc.project_features = None
    #     # model_copy.requires_grad_(False)
    #     # return model_copy
    #     pass

    # def set_num_updates(self, num_updates): # Fairseq specific training loop integration
    #     # super().set_num_updates(num_updates) # BaseFairseqModel method
    #     # if self.ema is not None and (
    #     #     (self.num_updates == 0 and num_updates > 1)
    #     #     or self.num_updates >= num_updates
    #     # ):
    #     #     pass
    #     # elif self.training and self.ema is not None:
    #     #     ema_weight_decay = None
    #     #     if self.cfg.ema_decay != self.cfg.ema_end_decay:
    #     #         if num_updates >= self.cfg.ema_anneal_end_step:
    #     #             decay = self.cfg.ema_end_decay
    #     #         else:
    #     #             decay = get_annealed_rate(
    #     #                 self.cfg.ema_decay,
    #     #                 self.cfg.ema_end_decay,
    #     #                 num_updates,
    #     #                 self.cfg.ema_anneal_end_step,
    #     #             )
    #     #         self.ema.set_decay(decay, weight_decay=ema_weight_decay)
    #     #     if self.ema.get_decay() < 1:
    #     #         self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)
    #     self.num_updates = num_updates # Keep this part if num_updates is used elsewhere directly

    # def state_dict(self, destination=None, prefix="", keep_vars=False): # Potentially override for EMA
    #     # state = super().state_dict(destination, prefix, keep_vars)
    #     # if self.ema is not None:
    #     #     state[prefix + "_ema"] = self.ema.fp32_params
    #     # return state
    #     return super().state_dict(destination, prefix, keep_vars)


    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs): # Potentially override for EMA
    #     # k = prefix + "_ema"
    #     # if self.ema is not None:
    #     #     assert k in state_dict
    #     #     self.ema.restore(state_dict[k], True)
    #     #     del state_dict[k]
    #     # elif k in state_dict:
    #     #     del state_dict[k]
    #     # return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    #     return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


    # @classmethod
    # def build_model(cls, cfg: Data2VecMultiConfig, task=None): # Fairseq specific factory
    #     """Build a new model instance."""
    #     # if task is None or not hasattr(task, "supported_modalities"):
    #     #     modalities = (
    #     #         [cfg.supported_modality]
    #     #         if cfg.supported_modality is not None
    #     #         else [
    #     #             Modality.AUDIO,
    #     #             Modality.IMAGE,
    #     #             Modality.TEXT,
    #     #         ]
    #     #     )
    #     # else:
    #     #     modalities = task.supported_modalities
    #     # return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema) # cls now has different signature
    #     pass

    def forward(
        self,
        source, # Assuming source is the primary input (e.g. spectrogram)
        target=None, # Usually for training
        id=None, # For mask_seeds, might not be needed without Fairseq task
        mode: Optional[str] = None, # e.g. "IMAGE" or "AUDIO"
        padding_mask=None,
        mask: bool =True, # Whether to apply masking (e.g. for MAE-style pretraining)
        features_only: bool =False,
        force_remove_masked: bool =False,
        remove_extra_tokens: bool =True,
        precomputed_mask=None, # Mask might need to be generated externally now
    ):
        # if mode is None:
        #     # This logic depends on cfg.supported_modality which might not be straightforwardly available
        #     # or might need to be passed explicitly if model supports multiple modalities.
        #     assert self.cfg.supported_modality is not None 
        #     mode = self.cfg.supported_modality.name # Assuming Modality enum has .name

        # if isinstance(mode, Modality): # This check might be redundant if mode is always str
        #     mode = mode.name
        
        # # feature_extractor = self.modality_encoders[mode] # Depends on self.modality_encoders init

        # # mask_seeds = None
        # # if id is not None: # id was Fairseq specific for distributed training
        #     # mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)


        # # extractor_out = feature_extractor( # This call is critical and depends on feature_extractor
        # #     source,
        # #     padding_mask,
        # #     mask,
        # #     remove_masked=not features_only or force_remove_masked,
        # #     clone_batch=self.cfg.clone_batch if not features_only else 1, # clone_batch for pretraining
        # #     mask_seeds=mask_seeds,
        # #     precomputed_mask=precomputed_mask,
        # # )
        
        # # x = extractor_out["x"]
        # # encoder_mask = extractor_out["encoder_mask"]
        # # masked_padding_mask = extractor_out["padding_mask"]
        # # masked_alibi_bias = extractor_out.get("alibi_bias", None)
        # # alibi_scale = extractor_out.get("alibi_scale", None)

        # # if self.dropout_input is not None:
        # #     x = self.dropout_input(x)
        
        # # layer_results = []
        # # for i, blk in enumerate(self.blocks): # Depends on self.blocks
        # #     if (
        # #         not self.training
        # #         or self.cfg.layerdrop == 0
        # #         or (np.random.random() > self.cfg.layerdrop)
        # #     ):
        # #         ab = masked_alibi_bias
        # #         if ab is not None and alibi_scale is not None:
        # #             scale = (
        # #                 alibi_scale[i]
        # #                 if alibi_scale.size(0) > 1
        # #                 else alibi_scale.squeeze(0)
        # #             )
        # #             ab = ab * scale.type_as(ab)

        # #         x, lr = blk(
        # #             x,
        # #             padding_mask=masked_padding_mask,
        # #             alibi_bias=ab,
        # #         )
        # #         if features_only:
        # #             layer_results.append(lr)

        # # if self.norm is not None: # Depends on self.norm
        # #     x = self.norm(x)

        # if features_only:
        #     # if remove_extra_tokens:
        #     #     # This logic depends on feature_extractor.modality_cfg
        #     #     x = x[:, feature_extractor.modality_cfg.num_extra_tokens :] 
        #     #     if masked_padding_mask is not None:
        #     #         masked_padding_mask = masked_padding_mask[
        #     #             :, feature_extractor.modality_cfg.num_extra_tokens :
        #     #         ]
        #     # return {
        #     #     "x": x, # x would be undefined here
        #     #     "padding_mask": masked_padding_mask, # undefined
        #     #     "layer_results": layer_results, # undefined (or empty)
        #     #     "mask": encoder_mask, # undefined
        #     # }
        #     pass


        # # The rest of the forward pass is for training (EMA, loss calculation) and is heavily Fairseq dependent
        # # For now, it's commented out. A refactored model would need a clear separation
        # # for feature extraction vs. training loss computation.

        # # xs = []
        # # if self.shared_decoder is not None:
        # #     dx = self.forward_decoder(x, feature_extractor, self.shared_decoder, encoder_mask)
        # #     xs.append(dx)
        # # if feature_extractor.decoder is not None: # Depends on feature_extractor
        # #     dx = self.forward_decoder(x, feature_extractor, feature_extractor.decoder, encoder_mask)
        # #     xs.append(dx)
        # #     orig_x = x
        # # assert len(xs) > 0


        # # EMA related logic heavily depends on Fairseq's EMAModule and training loop
        # # if self.ema is not None and self.training:
        #     # p = next(self.ema.model.parameters())
        #     # ... (EMA model sync) ...
        #     # tm = self.ema.model
            
        #     # with torch.no_grad():
        #     #     tm.eval()
        #     #     ... (teacher model forward pass) ...
        #     #     y = self.make_targets(y, self.average_top_k_layers)
        #     #     orig_targets = y
        #     #     if self.cfg.clone_batch > 1:
        #     #         y = y.repeat_interleave(self.cfg.clone_batch, 0)
        #     #     masked = encoder_mask.mask.unsqueeze(-1)
        #     #     masked_b = encoder_mask.mask.bool()
        #     #     y = y[masked_b]
        #     #     if xs[0].size(1) == masked_b.size(1):
        #     #         xs = [x_i[masked_b] for x_i in xs]
        #     #     else:
        #     #         xs = [x_i.reshape(-1, x_i.size(-1)) for x_i in xs]
                
        #     # sample_size = masked.sum().long()
        #     # result = {"losses": {}, "sample_size": sample_size}

        #     # ... (loss calculations: cls_loss, recon_loss, d2v_loss) ...
        #     # return result
        # # else:
        #     # If not training or no EMA, this model was designed for feature extraction if features_only=True
        #     # but we already returned if features_only was true and x was defined.
        #     # If features_only=False and not training, the original code would error or behave unexpectedly.
        #     # For a pure nn.Module, we'd typically just return the output of the network.
        #     # return x # x would be from student if EMA is not used.

        logger.warning("Data2VecMultiModel.forward: Body is stubbed out.")
        return source # Placeholder, just returning input

    def forward_decoder(
        self,
        x, # Output from encoder
        feature_extractor, # Modality specific encoder, contains decoder_input method
        decoder, # The actual decoder module
        mask_info, # Contains mask information
    ):
        # x_input_for_decoder = feature_extractor.decoder_input(x, mask_info) # This call is problematic
        # decoded_features = decoder(*x_input_for_decoder)
        # return decoded_features
        pass # Commented out due to dependency on feature_extractor

    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss
    
    def dino_loss(self,s,t):
        t = t.detach()
        s = F.softmax(s/self.soft_tem_s,dim=1)
        t = F.softmax((t-self.center)/self.soft_tem_t,dim=1)
        return - (t * torch.log(s)).sum(dim=1).mean()
    
    # average top-k layers output from teacher model
    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        self.ema = None
        self.cfg.clone_batch = 1
        self.recon_proj = None        

        if not keep_decoder:
            self.shared_decoder = None

        # modality_name = modality.lower() if modality is not None else None # Pythonic way to handle None
        # for k in list(self.modality_encoders.keys()):
        #     if modality_name is not None and k.lower() != modality_name:
        #         del self.modality_encoders[k]
        #     else:
        #         # This assumes modality_encoders[k] has 'remove_pretraining_modules'
        #         if hasattr(self.modality_encoders[k], 'remove_pretraining_modules'):
        #             self.modality_encoders[k].remove_pretraining_modules(
        #                 keep_decoder=keep_decoder
        #             )
        #         # This assumes modality_encoders[k] might have a 'decoder' attribute
        #         if not keep_decoder and hasattr(self.modality_encoders[k], 'decoder'):
        #             self.modality_encoders[k].decoder = None
        pass # Commented out, depends on self.modality_encoders structure
