import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from einops import rearrange
import random
import numpy as np
import math

from CustomCrossTransformerEncoderLayer import CrossTransformerEncoderLayer
from CustomMyTransformerEncoderLayer import MyTransformerEncoderLayer

class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: tp.Optional[float] = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
        sparse_self_attn: bool = False,
        sparse_cross_attn: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        auto_sparsity: bool = False,
        sparsity: float = 0.95,
    ):
        super().__init__()
        """
        """
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "mask_type": mask_type,
            "mask_random_seed": mask_random_seed,
            "sparse_attn_window": sparse_attn_window,
            "global_window": global_window,
            "sparsity": sparsity,
            "auto_sparsity": auto_sparsity,
            "batch_first": True,
        }

        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update({
            "sparse": sparse_self_attn,
        })
        kwargs_cross_encoder = dict(kwargs_common)
        kwargs_cross_encoder.update({
            "sparse": sparse_cross_attn,
        })

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:

                self.layers.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_classic_encoder)
                )

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))

                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_cross_encoder)
                )

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        )  # (1, C, Fr, T1)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  # now T2, B, C
        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            if self.training:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=self.cape_augment,
                    max_global_shift=self.cape_glob_loc_scale[0],
                    max_local_shift=self.cape_glob_loc_scale[1],
                    max_scale=self.cape_glob_loc_scale[2],
                )
            else:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=False,
                )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb

    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None:
            group["lr"] = self.lr
        return group
    

class ScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data *= scale / boost
        self.boost = boost

    @property
    def weight(self):
        return self.embedding.weight * self.boost

    def forward(self, x):
        return self.embedding(x) * self.boost
    

class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1:: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


def create_sin_embedding(
    length: int, dim: int, shift: int = 0, device="cpu", max_period=10000
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,  # True during training
    max_global_shift: float = 0.0,  # delta max
    max_local_shift: float = 0.0,  # epsilon max
    max_scale: float = 1.0,
    device: str = "cpu",
    max_period: float = 10000.0,
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1)  # (length, 1, 1)
    pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
    if mean_normalize:
        pos -= torch.nanmean(pos, dim=0, keepdim=True)

    if augment:
        delta = np.random.uniform(
            -max_global_shift, +max_global_shift, size=[1, batch_size, 1]
        )
        delta_local = np.random.uniform(
            -max_local_shift, +max_local_shift, size=[length, batch_size, 1]
        )
        log_lambdas = np.random.uniform(
            -np.log(max_scale), +np.log(max_scale), size=[1, batch_size, 1]
        )
        pos = (pos + delta + delta_local) * np.exp(log_lambdas)

    pos = pos.to(device)

    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    ).float()