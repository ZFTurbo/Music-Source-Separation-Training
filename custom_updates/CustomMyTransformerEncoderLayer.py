import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomMultiheadAttention import MultiheadAttention


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        group_norm=0,
        norm_first=False,
        norm_out=False,
        layer_norm_eps=1e-5,
        layer_scale=False,
        init_values=1e-4,
        device=None,
        dtype=None,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        auto_sparsity=False,
        sparsity=0.95,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        if sparse:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0,
            )
            self.__setattr__("src_mask", torch.zeros(1, 1))
            self.mask_random_seed = mask_random_seed

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        device = src.device
        x = src
        T, B, C = x.shape
        if self.sparse and not self.auto_sparsity:
            assert src_mask is None
            src_mask = self.src_mask
            if src_mask.shape[-1] != T:
                src_mask = get_mask(
                    T,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("src_mask", src_mask)

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x
    

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
    

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def get_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    Return a SparseCSRTensor mask that is a combination of elementary masks
    mask_type can be a combination of multiple masks: for instance "diag_jmask_random"
    """
    from xformers.sparse import SparseCSRTensor
    # create a list
    mask_types = mask_type.split("_")

    all_masks = [
        get_elementary_mask(
            T1,
            T2,
            mask,
            sparse_attn_window,
            global_window,
            mask_random_seed,
            sparsity,
            device,
        )
        for mask in mask_types
    ]

    final_mask = torch.stack(all_masks).sum(axis=0) > 0

    return SparseCSRTensor.from_dense(final_mask[None])


def get_elementary_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    When the input of the Decoder has length T1 and the output T2
    The mask matrix has shape (T2, T1)
    """
    assert mask_type in ["diag", "jmask", "random", "global"]

    if mask_type == "global":
        mask = torch.zeros(T2, T1, dtype=torch.bool)
        mask[:, :global_window] = True
        line_window = int(global_window * T2 / T1)
        mask[:line_window, :] = True

    if mask_type == "diag":

        mask = torch.zeros(T2, T1, dtype=torch.bool)
        rows = torch.arange(T2)[:, None]
        cols = (
            (T1 / T2 * rows + torch.arange(-sparse_attn_window, sparse_attn_window + 1))
            .long()
            .clamp(0, T1 - 1)
        )
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))

    elif mask_type == "jmask":
        mask = torch.zeros(T2 + 2, T1 + 2, dtype=torch.bool)
        rows = torch.arange(T2 + 2)[:, None]
        t = torch.arange(0, int((2 * T1) ** 0.5 + 1))
        t = (t * (t + 1) / 2).int()
        t = torch.cat([-t.flip(0)[:-1], t])
        cols = (T1 / T2 * rows + t).long().clamp(0, T1 + 1)
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))
        mask = mask[1:-1, 1:-1]

    elif mask_type == "random":
        gene = torch.Generator(device=device)
        gene.manual_seed(mask_random_seed)
        mask = (
            torch.rand(T1 * T2, generator=gene, device=device).reshape(T2, T1)
            > sparsity
        )

    mask = mask.to(device)
    return mask