import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomMultiheadAttention import MultiheadAttention

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        sparsity=0.95,
        auto_sparsity=None,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

        if sparse:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0)
            if not auto_sparsity:
                self.__setattr__("mask", torch.zeros(1, 1))
                self.mask_random_seed = mask_random_seed

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        device = q.device
        T, B, C = q.shape
        S, B, C = k.shape
        if self.sparse and not self.auto_sparsity:
            assert mask is None
            mask = self.mask
            if mask.shape[-1] != S or mask.shape[-2] != T:
                mask = get_mask(
                    S,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("mask", mask)

        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    # self-attention block
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    

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