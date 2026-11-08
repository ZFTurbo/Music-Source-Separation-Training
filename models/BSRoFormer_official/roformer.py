"""RoformerBlock and constituent layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / rms.clamp(min=1e-8) * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout_p = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, rotary_emb=None):
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # (b, n, h*d) -> (b, h, n, d)
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        # (b, h, n, d) -> (b, n, h*d)
        out = attn_out.transpose(1, 2).contiguous().view(b, n, h * d)
        return self.out_drop(self.out_proj(out))


class Transformer(nn.Module):
    """Single-layer pre-norm transformer (RMSNorm -> Attention, RMSNorm -> FFN)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm_0 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.norm_1 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, rotary_emb=None):
        x = self.attn(self.norm_0(x), rotary_emb) + x
        x = self.ff(self.norm_1(x)) + x
        return x


class RoFormerBlock(nn.Module):
    """One depth layer: time-domain transformer + band-domain transformer."""

    def __init__(self, num_feature, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.transform_t = Transformer(num_feature, heads, dim_head, mlp_dim, dropout)
        self.transform_k = Transformer(num_feature, heads, dim_head, mlp_dim, dropout)
