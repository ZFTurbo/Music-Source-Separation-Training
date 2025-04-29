import torch
import torch.nn as nn

class CustomMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        auto_sparsity=None,
    ):
        super().__init__()
        assert auto_sparsity is not None, "sanity check"
        self.num_heads = num_heads
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.proj_drop = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.auto_sparsity = auto_sparsity

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):

        if not self.batch_first:  # N, B, C
            query = query.permute(1, 0, 2)  # B, N_q, C
            key = key.permute(1, 0, 2)  # B, N_k, C
            value = value.permute(1, 0, 2)  # B, N_k, C
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        q = (
            self.q(query)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q.flatten(0, 1)
        k = (
            self.k(key)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = k.flatten(0, 1)
        v = (
            self.v(value)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = v.flatten(0, 1)

        if self.auto_sparsity:
            assert attn_mask is None
            x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        else:
            # Example frequency bias: boost center frequencies
            T_q = q.shape[-2]
            T_k = k.shape[-2]
            device = q.device
            freq_range = torch.linspace(0, 1, T_k, device=device)
            bias_vector = -((freq_range - 0.5)**2) * 10  # Mid frequencies boosted
            freq_bias = bias_vector.unsqueeze(0).unsqueeze(1)  # [1, 1, T_k]
            freq_bias = freq_bias.expand(q.shape[0], T_q, T_k)  # [B*H, T_q, T_k]

            x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop, freq_bias=freq_bias)

        #if self.auto_sparsity:
        #    assert attn_mask is None
        #    x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        #else:
        #    x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop)
        
        
        
        x = x.reshape(B, self.num_heads, N_q, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x, None

def dynamic_sparse_attention(query, key, value, sparsity, infer_sparsity=True, attn_bias=None):
    # assert False, "The code for the custom sparse kernel is not ready for release yet."
    from xformers.ops import find_locations, sparse_memory_efficient_attention
    n_hashes = 32
    proj_size = 4
    query, key, value = [x.contiguous() for x in [query, key, value]]
    with torch.no_grad():
        R = torch.randn(1, query.shape[-1], n_hashes, proj_size // 2, device=query.device)
        bucket_query = _compute_buckets(query, R)
        bucket_key = _compute_buckets(key, R)
        row_offsets, column_indices = find_locations(
            bucket_query, bucket_key, sparsity, infer_sparsity)
    return sparse_memory_efficient_attention(
        query, key, value, row_offsets, column_indices, attn_bias)

#def scaled_dot_product_attention(q, k, v, att_mask, dropout):
#    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
#    att = dropout(att)
#    y = att @ v
#    return y

def scaled_dot_product_attention(q, k, v, att_mask, dropout, freq_bias=None):
    att = scaled_query_key_softmax(q, k, att_mask=att_mask, freq_bias=freq_bias)
    att = dropout(att)
    y = att @ v
    return y

#def scaled_query_key_softmax(q, k, att_mask):
#    from xformers.ops import masked_matmul
#    q = q / (k.size(-1)) ** 0.5
#    att = masked_matmul(q, k.transpose(-2, -1), att_mask)
#    att = torch.nn.functional.softmax(att, -1)
#    return att

def scaled_query_key_softmax(q, k, att_mask, freq_bias=None):
    from xformers.ops import masked_matmul
    q = q / (k.size(-1)) ** 0.5
    att = masked_matmul(q, k.transpose(-2, -1), att_mask)

    if freq_bias is not None:
        att += freq_bias  # shape should match att: [B * num_heads, T_q, T_k]

    att = torch.nn.functional.softmax(att, -1)
    return att


def _compute_buckets(x, R):
    qq = torch.einsum('btf,bfhi->bhti', x, R)
    qq = torch.cat([qq, -qq], dim=-1)
    buckets = qq.argmax(dim=-1)

    return buckets.permute(0, 2, 1).byte().contiguous()