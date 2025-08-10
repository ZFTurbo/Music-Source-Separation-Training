from functools import wraps
from packaging import version
from collections import namedtuple

import os
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

def _print_once(msg):
    printed = False
    @wraps(print)
    def inner():
        nonlocal printed
        if not printed:
            print(msg)
            printed = True
    return inner

try:
    from sageattention import sageattn
    _has_sage_attention = True
    # _print_sage_found = _print_once("SageAttention found. Will be used when flash=True.")
    # _print_sage_found()
except ImportError:
    _has_sage_attention = False
    _print_sage_not_found = _print_once("SageAttention not found. Will fall back to PyTorch SDPA (if available) or manual einsum.")
    _print_sage_not_found()

# helpers
def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

# main class
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False, # If True, attempts to use SageAttention or PyTorch SDPA
        scale = None
    ):
        super().__init__()
        self.scale = scale # Store the scale if needed for einsum path
        self.dropout = dropout # Store dropout if needed for einsum/SDPA path

        # Determine which attention mechanism to *try* first
        self.use_sage = flash and _has_sage_attention
        self.use_pytorch_sdpa = False
        self._sdpa_checked = False # Flag to check PyTorch version only once

        if flash and not self.use_sage:
            # Only consider PyTorch SDPA if Sage isn't available/chosen
            if not self._sdpa_checked:
                if version.parse(torch.__version__) >= version.parse('2.0.0'):
                    self.use_pytorch_sdpa = True
                    _print_sdpa_used = _print_once("Using PyTorch SDPA backend (FlashAttention-2, Memory-Efficient, or Math).")
                    _print_sdpa_used()
                else:
                     _print_fallback_einsum = _print_once("Flash attention requested but Pytorch < 2.0 and SageAttention not found. Falling back to einsum.")
                     _print_fallback_einsum()
                self._sdpa_checked = True

        # Dropout layer for manual einsum implementation ONLY
        # SDPA and SageAttention handle dropout differently (or not at all in Sage's base API)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension

        Input tensors q, k, v expected in shape: (batch, heads, seq_len, dim_head) -> HND layout
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # --- Priority 1: SageAttention ---
        if self.use_sage:
            # Assumes q, k, v are FP16/BF16 (handled by autocast upstream)
            # Assumes scale is handled internally by sageattn
            # Assumes dropout is NOT handled by sageattn kernel
            # is_causal=False based on how Attend is called in mel_band_roformer
            out = sageattn(q, k, v, tensor_layout='HND', is_causal=False)
            return out
            try:
                return out
                # print("Attempting SageAttention") # Optional: for debugging
                out = sageattn(q, k, v, tensor_layout='HND', is_causal=False)
                return out
            except Exception as e:
                print(f"SageAttention failed with error: {e}. Falling back.")
                self.use_sage = False # Don't try Sage again if it failed once
                # Decide fallback: Check if PyTorch SDPA is an option
                if not self._sdpa_checked:
                    if version.parse(torch.__version__) >= version.parse('2.0.0'):
                        self.use_pytorch_sdpa = True
                        _print_sdpa_fallback = _print_once("Falling back to PyTorch SDPA.")
                        _print_sdpa_fallback()
                    else:
                        _print_einsum_fallback = _print_once("Falling back to einsum.")
                        _print_einsum_fallback()
                    self._sdpa_checked = True


        # --- Priority 2: PyTorch SDPA ---
        if self.use_pytorch_sdpa:
             # Use PyTorch's Scaled Dot Product Attention (SDPA)
             # It handles scaling and dropout internally.
            try:
                # print("Attempting PyTorch SDPA") # Optional: for debugging
                # Let PyTorch choose the best backend (Flash V2, Mem Efficient, Math)
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=None, # Assuming no explicit mask needed here
                        dropout_p = self.dropout if self.training else 0.,
                        is_causal=False # Assuming not needed based on usage context
                    )
                return out
            except Exception as e:
                 print(f"PyTorch SDPA failed with error: {e}. Falling back to einsum.")
                 self.use_pytorch_sdpa = False # Fallback to einsum on error


        # Calculate scale
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn) # Apply dropout ONLY in einsum path

        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
