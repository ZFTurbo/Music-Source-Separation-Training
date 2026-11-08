from __future__ import annotations

import math
from typing import Literal

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor
from torch.utils.checkpoint import checkpoint

from .frontend.subband.dsp import SubbandFilter, mel_linear_banks_triangle
from .frontend.subband.dsp.inference_optimized import InferenceSubbandFilter
from .frontend.subband.dsp.overlap_add import overlap_add

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MPSCompatibleLinear(nn.Linear):
    """Keep bias addition explicit on MPS, including virtualized Metal GPUs."""

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type == "mps" and self.bias is not None:
            output = F.linear(x, self.weight)
            return output + self.bias.to(output.dtype)
        return super().forward(x)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary 
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        w = torch.stack([torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1)  # (l, h/2, 2)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h)

        Outputs:
            out: (b, l, n, h)
        """
        
        L = x.shape[1]
        x = rearrange(x, 'b l n (h c) -> b l n h c', c=2)  # (b, l, n, h/2, 2)
        w = self.w[0 : L][None, :, None, :, :].to(device=x.device, dtype=x.dtype)  # (1, l, 1, h/2, 2)
        x = self.rotate(x, w)  # (b, l, n, h/2, 2)
        x = rearrange(x, 'b l n h c -> b l n (h c)')  # (b, l, n, h)
        
        return x

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        r"""Rotate x.

        x0 = cos(θp)·x0 - sin(θp)·x1
        x1 = sin(θp)·x0 + cos(θp)·x1

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h/2, 2)
            w: (1, l, 1, h/2, 2)

        Outputs:
            out: (b, l, n, h/2, 2)
        """

        out = torch.stack([
            w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
            w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0]
            ],
            dim=-1,
        )  # (b, l, n, h/2, 2)

        return out

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, l, n, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, l, n, h)
        """
        
        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(x, 'b l n (k h c) -> k b l n h c', k=K, c=2)  # (k, b, l, n, h/k/2, 2)
        out = torch.zeros_like(x, device=x.device)  # (k, b, l, n, h/k/2, 2)

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :].to(device=x.device, dtype=x.dtype)  # (1, l, 1, h/k/2, 2)
            out[i] = self.rotate(x[i], w)

        out = rearrange(out, 'k b l n h c -> b l n (k h c)')
        return out

class AbsolutePositionEmbedding(nn.Module):
    r"""Absolute Position Embedding.
    
    This module implements learnable absolute position embeddings that are added
    to the query and key projections in the attention mechanism. Unlike RoPE which
    applies rotations, absolute embeddings directly add positional information.
    
    Absolute position embeddings are simpler and can be more stable for certain tasks,
    especially when:
    - Sequence lengths are relatively fixed during training and inference
    - The model needs to learn specific positional patterns
    - Computational efficiency is critical (no rotation operations needed)
    
    Args:
        max_seq_len: Maximum sequence length to support
        dim: Embedding dimension (should match the model's hidden dimension)
        
    Shape:
        - Input: (batch_size, seq_len, num_heads, head_dim)
        - Output: (batch_size, seq_len, num_heads, head_dim)
    """
    
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.dim = dim
        
        # Learnable position embeddings
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        r"""Apply absolute position embeddings.
        
        Args:
            x: (b, l, n, h) - Input tensor
            
        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape
        
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds maximum supported length {self.max_seq_len}. "
                f"Please increase max_seq_len in the configuration."
            )
        
        # Get position embeddings for current sequence length
        pos_emb = self.pos_emb[:L]  # (l, d)
        
        # Reshape to match input dimensions: (l, d) -> (1, l, 1, d)
        pos_emb = pos_emb.view(1, L, 1, -1)
        
        # Add position embeddings to input
        out = x + pos_emb
        
        return out
    
    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply N-dimensional absolute position embeddings with sparse positions.
        
        This method supports multi-dimensional positional encoding (e.g., for images or videos)
        by indexing into the position embedding table using provided position indices.
        
        Args:
            x: (b, l, n, h) - Input tensor
            pos: (l, k) - Position indices for each dimension
            
        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape
        K = pos.shape[1]  # Number of dimensions
        
        # For N-D positions, we sum embeddings from each dimension
        # Each dimension gets H // K features
        dim_per_axis = self.dim // K
        
        pos_emb_sum = torch.zeros(L, self.dim, device=x.device, dtype=x.dtype)
        
        for i in range(K):
            p = pos[:, i]  # (l,)
            
            # Check bounds
            if torch.any(p >= self.max_seq_len):
                raise ValueError(
                    f"Position index {p.max().item()} exceeds maximum supported length {self.max_seq_len}"
                )
            
            # Get embeddings for this dimension
            start_idx = i * dim_per_axis
            end_idx = start_idx + dim_per_axis if i < K - 1 else self.dim
            pos_emb_sum[:, start_idx:end_idx] = self.pos_emb[p, start_idx:end_idx]
        
        # Reshape and add to input
        pos_emb_sum = pos_emb_sum.view(1, L, 1, -1)
        out = x + pos_emb_sum
        
        return out

class TransformerBlock(nn.Module):
    """Self-attention block with a configurable feed-forward network."""
    def __init__(
        self, 
        dim: int, 
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(
            dim, 
            num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )
        
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(dim, intermediate_dim=dim * 4)
        elif ffn_type == "gelu":
            self.ffn = GELUFeedForward(dim, intermediate_dim=dim * 4)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2) - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))

        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        
        input_dtype = x.dtype
        if input_dtype in {torch.float16, torch.bfloat16}:
            x = x.float()
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = (x * torch.rsqrt(norm_x + self.eps)).to(input_dtype) * self.scale
        return output


class SelfAttention(nn.Module):
    r"""Self-Attention module with configurable position embedding and attention backend.
    
    This module supports two types of position embeddings:
    
    1. RoPE (Rotary Position Embedding) - Default
       - Applies rotations to query and key vectors
       - Better extrapolation to longer sequences
       - No learnable parameters for position encoding
       - Recommended for variable-length sequences
       
    2. Absolute Position Embedding
       - Adds learnable position embeddings to query and key
       - Simpler and more stable for fixed-length sequences
       - Requires specifying max_seq_len
       - Can be more efficient computationally
    
    This module also supports two attention backends:
    
    1. torch - Uses PyTorch's scaled_dot_product_attention
       - Supports windowed attention via attention mask
       
    2. flash_attn - Uses flash_attn library for efficient attention
       - Supports windowed attention via window_size argument
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        position_embedding_type: Type of position embedding ("rope" or "absolute")
        max_seq_len: Maximum sequence length (required for absolute embeddings)
        attn_backend: Attention backend ("torch" or "flash_attn")
        window_size: Window size for local attention (None for global attention)
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        
        # Validate attention backend
        valid_backends = ["torch", "flash_attn"]
        if attn_backend not in valid_backends:
            raise ValueError(
                f"Invalid attn_backend: '{attn_backend}'. "
                f"Must be one of {valid_backends}"
            )
        
        # Check flash_attn availability
        if attn_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn backend requested but flash_attn is not installed. "
                "Please install it with: pip install flash-attn --no-build-isolation"
            )
        
        self.attn_backend = attn_backend
        
        # Validate position embedding type
        valid_types = ["rope", "absolute"]
        if position_embedding_type not in valid_types:
            raise ValueError(
                f"Invalid position_embedding_type: '{position_embedding_type}'. "
                f"Must be one of {valid_types}"
            )
        
        self.position_embedding_type = position_embedding_type
        
        # Initialize position embedding module if using absolute embeddings
        if position_embedding_type == "absolute":
            self.pos_embedding = AbsolutePositionEmbedding(
                max_seq_len=max_seq_len,
                dim=self.head_dim,
            )
        else:
            self.pos_embedding = None

        self.qkv_linear = MPSCompatibleLinear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = MPSCompatibleLinear(dim, dim)
    
    def _create_window_mask(self, seq_len: int, device: torch.device) -> Tensor:
        r"""Create attention mask for windowed attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            
        Returns:
            mask: (seq_len, seq_len) - Boolean mask where True indicates positions to attend to
        """
        if self.window_size is None:
            return None
        
        # Create a mask where each position can attend to positions within the window
        # Window is centered on each position
        positions = torch.arange(seq_len, device=device)
        
        # Calculate distance matrix
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Create mask: True for positions within window, False otherwise
        mask = distance <= self.window_size // 2
        
        return mask

    def forward(
        self,
        x: Tensor,
        rope: nn.Module | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention with configurable position embedding and backend.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d) - Input tensor
            rope: RoPE module - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            x: (b, l, d) - Output tensor
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply position embeddings based on type
        if self.position_embedding_type == "rope":
            # Use RoPE (Rotary Position Embedding)
            if rope is None:
                raise ValueError(
                    "RoPE module must be provided when position_embedding_type='rope'"
                )
            
            if pos is None:
                q = rope(q)  # (b, l, n, h)
                k = rope(k)  # (b, l, n, h)
            else:
                q = rope.apply_nd(q, pos)  # (b, l, n, h)
                k = rope.apply_nd(k, pos)  # (b, l, n, h)
                
        elif self.position_embedding_type == "absolute":
            # Use Absolute Position Embedding
            if pos is None:
                q = self.pos_embedding(q)  # (b, l, n, h)
                k = self.pos_embedding(k)  # (b, l, n, h)
            else:
                q = self.pos_embedding.apply_nd(q, pos)  # (b, l, n, h)
                k = self.pos_embedding.apply_nd(k, pos)  # (b, l, n, h)

        # Compute attention based on backend
        if self.attn_backend == "flash_attn":
            # Use flash_attn for efficient attention
            # flash_attn_func expects (b, l, n, h) layout and returns (b, l, n, h)
            if q.dtype not in (torch.float16, torch.bfloat16):
                if v.dtype in (torch.float16, torch.bfloat16):
                    flash_dtype = v.dtype
                elif q.is_cuda and torch.is_autocast_enabled("cuda"):
                    flash_dtype = torch.get_autocast_dtype("cuda")
                else:
                    raise RuntimeError(
                        "flash_attn backend requires fp16/bf16 q/k/v tensors. "
                        f"Got q={q.dtype}, k={k.dtype}, v={v.dtype}."
                    )
                q = q.to(flash_dtype)
                k = k.to(flash_dtype)
                v = v.to(flash_dtype)

            if self.window_size is not None:
                # flash_attn uses (window_left, window_right) for sliding window attention
                # We use symmetric window: (window_size // 2, window_size // 2)
                window_size_half = self.window_size // 2
                window_size_tuple = (window_size_half, window_size_half)
            else:
                window_size_tuple = (-1, -1)  # (-1, -1) means global attention
            
            x = flash_attn_func(
                q, k, v,
                window_size=window_size_tuple,
            )  # (b, l, n, h)
            
        else:
            # Use PyTorch's scaled_dot_product_attention
            # Create attention mask for windowed attention if needed
            attn_mask = None
            if self.window_size is not None:
                attn_mask = self._create_window_mask(x.shape[1], x.device)
                # scaled_dot_product_attention expects mask where True = masked out
                # So we need to invert our mask
                attn_mask = ~attn_mask
            
            x = F.scaled_dot_product_attention(
                query=rearrange(q, 'b l n h -> b n l h'), 
                key=rearrange(k, 'b l n h -> b n l h'), 
                value=rearrange(v, 'b l n h -> b n l h'), 
                attn_mask=attn_mask, 
                dropout_p=0.0
            )  # (b, n, l, h)
            x = rearrange(x, 'b n l h -> b l n h')

        x = rearrange(x, 'b l n h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        
        intermediate_dim = intermediate_dim * 2 // 3
        
        self.gate_proj = MPSCompatibleLinear(dim, intermediate_dim)
        self.up_proj = MPSCompatibleLinear(dim, intermediate_dim)
        self.down_proj = MPSCompatibleLinear(intermediate_dim, dim)
        self.act_fn = nn.SiLU()
        
    
    def forward(self, x: Tensor) -> Tensor:
        r"""Gated MLP.

        Args:
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GELUFeedForward(nn.Module):
    """GELU FFN with the same leading-order parameter count as SwiGLU."""

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = MPSCompatibleLinear(dim, intermediate_dim)
        self.act_fn = nn.GELU()
        self.down_proj = MPSCompatibleLinear(intermediate_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))

class RMSNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        r"""RMSNorm over channel dimension for 2D feature map.

        Args:
            x: (b, d, t, f)

        Outputs:
            output: (b, d, t, f)
        """

        norm_x = x.norm(2, dim=1, keepdim=True)
        rms = norm_x * (x.shape[1] ** -0.5)
        x = x / (rms + self.eps)
        return x * self.scale


class GEGLUFusion(nn.Module):
    """
    Robust GEGLU fusion module. 
    Concatenates features, projects to gate/value, and fuses.
    """

    def __init__(self, dim, dim_mult=2 ):
        super().__init__()
        self.act = nn.GELU()

        # 1. Normalization (Pre-norm) - separate norms for decoder and encoder
        self.norm_dec = RMSNorm2d(dim)
        self.norm_enc = RMSNorm2d(dim)

        # 2. Projection
        # We take 2 * dim inputs (enc + dec)
        # We project to 2 * (dim * dim_mult) because GEGLU requires splitting
        hidden_dim = int(dim * dim_mult)
        self.proj_in = nn.Conv2d(dim * 2, hidden_dim * 2, kernel_size=1)
        
        # 3. Output projection
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x_dec, x_enc):
        """
        Args:
            x_dec: Decoder features (B, C, H, W) -> Acts as the residual backbone
            x_enc: Encoder features (B, C, H', W')
        """
        # 1. Spatial Alignment (Crucial for U-Nets)
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode='bilinear', align_corners=False)

        # 2. Prepare inputs
        # We apply the residual connection to the Decoder stream, so we save x_dec
        residual = x_dec
        
        # Norm and Concat
        # (Using separate norms for decoder and encoder features)
        x_dec = self.norm_dec(x_dec)
        x_enc = self.norm_enc(x_enc)
        
        combined = torch.cat([x_dec, x_enc], dim=1)

        # 3. Project and Split (The "GEGLU" mechanism)
        # Project to [B, 2*hidden, H, W]
        gate_value = self.proj_in(combined) 
        
        # Split into two chunks along channel dim
        gate, value = gate_value.chunk(2, dim=1)

        # 4. Gating
        hidden = self.act(gate) * value

        # 5. Output Project + Residual
        return residual + self.proj_out(hidden)


class SiLUFusion(GEGLUFusion):
    def __init__(self, dim: int, dim_mult: float = 2) -> None:
        super().__init__(dim, dim_mult=dim_mult)
        self.act = nn.SiLU()


class AdditiveFusion(nn.Module):
    """Elementwise addition of decoder and skip features."""

    def __init__(self, dim: int, dim_mult: float = 2) -> None:
        super().__init__()
        del dim, dim_mult

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode="bilinear", align_corners=False)
        return x_dec + x_enc


class MultiplicativeFusion(nn.Module):
    """Elementwise multiplication of decoder and skip features."""

    def __init__(self, dim: int, dim_mult: float = 2) -> None:
        super().__init__()
        del dim, dim_mult

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode="bilinear", align_corners=False)
        return x_dec * x_enc


class NoSkipFusion(nn.Module):
    """Return decoder features without a high-resolution skip."""

    def __init__(self, dim: int, dim_mult: float = 2) -> None:
        super().__init__()
        del dim, dim_mult

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        del x_enc
        return x_dec


class LinearFusion(nn.Module):
    """Linearly project concatenated decoder and skip features."""

    def __init__(self, dim: int, dim_mult: float = 2) -> None:
        super().__init__()
        del dim_mult
        self.linear = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode="bilinear", align_corners=False)
        return self.linear(torch.cat([x_dec, x_enc], dim=1))

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class TimeMixBlock(nn.Module):
    def __init__(self, dim, dim_mult=4, ffn_type="swiglu"):
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == "swiglu":
            hidden_dim = int(dim * dim_mult * 2 // 3)
            projection_dim = hidden_dim * 2
            self.act = nn.SiLU()
        elif ffn_type == "gelu":
            hidden_dim = int(dim * dim_mult)
            projection_dim = hidden_dim
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = RMSNorm(dim)
        
        self.proj = MPSCompatibleLinear(dim, projection_dim)
        self.grn = GRN(hidden_dim)
        self.proj_out = MPSCompatibleLinear(hidden_dim, dim)

    def forward(self, x):
        h = self.pwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        h = self.proj(h)
        if self.ffn_type == "swiglu":
            gate, value = h.chunk(2, dim=-1)
            h = self.act(gate) * value
        else:
            h = self.act(h)
        h = self.grn(h)
        h = self.proj_out(h)
        return x + h


class BSRoformerBlock(nn.Module):
    """Band-Split RoFormer block with time and/or frequency attention."""

    def __init__(self, dim, n_heads, axis='tf', use_time_mix=False, ffn_type="swiglu"):
        super().__init__()
        assert axis in ['t', 'f', 'tf', 'ft'], "Axis must be one of 't', 'f', 'tf', or 'ft'."
        self.axis = axis
        self.use_time_mix = use_time_mix
        if 't' in axis:
            if use_time_mix:
                self.time_block = TimeMixBlock(dim, ffn_type=ffn_type)
            else:
                self.time_block = TransformerBlock(dim, n_heads, ffn_type=ffn_type)
        self.freq_block = TransformerBlock(dim, n_heads, ffn_type=ffn_type) if 'f' in axis else None

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        """
        Apply time and/or frequency attention.

        Args:
            x: Input tensor of shape (b, d, t, f).
            rope: Rotary position embedding.

        Returns:
            Output tensor of shape (b, d, t, f).
        """
        B = x.shape[0]

        if self.time_block is not None:
            # Time attention
            x = rearrange(x, 'b d t f -> (b f) t d')
            if self.use_time_mix:
                x = self.time_block(x)
            else:
                x = self.time_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b f) t d -> b d t f', b=B)

        if self.freq_block is not None:
            # Frequency attention
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = self.freq_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x

class STFTLearnable(nn.Module):
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        n_fractions=1, 
        dft_init=True,
        learnable=True
    ):
        r"""
        f: freq_bins
        r: n_fractions
        n: n_samples
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        N = n_fft
        R = n_fractions

        if dft_init:    
            k = torch.arange(0, N * R)  # (f*r,)
            n = torch.arange(0, N)  # (n,)
            kn = torch.outer(k, n)  # (f, n)
            w = torch.exp(-1.j * 2 * math.pi / (N * R) * kn) / math.sqrt(N * R)  # (f*r, n)
        else:
            w = torch.empty(N * R, N)
            torch.nn.init.xavier_uniform_(w)  # (f*r, n)

        if learnable:
            self.w = nn.Parameter(w)  # (f*r, n)
        else:
            self.register_buffer("w", w)  # (f*r, n)

    def analysis(self, x: Tensor) -> Tensor:
        r"""STFT.

        b: batch_size
        l: audio_samples
        t: n_frames
        n: frame_samples
        f: freq_bins

        Args:
            x: (b, l), complex

        Returns:
            out: (b, t, f), complex
        """
        N = self.n_fft
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, l)
        x = x.unfold(dimension=-1, size=N, step=self.hop_length).contiguous()  # (b, t, n)
        x *= self.window  # (b, t, n)
        x = x.to(self.w.dtype)  # (b, t, n)
        out = x @ self.w.T  # (b, t, f)
        return out

    def synthesis(self, x: Tensor, length: int) -> Tensor:
        r"""Inverse STFT.

        b: batch_size
        t: n_frames
        f: freq_bins
        n: frame_samples
        l: audio_samples

        Args:
            x: (b, t, f), complex

        Returns:
            x: (b, l), complex
        """
        x = x @ self.w.conj()  # (b, t, n)

        # Overlap add
        out = overlap_add(x=x, hop_length=self.hop_length, window=self.window)  # (b, l)
        out = out[..., self.n_fft // 2 :]  # (b, l)
        
        if length is not None:
            out = out[..., 0 : length]  # (b, l)

        return out


DEFAULT_GLOBAL_BS_FREQS_PER_BAND = (
    *((2,) * 24),
    *((4,) * 12),
    *((12,) * 8),
    *((24,) * 8),
    *((48,) * 8),
    128,
    129,
)


def _ragged_index_map(dim_inputs: tuple[int, ...]) -> Tensor:
    max_dim = max(dim_inputs)
    total_dim = sum(dim_inputs)
    indexes = torch.full((len(dim_inputs), max_dim), total_dim, dtype=torch.long)
    offset = 0
    for band_index, dim_input in enumerate(dim_inputs):
        indexes[band_index, :dim_input] = torch.arange(offset, offset + dim_input)
        offset += dim_input
    return indexes


class BandRMSNorm(nn.Module):
    """Apply independent ZFTurbo-style RMSNorms to all padded bands at once."""

    def __init__(self, dim_inputs: tuple[int, ...], eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(sum(dim_inputs)))
        self.register_buffer("parameter_indexes", _ragged_index_map(dim_inputs), persistent=False)
        self.register_buffer(
            "valid_mask",
            torch.arange(max(dim_inputs))[None, :] < torch.tensor(dim_inputs)[:, None],
            persistent=False,
        )
        self.register_buffer("counts", torch.tensor(dim_inputs).float(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        mask = self.valid_mask.to(dtype=x.dtype)
        norm = (x.square() * mask).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        scale = self.counts.to(dtype=x.dtype)[None, None, :, None].sqrt() / norm
        padded_weight = torch.cat([self.weight, self.weight.new_zeros(1)])[self.parameter_indexes]
        return x * scale * padded_weight * mask


class RaggedBandLinear(nn.Module):
    """Project ragged band inputs with one gathered weight tensor and one einsum."""

    def __init__(self, dim_inputs: tuple[int, ...], out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(sum(dim_inputs), out_features))
        self.bias = nn.Parameter(torch.empty(len(dim_inputs), out_features))
        self.register_buffer("parameter_indexes", _ragged_index_map(dim_inputs), persistent=False)

        offset = 0
        for band_index, dim_input in enumerate(dim_inputs):
            bound = 1 / math.sqrt(dim_input)
            nn.init.uniform_(self.weight.data[offset:offset + dim_input], -bound, bound)
            nn.init.uniform_(self.bias.data[band_index], -bound, bound)
            offset += dim_input

    def forward(self, x: Tensor) -> Tensor:
        zero_row = self.weight.new_zeros(1, self.out_features)
        weight = torch.cat([self.weight, zero_row])[self.parameter_indexes]
        return torch.einsum("btki,kio->btko", x, weight) + self.bias


class BandSplit(nn.Module):
    """One-shot equivalent of ZFTurbo's per-band RMSNorm and Linear modules."""

    def __init__(self, dim_inputs: tuple[int, ...], dim: int) -> None:
        super().__init__()
        self.norm = BandRMSNorm(dim_inputs)
        self.proj = RaggedBandLinear(dim_inputs, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


class RaggedBandMaskEstimator(nn.Module):
    """Decode one complex mask value per ragged band occurrence with one einsum."""

    def __init__(self, dim_inputs: tuple[int, ...], dim: int) -> None:
        super().__init__()
        total_dim = sum(dim_inputs)
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(total_dim, 2, dim))
        self.bias = nn.Parameter(torch.empty(total_dim, 2))
        self.register_buffer("parameter_indexes", _ragged_index_map(dim_inputs), persistent=False)
        bound = 1 / math.sqrt(dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        zero_weight = self.weight.new_zeros(1, 2, self.dim)
        zero_bias = self.bias.new_zeros(1, 2)
        weight = torch.cat([self.weight, zero_weight])[self.parameter_indexes]
        bias = torch.cat([self.bias, zero_bias])[self.parameter_indexes]
        mask = torch.einsum("btkd,kird->btkir", x, weight) + bias
        value, gate = mask.unbind(dim=-1)
        return value * gate.sigmoid()


class GlobalSTFTFrontend(nn.Module):
    """ZFTurbo-style global BS or overlapping binary-mel STFT frontend."""

    def __init__(
        self,
        audio_channels: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        feature_dim: int,
        projection: str,
        mel_bands: int = 60,
        freqs_per_bands: tuple[int, ...] | list[int] | None = None,
        normalized: bool = False,
        zero_dc: bool = True,
    ) -> None:
        super().__init__()
        if projection not in {"bs", "mel"}:
            raise ValueError(f"Unsupported global STFT projection: {projection}")
        if win_length > n_fft:
            raise ValueError(f"global_stft_win_length={win_length} exceeds n_fft={n_fft}")

        self.audio_channels = int(audio_channels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.feature_dim = int(feature_dim)
        self.projection = projection
        self.normalized = bool(normalized)
        self.zero_dc = bool(zero_dc)
        self.freq_bins = self.n_fft // 2 + 1

        band_freq_indexes = self._build_band_freq_indexes(
            sample_rate=sample_rate,
            mel_bands=int(mel_bands),
            freqs_per_bands=freqs_per_bands,
        )
        self.n_bands = len(band_freq_indexes)
        dim_inputs = tuple(len(indexes) * self.audio_channels * 2 for indexes in band_freq_indexes)
        self.dim_inputs = dim_inputs
        self.max_dim_input = max(dim_inputs)

        raw_dim = self.freq_bins * self.audio_channels * 2
        input_indexes = torch.full((self.n_bands, self.max_dim_input), raw_dim, dtype=torch.long)
        occurrences_per_raw = torch.zeros(raw_dim)
        for band_index, frequency_indexes in enumerate(band_freq_indexes):
            raw_indexes = []
            for frequency_index in frequency_indexes:
                for channel_index in range(self.audio_channels):
                    for complex_index in range(2):
                        raw_index = ((frequency_index * self.audio_channels + channel_index) * 2 + complex_index)
                        raw_indexes.append(raw_index)
                        occurrences_per_raw[raw_index] += 1
            input_indexes[band_index, :len(raw_indexes)] = torch.tensor(raw_indexes)

        if torch.any(occurrences_per_raw == 0):
            raise ValueError("Global STFT bands do not cover every complex frequency input")
        self.register_buffer("input_indexes", input_indexes, persistent=False)
        self.register_buffer("occurrences_per_raw", occurrences_per_raw, persistent=False)
        self.register_buffer("window", torch.hann_window(self.win_length))
        self.band_split = BandSplit(dim_inputs, self.feature_dim)
        self.mask_estimator = RaggedBandMaskEstimator(dim_inputs, self.feature_dim)

    def _build_band_freq_indexes(
        self,
        sample_rate: int,
        mel_bands: int,
        freqs_per_bands: tuple[int, ...] | list[int] | None,
    ) -> list[list[int]]:
        if self.projection == "bs":
            widths = tuple(int(value) for value in (freqs_per_bands or DEFAULT_GLOBAL_BS_FREQS_PER_BAND))
            if sum(widths) != self.freq_bins:
                raise ValueError(
                    f"global_freqs_per_bands sum to {sum(widths)}, expected {self.freq_bins} for n_fft={self.n_fft}"
                )
            indexes = []
            start = 0
            for width in widths:
                indexes.append(list(range(start, start + width)))
                start += width
            return indexes

        mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=self.n_fft,
            n_mels=mel_bands,
            norm=None,
        )
        mel_filter[0, 0] = 1
        mel_filter[-1, -1] = 1
        supports = torch.from_numpy(mel_filter) > 0
        if not supports.any(dim=0).all():
            raise ValueError("Mel bands do not cover every STFT frequency bin")
        if not supports.any(dim=1).all():
            raise ValueError("At least one mel band contains no STFT frequency bins")
        return [torch.nonzero(row, as_tuple=True)[0].tolist() for row in supports]

    def analysis(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, channels = audio.shape[:2]
        spec = torch.stft(
            rearrange(audio, "b c l -> (b c) l"),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=audio.device, dtype=audio.dtype),
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        spec = rearrange(spec, "(b c) f t -> b c t f", b=batch_size, c=channels)
        raw = rearrange(torch.view_as_real(spec), "b c t f ri -> b t (f c ri)")
        raw = torch.cat([raw, raw.new_zeros(*raw.shape[:-1], 1)], dim=-1)
        bands = raw[..., self.input_indexes]
        features = self.band_split(bands)
        return spec, rearrange(features, "b t k d -> b d t k")

    def aggregate_masks(self, band_masks: Tensor) -> Tensor:
        batch_size, time_steps = band_masks.shape[:2]
        raw_dim = self.freq_bins * self.audio_channels * 2
        scatter_indexes = self.input_indexes.flatten().view(1, 1, -1).expand(batch_size, time_steps, -1)
        masks = band_masks.flatten(2)
        summed = masks.new_zeros(batch_size, time_steps, raw_dim + 1).scatter_add_(
            dim=-1,
            index=scatter_indexes,
            src=masks,
        )
        averaged = summed[..., :raw_dim] / self.occurrences_per_raw.to(dtype=summed.dtype)
        averaged = rearrange(
            averaged,
            "b t (f c ri) -> b c t f ri",
            f=self.freq_bins,
            c=self.audio_channels,
            ri=2,
        )
        return torch.view_as_complex(averaged.contiguous().float())

    def apply_mask_and_synthesize(self, spec: Tensor, mask: Tensor, length: int) -> Tensor:
        latent = rearrange(mask, "b d t k -> b t k d")
        band_masks = self.mask_estimator(latent)
        complex_mask = self.aggregate_masks(band_masks)
        estimate = spec * complex_mask
        estimate = rearrange(estimate, "b c t f -> (b c) f t")
        if self.zero_dc:
            estimate = estimate.index_fill(1, torch.tensor(0, device=estimate.device), 0)
        audio = torch.istft(
            estimate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=estimate.device, dtype=estimate.real.dtype),
            normalized=self.normalized,
            onesided=True,
            length=length,
        )
        return rearrange(audio, "(b c) l -> b c l", b=spec.shape[0], c=self.audio_channels)

class SBRoformer(nn.Module):
    """RoFormer separator with ablatable frontends, analysis, fusion, and FFNs."""

    def __init__(
        self,
        audio_channels: int = 2,
        num_stems: int = 1,
        sample_rate: int = 48000,
        n_bands: int | None = None,
        subband_n_bands: int | None = None,
        dim_sp: int = 96,
        dim: int = 384,
        dim_head: int = 32,
        patch_size: list[int] | tuple[int, int] = (4, 4),
        n_layers: int = 12,
        n_pre_layers: int = 1,
        n_post_layers: int = 1,
        rope_len: int = 8192,
        hop_length: int = 4,
        frontend_type: str = "waveform_subband",
        analysis_type: str = "cnn",
        mask_mode: str | None = None,
        cnn_mask_mode: str | None = None,
        stft_mask_mode: str | None = None,
        n_fft: int = 32,
        stft_hop_length: int = 8,
        stft_backend: str | None = None,
        learnable_stft: bool | None = None,
        stft_n_fractions: int = 1,
        stft_dft_init: bool = True,
        stft_learnable: bool = True,
        direct_frame_size: int | None = None,
        analysis_centered: bool = False,
        bank_type: str = "mel_linear_triangle",
        subband_factor: int | None = None,
        max_bandwidth: int = 390,
        max_half_bandwidth: int | None = None,
        chunk_size: int = 16,
        bandpass_filter_len: int = 48000,
        upsample_filter_len: int = 12000,
        debug_print_shapes: bool = False,
        fusion_type: str | None = None,
        fusion_dim_mult: float = 2,
        use_fusion: bool | None = None,
        ffn_type: str = "swiglu",
        global_stft_n_fft: int = 2048,
        global_stft_hop_length: int = 480,
        global_stft_win_length: int | None = None,
        global_stft_normalized: bool = False,
        global_zero_dc: bool = True,
        global_band_dim: int = 128,
        global_mel_bands: int = 60,
        global_freqs_per_bands: list[int] | tuple[int, ...] | None = None,
        attn_backend: str | None = None,
        use_flash_attention: bool | None = None,
        gradient_checkpointing: bool = False,
        dsp_backend: str = "dsp",
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_n_fft: int = 2048,
        multi_stft_window_fn=torch.hann_window,
        **kwargs,
    ) -> None:
        super().__init__()

        # Kept only for config compatibility with 56b YAMLs.
        kwargs.pop("band_dim", None)
        del kwargs

        self.audio_channels = audio_channels
        self.num_stems = int(num_stems)
        if self.num_stems < 1:
            raise ValueError(f"num_stems must be positive, got {self.num_stems}")
        self.patch_size = tuple(int(v) for v in patch_size)
        if len(self.patch_size) != 2 or any(value < 1 for value in self.patch_size):
            raise ValueError(f"patch_size must contain two positive integers, got {self.patch_size}")
        self.subband_n_bands = int(subband_n_bands or n_bands or 118)
        self.hop_length = int(hop_length)
        self.frontend_type = self._normalize_frontend_type(frontend_type)
        self.dsp_backend = str(dsp_backend).lower()
        if self.dsp_backend not in {"dsp", "dsp2", "dsp3"}:
            raise ValueError(f"Unsupported dsp_backend: {dsp_backend}")
        if self.dsp_backend in {"dsp2", "dsp3"} and self.frontend_type != "waveform_subband":
            raise ValueError(f"dsp_backend='{self.dsp_backend}' requires frontend_type='waveform_subband'")
        self.analysis_type = self._normalize_analysis_type(analysis_type)
        self.mask_mode = self._resolve_mask_mode(mask_mode, cnn_mask_mode, stft_mask_mode)
        self.cnn_mask_mode = self.mask_mode
        self.stft_mask_mode = self.mask_mode
        self.n_fft = int(n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.stft_backend = self._resolve_stft_backend(stft_backend, learnable_stft)
        self.stft_n_fractions = int(stft_n_fractions)
        if self.stft_n_fractions < 1:
            raise ValueError(f"stft_n_fractions must be >= 1, got {self.stft_n_fractions}")
        self.stft_bins = self.n_fft * self.stft_n_fractions if self.stft_backend == "learnable" else self.n_fft
        self.bank_type = bank_type
        self.sb_factor = int(subband_factor or max(sample_rate // 800, 1))
        self._last_cnn_input_length: int | None = None
        self._last_stft_input_length: int | None = None
        self._last_direct_input_length: int | None = None
        self.direct_frame_size = int(direct_frame_size or self.stft_hop_length)
        if self.direct_frame_size < 1:
            raise ValueError(f"direct_frame_size must be positive, got {self.direct_frame_size}")
        self.analysis_centered = bool(analysis_centered)
        self.debug_print_shapes = debug_print_shapes
        self.fusion_type = self._resolve_fusion_type(fusion_type, use_fusion)
        self.use_fusion = self.fusion_type not in {"none", "disabled"}
        self.ffn_type = self._normalize_ffn_type(ffn_type)
        self.attn_backend = self._resolve_attn_backend(attn_backend, use_flash_attention)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        banks = self._build_banks(
            sample_rate,
            self.subband_n_bands,
            int(max_half_bandwidth or max_bandwidth),
        )
        self.model_n_bands = len(banks)
        if self.frontend_type == "waveform_subband":
            filter_class = SubbandFilter
            if self.dsp_backend == "dsp2":
                from .frontend.subband.dsp2 import SubbandFilter as FixedPointSubbandFilter

                filter_class = FixedPointSubbandFilter
            elif self.dsp_backend == "dsp3":
                from .frontend.subband.dsp3 import SubbandFilter as FusedSubbandFilter

                filter_class = FusedSubbandFilter
            self.sb_filter = filter_class(
                sample_rate,
                banks,
                self.sb_factor,
                chunk_size=chunk_size,
                bandpass_filter_len=bandpass_filter_len,
                upsample_filter_len=upsample_filter_len,
            )
            if self.dsp_backend == "dsp":
                self.sb_filter = InferenceSubbandFilter(self.sb_filter, level=3)
            elif self.dsp_backend == "dsp2":
                from .frontend.subband.dsp2.inference_optimized import InferenceSubbandFilter as FixedPointInferenceSubbandFilter

                self.sb_filter = FixedPointInferenceSubbandFilter(self.sb_filter)
            else:
                from .frontend.subband.dsp3.inference_optimized import InferenceSubbandFilter as FusedInferenceSubbandFilter

                self.sb_filter = FusedInferenceSubbandFilter(self.sb_filter)
        else:
            self.sb_filter = None

        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)
        if self.frontend_type != "waveform_subband":
            self.feature_dim = int(global_band_dim)
        elif self.analysis_type in {"cnn", "direct"}:
            self.feature_dim = dim_sp
        else:
            self.feature_dim = audio_channels * self.stft_bins * 2
        if self.feature_dim % dim_head != 0:
            raise ValueError(f"feature_dim={self.feature_dim} must be divisible by dim_head={dim_head}")

        if self.frontend_type == "waveform_subband":
            self.global_frontend = None
        else:
            projection = "bs" if self.frontend_type == "global_stft_bs" else "mel"
            self.global_frontend = GlobalSTFTFrontend(
                audio_channels=audio_channels,
                sample_rate=sample_rate,
                n_fft=int(global_stft_n_fft),
                hop_length=int(global_stft_hop_length),
                win_length=int(global_stft_win_length or global_stft_n_fft),
                feature_dim=self.feature_dim,
                projection=projection,
                mel_bands=int(global_mel_bands),
                freqs_per_bands=global_freqs_per_bands,
                normalized=bool(global_stft_normalized),
                zero_dc=bool(global_zero_dc),
            )
            self.model_n_bands = self.global_frontend.n_bands

        if self.analysis_type == "stft" and self.stft_backend == "learnable":
            self.stft_transform = STFTLearnable(
                n_fft=self.n_fft,
                hop_length=self.stft_hop_length,
                n_fractions=self.stft_n_fractions,
                dft_init=stft_dft_init,
                learnable=stft_learnable,
            )
        else:
            self.stft_transform = None

        if self.frontend_type == "waveform_subband" and self.analysis_type == "cnn":
            self.cnn_encoder = nn.Conv2d(
                audio_channels * 2,
                dim_sp,
                kernel_size=(self.hop_length, 1),
                stride=(self.hop_length, 1),
                bias=False,
            )
            self.cnn_decoder = nn.ConvTranspose2d(
                dim_sp,
                audio_channels * 2,
                kernel_size=(self.hop_length, 1),
                stride=(self.hop_length, 1),
                bias=False,
            )
        else:
            self.cnn_encoder = None
            self.cnn_decoder = None

        self.down = nn.Conv2d(self.feature_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.up = nn.ConvTranspose2d(dim, self.feature_dim, kernel_size=self.patch_size, stride=self.patch_size)
        fusion_modules = {
            "geglu": lambda: GEGLUFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "silu": lambda: SiLUFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "add": lambda: AdditiveFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "multiply": lambda: MultiplicativeFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "none": lambda: NoSkipFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "linear": lambda: LinearFusion(self.feature_dim, dim_mult=fusion_dim_mult),
            "disabled": nn.Identity,
        }
        self.fusion = fusion_modules[self.fusion_type]()

        self.pre_blocks = nn.ModuleList(
            [
                self._make_block(
                    self.feature_dim,
                    self.feature_dim // dim_head,
                    axis="tf",
                    use_time_mix=True,
                )
                for _ in range(n_pre_layers)
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                self._make_block(
                    self.feature_dim,
                    self.feature_dim // dim_head,
                    axis="tf",
                    use_time_mix=True,
                )
                for _ in range(n_post_layers)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                self._make_block(dim, dim // dim_head, axis="tf")
                for _ in range(n_layers)
            ]
        )
        self.stem_mask_heads = nn.ModuleList(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
            for _ in range(self.num_stems)
        ) if self.num_stems > 1 else None

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = multi_stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )

    def _resolve_attn_backend(
        self,
        attn_backend: str | None,
        use_flash_attention: bool | None,
    ) -> str:
        if attn_backend is None:
            backend = "flash_attn" if use_flash_attention else "torch"
        else:
            aliases = {
                "torch": "torch",
                "sdpa": "torch",
                "pytorch": "torch",
                "flash": "flash_attn",
                "flash_attn": "flash_attn",
                "flash_attention": "flash_attn",
            }
            try:
                backend = aliases[attn_backend.lower()]
            except KeyError as exc:
                raise ValueError(f"Unsupported attn_backend: {attn_backend}") from exc

            if use_flash_attention is not None:
                requested = "flash_attn" if use_flash_attention else "torch"
                if backend != requested:
                    raise ValueError(
                        f"attn_backend={backend!r} conflicts with "
                        f"use_flash_attention={use_flash_attention!r}"
                    )

        if backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn backend requested but flash_attn is not installed. "
                "Please install it with: pip install flash-attn --no-build-isolation"
            )
        return backend

    def _make_block(self, *args, **kwargs) -> BSRoformerBlock:
        kwargs.setdefault("ffn_type", self.ffn_type)
        block = BSRoformerBlock(*args, **kwargs)
        for module in block.modules():
            if isinstance(module, SelfAttention):
                module.attn_backend = self.attn_backend
        return block

    def _run_block(self, block: BSRoformerBlock, x: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(lambda value: block(value, rope=self.rope), x, use_reentrant=False)
        return block(x, rope=self.rope)

    def _normalize_frontend_type(self, frontend_type: str) -> str:
        aliases = {
            "waveform": "waveform_subband",
            "waveform_subband": "waveform_subband",
            "subband": "waveform_subband",
            "global_stft_bs": "global_stft_bs",
            "stft_bs": "global_stft_bs",
            "global_stft_mel": "global_stft_mel",
            "stft_mel": "global_stft_mel",
        }
        try:
            return aliases[frontend_type.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported frontend_type: {frontend_type}") from exc

    def _normalize_analysis_type(self, analysis_type: str) -> str:
        analysis_type = analysis_type.lower()
        if analysis_type not in {"cnn", "stft", "direct"}:
            raise ValueError(f"Unsupported analysis_type: {analysis_type}")
        return analysis_type

    def _resolve_fusion_type(self, fusion_type: str | None, use_fusion: bool | None) -> str:
        if fusion_type is None:
            return "geglu" if use_fusion is not False else "disabled"
        fusion_type = fusion_type.lower()
        if fusion_type not in {"geglu", "silu", "add", "multiply", "none", "linear"}:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        if use_fusion is not None and (fusion_type != "none") != bool(use_fusion):
            raise ValueError(f"fusion_type={fusion_type!r} conflicts with use_fusion={use_fusion!r}")
        return fusion_type

    def _normalize_ffn_type(self, ffn_type: str) -> str:
        ffn_type = ffn_type.lower()
        if ffn_type not in {"swiglu", "gelu"}:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
        return ffn_type

    def _resolve_stft_backend(self, stft_backend: str | None, learnable_stft: bool | None) -> str:
        if stft_backend is None:
            return "learnable" if learnable_stft else "torch"

        backend = stft_backend.lower()
        aliases = {
            "torch": "torch",
            "pytorch": "torch",
            "fixed": "torch",
            "learnable": "learnable",
            "learned": "learnable",
            "stft_learnable": "learnable",
        }
        try:
            backend = aliases[backend]
        except KeyError as exc:
            raise ValueError(f"Unsupported stft_backend: {stft_backend}") from exc

        if learnable_stft is not None:
            requested = "learnable" if learnable_stft else "torch"
            if backend != requested:
                raise ValueError(
                    f"stft_backend={backend!r} conflicts with learnable_stft={learnable_stft!r}"
                )
        return backend

    def _resolve_mask_mode(
        self,
        mask_mode: str | None,
        cnn_mask_mode: str | None,
        stft_mask_mode: str | None,
    ) -> str:
        legacy_mask_mode = cnn_mask_mode if self.analysis_type == "cnn" else stft_mask_mode
        if mask_mode is not None and legacy_mask_mode is not None:
            mask_mode = self._normalize_mask_mode(mask_mode)
            legacy_mask_mode = self._normalize_mask_mode(legacy_mask_mode)
            if mask_mode != legacy_mask_mode:
                raise ValueError(
                    f"mask_mode={mask_mode!r} conflicts with legacy "
                    f"{self.analysis_type}_mask_mode={legacy_mask_mode!r}"
                )
            return mask_mode
        if mask_mode is not None:
            return self._normalize_mask_mode(mask_mode)
        if legacy_mask_mode is not None:
            return self._normalize_mask_mode(legacy_mask_mode)
        return "features"

    def _normalize_mask_mode(self, mask_mode: str) -> str:
        aliases = {
            "feature": "features",
            "features": "features",
            "transform": "features",
            "transform_feature": "features",
            "transform_features": "features",
            "cnn": "features",
            "cnn_feature": "features",
            "cnn_features": "features",
            "stft": "features",
            "stft_feature": "features",
            "stft_features": "features",
            "subband": "subband",
            "subbands": "subband",
            "subband_analysis": "subband",
        }
        try:
            return aliases[mask_mode.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}") from exc

    def _build_banks(
        self,
        sample_rate: int,
        n_bands: int,
        max_half_bandwidth: int,
    ) -> list[tuple[float, float]]:
        if self.bank_type in {"mel_linear_triangle", "mel_triangle", "115d"}:
            return mel_linear_banks_triangle(sample_rate, n_bands, max_half_bandwidth)
        raise ValueError(f"Unsupported bank_type: {self.bank_type}")

    def predict_mask(self, features: Tensor) -> Tensor:
        frames_num = features.shape[2]
        bands_num = features.shape[3]

        x = self.pad_tensor(features)
        if self.debug_print_shapes:
            print(f"down_before: {tuple(x.shape)}")
        x = features = x

        for block in self.pre_blocks:
            x = self._run_block(block, x)
        h = x
        x = self.down(x)
        if self.debug_print_shapes:
            print(f"down_after: {tuple(x.shape)}")


        for block in self.blocks:
            x = self._run_block(block, x)

        x = self.up(x)
        if self.fusion_type != "disabled":
            x = self.fusion(x, h)
        for block in self.post_blocks:
            x = self._run_block(block, x)

        if self.debug_print_shapes:
            print(f"mask_before_crop: {tuple(x.shape)}")
        if self.stem_mask_heads is None:
            mask = x[:, :, :frames_num, :bands_num]
        else:
            mask = torch.stack(
                [head(x)[:, :, :frames_num, :bands_num] for head in self.stem_mask_heads],
                dim=1,
            )
        return mask

    def forward(
        self,
        audio: Tensor,
        target: Tensor | None = None,
        active_stem_ids: list[int] | None = None,
        return_loss_breakdown: bool = False,
    ):
        device = audio.device

        recon_audio = self.separate(audio)

        if target is None:
            return recon_audio

        if recon_audio.ndim == 3:
            recon_audio = rearrange(recon_audio, "b s t -> b 1 s t")

        if active_stem_ids is None:
            stem_ids = list(range(self.num_stems))
        else:
            stem_ids = active_stem_ids
            recon_audio = recon_audio[:, stem_ids]

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        if target.ndim == 3:
            target = rearrange(target, "b s t -> b 1 s t")

        target = target[..., :recon_audio.shape[-1]]

        target_sel = target[:, stem_ids]
        loss = F.l1_loss(recon_audio, target_sel)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, "b n s t -> (b n s) t"), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target_sel, "b n s t -> (b n s) t"), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)

    def separate(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        global_spec = None
        subbands = None
        if self.frontend_type == "waveform_subband":
            if self.sb_filter is None:
                raise RuntimeError("Waveform subband frontend is not initialized")
            audio = self.pad_audio(audio, self.sb_factor)
            subbands = self.sb_filter.analysis(audio)
            features = self.analysis(subbands)
        else:
            if self.global_frontend is None:
                raise RuntimeError("Global STFT frontend is not initialized")
            global_spec, features = self.global_frontend.analysis(audio)
        mask = self.predict_mask(features)
        if self.frontend_type != "waveform_subband":
            if self.global_frontend is None or global_spec is None:
                raise RuntimeError("Global STFT analysis must run before synthesis")
            if self.num_stems > 1:
                batch_size = global_spec.shape[0]
                flat_spec = rearrange(
                    global_spec[:, None].expand(-1, self.num_stems, -1, -1, -1),
                    "b n c t f -> (b n) c t f",
                )
                flat_mask = rearrange(mask, "b n d t k -> (b n) d t k")
                estimates = self.global_frontend.apply_mask_and_synthesize(flat_spec, flat_mask, audio_length)
                return rearrange(estimates, "(b n) c l -> b n c l", b=batch_size, n=self.num_stems)
            return self.global_frontend.apply_mask_and_synthesize(global_spec, mask, audio_length)

        if self.sb_filter is None or subbands is None:
            raise RuntimeError("Waveform subband analysis must run before synthesis")
        if self.num_stems > 1:
            batch_size = subbands.shape[0]
            flat_subbands = rearrange(
                subbands[:, None].expand(-1, self.num_stems, -1, -1, -1),
                "b n c k l -> (b n) c k l",
            )
            flat_features = rearrange(
                features[:, None].expand(-1, self.num_stems, -1, -1, -1),
                "b n d t k -> (b n) d t k",
            )
            flat_mask = rearrange(mask, "b n d t k -> (b n) d t k")
            estimates = self.apply_mask_and_synthesize(flat_subbands, flat_features, flat_mask)
            estimates = self.sb_filter.synthesis(estimates)
            estimates = self.trim_or_pad_output(estimates, audio_length)
            return rearrange(estimates, "(b n) c l -> b n c l", b=batch_size, n=self.num_stems)
        x = self.apply_mask_and_synthesize(subbands, features, mask)
        out = self.sb_filter.synthesis(x)
        return self.trim_or_pad_output(out, audio_length)

    def analysis(self, x: Tensor) -> Tensor:
        if self.analysis_type == "cnn":
            return self.cnn_analysis(x)
        if self.analysis_type == "stft":
            return self.stft_analysis(x)
        if self.analysis_type == "direct":
            return self.direct_analysis(x)
        raise RuntimeError(f"Unhandled analysis_type: {self.analysis_type}")

    def apply_mask_and_synthesize(self, subbands: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        if self.mask_mode == "features":
            if self.analysis_type == "stft":
                complex_sp = self.stft_channels_to_complex(features)
                complex_mask = self.stft_channels_to_complex(mask)
                return self.istft(complex_sp * complex_mask)
            if self.analysis_type == "direct":
                complex_sp = self.direct_channels_to_complex(features)
                complex_mask = self.direct_channels_to_complex(mask)
                return complex_sp * complex_mask
            return self.cnn_synthesis(features * mask)

        if self.mask_mode == "subband":
            if self.analysis_type == "stft":
                complex_mask = self.stft_channels_to_complex(mask)
                return subbands * self.istft(complex_mask)
            if self.analysis_type == "direct":
                return subbands * self.direct_channels_to_complex(mask)
            return subbands * self.cnn_synthesis(mask)

        raise RuntimeError(f"Unhandled mask_mode: {self.mask_mode}")

    def stft_analysis(self, x: Tensor) -> Tensor:
        self._last_stft_input_length = x.shape[-1]
        complex_sp = self.stft(x)
        return self.complex_stft_to_channels(complex_sp)

    def complex_stft_to_channels(self, x: Tensor) -> Tensor:
        return rearrange(torch.view_as_real(x), "b c k t f ri -> b (c f ri) t k")

    def stft_channels_to_complex(self, x: Tensor) -> Tensor:
        x = rearrange(
            x,
            "b (c f ri) t k -> b c k t f ri",
            c=self.audio_channels,
            f=self.stft_bins,
            ri=2,
        )
        return torch.view_as_complex(x.contiguous().float())

    def direct_analysis(self, x: Tensor) -> Tensor:
        self._last_direct_input_length = x.shape[-1]
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        if self.analysis_centered:
            center_pad = self.direct_frame_size // 2
            x = F.pad(x, pad=(center_pad, center_pad, 0, 0), mode="reflect")
        pad_length = -x.shape[-1] % self.direct_frame_size
        x = F.pad(x, pad=(0, pad_length))
        x = rearrange(
            torch.view_as_real(x.contiguous()),
            "b c k (t h) ri -> b (c h ri) t k",
            h=self.direct_frame_size,
        )
        if x.shape[1] > self.feature_dim:
            raise ValueError(
                f"Direct subband samples need {x.shape[1]} channels, but feature_dim={self.feature_dim}"
            )
        return F.pad(x, pad=(0, 0, 0, 0, 0, self.feature_dim - x.shape[1]))

    def direct_channels_to_complex(self, x: Tensor) -> Tensor:
        direct_channels = self.audio_channels * self.direct_frame_size * 2
        x = rearrange(
            x[:, :direct_channels],
            "b (c h ri) t k -> b c k (t h) ri",
            c=self.audio_channels,
            h=self.direct_frame_size,
            ri=2,
        )
        x = torch.view_as_complex(x.contiguous().float())
        if self._last_direct_input_length is None:
            raise RuntimeError("direct_analysis() must be called before direct synthesis")
        start = self.direct_frame_size // 2 if self.analysis_centered else 0
        return x[..., start:start + self._last_direct_input_length]

    def stft(self, x: Tensor) -> Tensor:
        B, C = x.shape[0:2]
        x = rearrange(x, "b c k l -> (b c k) l")
        if self.stft_backend == "learnable":
            if self.stft_transform is None:
                raise RuntimeError("Learnable STFT backend was requested but stft_transform is not initialized")
            x = self.stft_transform.analysis(x)
            return rearrange(x, "(b c k) t f -> b c k t f", b=B, c=C)

        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.stft_hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        return rearrange(x, "(b c k) f t -> b c k t f", b=B, c=C)

    def istft(self, x: Tensor) -> Tensor:
        B, C = x.shape[0:2]
        if self.stft_backend == "learnable":
            if self.stft_transform is None:
                raise RuntimeError("Learnable STFT backend was requested but stft_transform is not initialized")
            x = rearrange(x, "b c k t f -> (b c k) t f")
            x = self.stft_transform.synthesis(x, self._last_stft_input_length)
            return rearrange(x, "(b c k) l -> b c k l", b=B, c=C)

        x = rearrange(x, "b c k t f -> (b c k) f t")
        x = torch.istft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.stft_hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
            length=self._last_stft_input_length,
        )
        return rearrange(x, "(b c k) l -> b c k l", b=B, c=C)

    def cnn_analysis(self, x: Tensor) -> Tensor:
        if self.cnn_encoder is None:
            raise RuntimeError("cnn_analysis() is only available when analysis_type='cnn'")
        self._last_cnn_input_length = x.shape[-1]
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = torch.view_as_real(x.contiguous())
        x = rearrange(x, "b c f t ri -> b (c ri) t f")
        if self.analysis_centered:
            center_pad = self.cnn_encoder.kernel_size[0] // 2
            x = F.pad(x, pad=(0, 0, center_pad, center_pad), mode="reflect")
        x = self._pad_for_cnn_analysis(x)
        if self.debug_print_shapes:
            print(f"cnn_analysis_before: {tuple(x.shape)}")
        x = self.cnn_encoder(x)
        if self.debug_print_shapes:
            print(f"cnn_analysis_after: {tuple(x.shape)}")
        return x

    def cnn_synthesis(self, x: Tensor) -> Tensor:
        if self.cnn_decoder is None:
            raise RuntimeError("cnn_synthesis() is only available when analysis_type='cnn'")
        target_length = self._last_cnn_input_length
        if target_length is None:
            raise RuntimeError("cnn_analysis() must be called before cnn_synthesis() so synthesis length is known")

        if self.debug_print_shapes:
            print(f"cnn_synthesis_before: {tuple(x.shape)}")
        x = self.cnn_decoder(x)
        start = self.cnn_decoder.kernel_size[0] // 2 if self.analysis_centered else 0
        x = x[:, :, start:start + target_length, :]
        if x.shape[2] < target_length:
            x = F.pad(x, pad=(0, 0, 0, target_length - x.shape[2]))
        if self.debug_print_shapes:
            print(f"cnn_synthesis_after: {tuple(x.shape)}")
        x = rearrange(
            x,
            "b (c ri) t f -> b c f t ri",
            c=self.audio_channels,
            ri=2,
        )
        return torch.view_as_complex(x.contiguous().float())

    def _pad_for_cnn_analysis(self, x: Tensor) -> Tensor:
        if self.cnn_encoder is None:
            raise RuntimeError("_pad_for_cnn_analysis() is only available when analysis_type='cnn'")
        length = x.shape[-2]
        kernel_size = self.cnn_encoder.kernel_size[0]
        stride = self.cnn_encoder.stride[0]
        if length <= kernel_size:
            target_length = kernel_size
        else:
            frames = (length - kernel_size + stride - 1) // stride + 1
            target_length = (frames - 1) * stride + kernel_size
        pad_right = target_length - length
        if pad_right > 0:
            x = F.pad(x, pad=(0, 0, 0, pad_right))
        return x

    def pad_tensor(self, x: Tensor) -> Tensor:
        pad_t = -x.shape[2] % self.patch_size[0]
        pad_f = -x.shape[3] % self.patch_size[1]
        return F.pad(x, pad=(0, pad_f, 0, pad_t))

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def trim_or_pad_output(self, out: Tensor, audio_length: int) -> Tensor:
        out = out[..., :audio_length]
        if out.shape[-1] < audio_length:
            out = F.pad(out, pad=(0, audio_length - out.shape[-1]))
        return out
