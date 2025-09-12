from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import math

from typing import Optional, Tuple


def window_partition_nchw(x, window_size):
    """
    Partition NCHW tensor into windows.
    Args:
      x: (B, C, H, W)
      window_size: (wh, ww)
    Returns:
      windows: (num_windows_total, wh*ww, C) i.e. (B * n_h, n_w, wh * ww, C) later the num_window_total will be treaded as
      normal batches and the attension will work on each window independently, and then recombined via reverse operation.
      (Hp, Wp, pad_h, pad_w, n_h, n_w)
    """
    B, C, H, W = x.shape
    wh, ww = window_size
    # Calculate padding size for height and width to be divisible by window size
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        # F.pad expects (left, right, top, bottom) for the last dimension,
        # then the second to last, and so on.
        # So for (B, C, H, W), padding is (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_c_front, pad_c_back, pad_b_front, pad_b_back)
        # We only need to pad W and H, so it's (0, pad_w, 0, pad_h) applied to the last two spatial dimensions (W and H)
        x = F.pad(x, (0, pad_w, 0, pad_h))
    # Calculate padded height and width
    Hp, Wp = H + pad_h, W + pad_w
    # Calculate number of windows along height and width
    n_h = Hp // wh
    n_w = Wp // ww
    # reshape to (B, C, n_h, wh, n_w, ww) - arrange windows within the image
    x = x.view(B, C, n_h, wh, n_w, ww)
    # permute to (B, n_h, n_w, wh, ww, C) - move channels to the end
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    # reshape to (num_windows_total, wh*ww, C) - flatten windows and combine batch and window dimensions
    windows = x.view(-1, wh * ww, C)  # (B*n_h*n_w, wh*ww, C)
    # Return windows and information needed for reverse operation
    return windows, (Hp, Wp, pad_h, pad_w, n_h, n_w)

def window_reverse_nchw(windows, window_size, Hp, Wp, pad_h, pad_w, n_h, n_w, B):
    """
    Reverse windows back to NCHW tensor.
    Args:
      windows: (B*n_h*n_w, wh*ww, C) - flattened windows
      window_size: (wh, ww) - height and width of each window
      Hp, Wp: padded H and W - height and width after padding
      pad_h, pad_w: paddings - amount of padding applied to height and width
      n_h, n_w: #windows per dim - number of windows along height and width
      B: original batch size
    Returns:
      x: (B, C, H, W) restored (unpadded) tensor
    """
    wh, ww = window_size
    C = windows.shape[-1] # Get the number of channels from the windows tensor
    # Reshape the windows back to their spatial arrangement within each image
    x = windows.view(B, n_h, n_w, wh, ww, C)
    # Permute the dimensions back to the original NCHW format (B, C, H, W)
    # This reverses the permutation done in window_partition_nchw
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, n_h, wh, n_w, ww)
    # Reshape to combine the window dimensions with the number of windows dimensions
    x = x.view(B, C, Hp, Wp) # (B, C, Hp, Wp) - padded tensor

    # Remove padding if it was applied
    if pad_h > 0:
        h = Hp - pad_h
    else:
        h = Hp
    if pad_w > 0:
        w = Wp - pad_w
    else:
        w = Wp
    # Slice the tensor to remove the padding and get the original height and width
    x = x[:, :, :h, :w].contiguous()
    return x

class PatchEmbedSVTR(nn.Module):
    """Overlapping Patch Embedding (NCHW) similar to SVTR paper."""
    def __init__(self, img_size=(64, 256), in_chans=3, embed_dim=64):
        super().__init__()
        H, W = img_size
        self.img_size = img_size
        # two conv layers with stride 2 each -> downsample by (4,4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        # store patches resolution
        self.patches_resolution = (H // 4, W // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input size {(H,W)} != expected {self.img_size}"
        x = self.conv1(x)
        x = self.conv2(x)
        # now x: (B, embed_dim, H/4, W/4)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, h*w, D)
        x = self.norm(x)
        return x  # (B, N, D)

class MLP(nn.Module):
    def __init__(self, in_dim, ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(in_dim * ratio)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)

class GlobalMixing(nn.Module):
    """Global mixing (Transformer-style) block."""
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) # Layer normalization before attention
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop) # Multi-head self-attention
        self.norm2 = nn.LayerNorm(dim) # Layer normalization after attention and before MLP
        self.mlp = MLP(dim, ratio=mlp_ratio, drop=drop) # MLP block (Feed-forward network)

    def forward(self, x):
        # x: (B, N, D) where B is batch size, N is sequence length, D is dimension
        res = x # Residual connection
        x = self.norm1(x) # Apply LayerNorm Improved Training Stability
        # Apply Multi-head self-attention. Query, Key, and Value are all the same (self-attention).
        attn_out, _ = self.attn(x, x, x)
        x = res + attn_out # Add attention output to the residual connection

        res = x # Second residual connection
        x = self.norm2(x) # Apply LayerNorm
        x = res + self.mlp(x) # Add MLP output to the residual connection (including dropout within MLP)
        return x # Output tensor with the same shape as input (B, N, D)

class NOLMWA(nn.Module):
    """
    Non-overlapping local window attention:
      - partition to windows
      - apply MultiheadAttention on each window independently by batching them
      - reverse
    Inputs to forward(): x is (B, N, D) with h,w passed separately.
    """
    def __init__(self, dim, num_heads, mlp_ratio=2.0, window_size=(7,11), drop=0.0):
        super().__init__()
        self.dim = dim # Input dimension
        self.num_heads = num_heads # Number of attention heads
        self.wh, self.ww = window_size # Window height and width
        self.norm1 = nn.LayerNorm(dim) # Layer normalization before attention
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop) # Multi-head self-attention
        self.norm2 = nn.LayerNorm(dim) # Layer normalization after attention and before MLP
        self.mlp = MLP(dim, ratio=mlp_ratio, drop=drop) # MLP block (Feed-forward network)

    def forward(self, x, h, w, shift=(0,0)):
        # x: (B, N, D) where B is batch size, N is sequence length (h*w), D is dimension
        B, N, D = x.shape
        assert N == h * w, "N must equal h*w" # Assert that sequence length matches spatial dimensions

        res = x # Residual connection

        # Pre-normalization
        x = self.norm1(x)

        # to NCHW for windowing - reshape to (B, C, h, w)
        x2 = x.transpose(1, 2).reshape(B, D, h, w)

        # Partition into windows - windows shape (B*n_h*n_w, wh*ww, C)
        windows, (Hp, Wp, pad_h, pad_w, n_h, n_w) = window_partition_nchw(x2, (self.wh, self.ww))

        # Apply Multi-head self-attention on windows
        # windows shape is (num_windows_total, wh*ww, C) but C==D
        attn_out, _ = self.attn(windows, windows, windows)

        # Reverse window partitioning - reconstruct to (B, C, h, w)
        x2 = window_reverse_nchw(attn_out, (self.wh, self.ww), Hp, Wp, pad_h, pad_w, n_h, n_w, B)

        # Reshape back to (B, N, D)
        x = x2.view(B, D, h * w).transpose(1, 2).contiguous()

        # Add residual connection after attention
        x = res + x

        # Second residual connection
        res = x

        # Pre-normalization before MLP
        x = self.norm2(x)

        # Apply MLP and add residual connection
        x = res + self.mlp(x)

        return x # Output tensor with the same shape as input (B, N, D)

class SWLMWA(nn.Module):
    """
    Local windowed attention with optional cyclic shift and per-window attention mask.
    Implements manual multi-head attention so we can apply a (num_windows, ws, ws) mask.
    """

    def __init__(self, dim, num_heads, mlp_ratio=2.0, window_size=(7,11), drop=0.0, shift_size=(0,0)):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.wh, self.ww = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.shift_size = shift_size if shift_size is not None else (0,0)

        # Layer norm before attention and QKV projection
        self.norm1 = nn.LayerNorm(dim)
        # QKV projection: linearly transform input to Query, Key, and Value
        # Output size is 3 * dim for Q, K, and V combined
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # Output projection after attention
        self.proj = nn.Linear(dim, dim, bias=True)

        # Layer norm before MLP
        self.norm2 = nn.LayerNorm(dim)
        # MLP (Feed-forward network) - Reusing the MLP class defined earlier
        self.mlp = MLP(dim, ratio=mlp_ratio, drop=drop)

        # The attention mask will be generated in the forward pass based on input dimensions.
        # We can register an empty buffer or None here, or simply generate it directly in forward.
        # We will generate it directly in forward.


    def forward(self, x, h, w):
        """
        x: (B, N, D) with N == h*w
        h,w: spatial dims (tokens grid) before windowing
        """
        B, N, D = x.shape
        assert N == h * w, f"N({N}) must equal h({h})*w({w})"
        device = x.device
        res = x # Residual connection

        # pre-norm
        x = self.norm1(x)

        # to NCHW for windowing
        # Reshape to (B, C, H, W) where C=D, H=h, W=w
        x2 = x.transpose(1, 2).reshape(B, D, h, w)  # (B, D, H, W)

        # apply cyclic shift if requested (negative roll to shift content BEFORE partitioning,
        # which matches Swin's order)
        if self.shift_size[0] != 0 or self.shift_size[1] != 0:
            shift_h, shift_w = self.shift_size
            # Roll the tensor along height and width dimensions
            x2 = torch.roll(x2, shifts=(-shift_h, -shift_w), dims=(2, 3))

        # partition into windows
        # Output windows shape: (B * n_h * n_w, ws, C) where ws = wh*ww
        windows, (Hp, Wp, pad_h, pad_w, n_h, n_w) = window_partition_nchw(x2, (self.wh, self.ww))
        # windows: (num_windows_total, ws, C) where num_windows_total = B * n_h * n_w

        num_windows_total = windows.shape[0] # Total number of windows across the batch
        ws = windows.shape[1]  # seq length inside a window
        num_windows_per_image = n_h * n_w # Number of windows per image

        # Generate attention mask dynamically based on current input dimensions
        # Create a mask grid that labels each (Hp x Wp) token with its window index
        img_mask = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.int32)
        cnt = 0
        for i in range(0, Hp, self.wh):
            for j in range(0, Wp, self.ww):
                img_mask[:, :, i:i + self.wh, j:j + self.ww] = cnt
                cnt += 1

        # Partition that mask the same way we partition x2 (but on a single image)
        # mask_windows shape: (1 * n_h * n_w, ws, 1)
        mask_windows, _ = window_partition_nchw(img_mask, (self.wh, self.ww))

        # Reshape mask_windows to (num_windows_per_image, ws)
        # Remove the singleton channel dimension from mask_windows
        mask_windows = mask_windows.squeeze(-1) # Shape becomes (num_windows_per_image, ws)


        # Create per-window boolean mask where True indicates "forbid attention"
        # (n_h * n_w, ws, ws) of booleans
        attn_mask = (mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2))

        # Repeat the mask for the batch dimension
        # attn_mask shape: (n_h * n_w, ws, ws) -> repeat B times -> (B, n_h * n_w, ws, ws)
        repeated_attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1)
        # Reshape to match the batched windows dimension: (B * n_h * n_w, ws, ws)
        repeated_attn_mask = repeated_attn_mask.view(num_windows_total, ws, ws)


        # Manual multi-head attention over windows, with mask applied per-window
        # windows: (num_windows_total, ws, C)
        qkv = self.qkv(windows)  # (num_windows_total, ws, 3*D)
        # Split QKV into Query, Key, Value along the last dimension
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for heads: (num_windows_total, num_heads, ws, head_dim)
        def reshape_to_heads(t):
            # Reshape to (num_windows_total, ws, num_heads, head_dim)
            t = t.view(num_windows_total, ws, self.num_heads, self.head_dim)
            # Permute to (num_windows_total, num_heads, ws, head_dim) for batch matrix multiplication
            return t.permute(0, 2, 1, 3).contiguous()

        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        # scaled dot-product: (num_windows_total, num_heads, ws, ws)
        # Calculate attention scores: Q @ K_transpose / sqrt(head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # apply mask: set to -inf where forbidden
        # Use a large negative number for masked positions so that softmax makes them effectively zero
        inf_mask = torch.tensor(-1e9, device=device, dtype=attn.dtype)
        attn = attn.masked_fill(repeated_attn_mask.unsqueeze(1), inf_mask)

        # softmax & dropout (dropout omitted here but can be added)
        attn = torch.softmax(attn, dim=-1)
        # Optionally add dropout after softmax: attn = F.dropout(attn, p=drop, training=self.training)

        # attention output (num_windows_total, num_heads, ws, head_dim)
        # Compute the output by multiplying attention probabilities with Value tensor
        out = torch.matmul(attn, v)

        # merge heads -> (num_windows_total, ws, D)
        # Permute back to (num_windows_total, ws, num_heads, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        # Reshape to (num_windows_total, ws, D) by combining num_heads and head_dim
        out = out.view(num_windows_total, ws, D)
        # Apply output projection
        out = self.proj(out)

        # reverse windows -> reconstruct (B, C, Hp, Wp)
        # Use window_reverse_nchw to bring the windows back to the padded image shape
        x2 = window_reverse_nchw(out, (self.wh, self.ww), Hp, Wp, pad_h, pad_w, n_h, n_w, B)

        # reverse cyclic shift (if applied) by positive shift
        if self.shift_size[0] != 0 or self.shift_size[1] != 0:
            shift_h, shift_w = self.shift_size
            # Roll back the tensor to reverse the cyclic shift
            x2 = torch.roll(x2, shifts=(shift_h, shift_w), dims=(2, 3))

        # back to (B, N, D)
        # Reshape from (B, D, H, W) to (B, N, D) where N=H*W
        x = x2.view(B, D, h * w).transpose(1, 2).contiguous()

        # residual + MLP
        x = res + x # Add residual connection after attention
        res2 = x # Second residual connection before MLP
        x = self.norm2(x) # Pre-normalization before MLP
        x = res2 + self.mlp(x) # Apply MLP and add residual connection

        return x # Output tensor with the same shape as input (B, N, D)

class OffsetPredictor(nn.Module):
    """
    Predict offsets for each query token.

    Produces offsets shaped (B, N, H, P, 2) in pixel units (dx, dy).
    The output of the linear layer is squashed with tanh and multiplied by offset_scale.
    """
    def __init__(self, dim: int, num_heads: int, n_points: int, offset_scale: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_points = n_points
        self.offset_scale = float(offset_scale)
        self.linear = nn.Linear(dim, num_heads * n_points * 2, bias=True)

        # initialize offset predictor to small values so offsets start near zero
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        out = self.linear(x)  # (B, N, H * P * 2)
        out = out.view(B, N, self.num_heads, self.n_points, 2)  # (B, N, H, P, 2)
        out = out.tanh() * self.offset_scale  # scaled pixel offsets (dx, dy)
        return out  # (B, N, H, P, 2)

def make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Construct base pixel coordinates for each spatial location.
    Returned shape: (N, 2) with coordinate order (x, y).
    N == h * w
    """
    # coords_y: [0, 1, ..., h-1], coords_x: [0, 1, ..., w-1]
    coords_y = torch.arange(h, device=device, dtype=dtype)
    coords_x = torch.arange(w, device=device, dtype=dtype)
    # meshgrid with indexing='ij' gives grid_y shape (h, w) and grid_x shape (h, w)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
    # stack as (x, y) per pixel and flatten to (N, 2)
    base_xy = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)  # (N, 2), columns: (x, y)
    return base_xy

class FeatureMapProducer(nn.Module):
    """
    Produce key and value feature maps from flattened tokens (B, N, D).
    Produces k_map and v_map each shaped (B, D, h, w).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.kv_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, N, D) where N == h * w
        B, N, D = tokens.shape
        feat = tokens.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)
        kv = self.kv_conv(feat)  # (B, 2*D, h, w)
        k_map, v_map = kv.chunk(2, dim=1)  # each (B, D, h, w)
        return k_map, v_map
    
class FeatureSampler:
    """
    Utility functions (not a nn.Module) to sample features at fractional positions using F.grid_sample.

    - Sample per-head feature maps: we reshape feature maps into (B*H, C_head, h, w)
    - Build grid and call F.grid_sample with align_corners=True to match normalization convention used.
    """
    @staticmethod
    def sample_maps_at_points(
        feature_map: torch.Tensor,
        sample_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample feature_map at normalized points.

        feature_map: (B * H, C_head, h, w)
        sample_norm: (B * H, N * P, 2) in normalized coords [-1, 1], ordering (x, y) per point.
        Returns:
            sampled: (B, H, N, P, C_head)
        """
        B_H, C_head, h, w = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype

        # grid requires shape (N_batch, H_out, W_out, 2). We'll set H_out = N*P, W_out = 1
        grid = sample_norm.view(B_H, -1, 1, 2)  # (B*H, N*P, 1, 2)

        # grid_sample -> output (B*H, C_head, N*P, 1)
        sampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # reshape to (B, H, C_head, N, P, 1) -> drop last dim -> (B, H, C_head, N, P)
        # we know B_H = B * H, so recover B and H later
        # first reshape to (B, H, C_head, N, P)
        # But we need N and P; they can be derived from sample_norm original leading dim: sample_norm.view(B_H, N*P, 2)
        # So compute N*P:
        _, NP, _ = sample_norm.shape  # NP = N * P
        # We'll infer N by dividing with known P (passed via context in caller) — but to keep method generic,
        # the caller should have arranged sample_norm with the correct ordering and will later reshape.
        # Here we produce (B_H, C_head, NP, 1) -> reshape to (B, H, C_head, N, P) by caller's knowledge.
        sampled = sampled.squeeze(-1)  # (B*H, C_head, NP)

        return sampled  # (B*H, C_head, N*P)
    
class DeformableAttention(nn.Module):
    """
    Readable, modular implementation of deformable attention.

    Key features:
    - Per-query, per-head learnable offsets (n_points per query per head)
    - Bilinear sampling of K and V feature maps using F.grid_sample
    - Dot-product attention between query and sampled keys (softmax over P)
    - Output projected back to embedding dimension

    Parameters:
        dim: input & output embedding dimension
        num_heads: number of attention heads
        n_points: sampling points per query per head
        offset_scale: scale multiplier for tanh-squashed offsets (in pixels)
    """
    def __init__(self, dim: int, num_heads: int, n_points: int = 9, offset_scale: float = 4.0, debug: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.n_points = n_points
        self.head_dim = dim // num_heads
        self.offset_scale = float(offset_scale)
        self.debug = debug

        # modules
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.offset_predictor = OffsetPredictor(dim, num_heads, n_points, offset_scale=offset_scale)
        self.feature_producer = FeatureMapProducer(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B, H, N, head_dim)
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, head_dim) -> (B, N, D)
        B, H, N, hd = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * hd)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: (B, N, D), where N == h * w
        returns: (B, N, D)
        """
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype
        assert N == h * w, "N must equal h*w"

        if self.debug:
            print(f"[DeformableAttention] input: {x.shape}, h={h}, w={w}")

        # 1) Query projection -> (B, H, N, head_dim)
        q = self.to_q(x)
        q = self._split_heads(q)  # (B, H, N, hd)

        # 2) Offsets -> (B, N, H, P, 2) in pixel units (dx, dy)
        offsets = self.offset_predictor(x)  # (B, N, H, P, 2)

        # 3) Feature maps (k_map, v_map) each (B, D, h, w)
        k_map, v_map = self.feature_producer(x, h, w)  # (B, D, h, w)
        if self.debug:
            print(f"[DeformableAttention] k_map shape: {k_map.shape}, v_map shape: {v_map.shape}")

        # 4) Split K and V per head: -> (B*H, head_dim, h, w)
        B, D, _, _ = k_map.shape
        k_map_heads = k_map.view(B, self.num_heads, self.head_dim, h, w).reshape(B * self.num_heads, self.head_dim, h, w)
        v_map_heads = v_map.view(B, self.num_heads, self.head_dim, h, w).reshape(B * self.num_heads, self.head_dim, h, w)

        # 5) Base grid (pixel coords) and absolute sample positions
        base_xy = make_base_grid(h, w, device=device, dtype=dtype)  # (N, 2) with (x, y)
        base_xy = base_xy.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # offsets currently (B, N, H, P, 2) in (dx, dy) ordering; grid_sample needs (x, y)
        # swap to (dx, dy) -> (x,y) means ordering (dx, dy) corresponds to adding to base (x,y)
        # we have offsets as (dx, dy) ordering already if OffsetPredictor produced (dx,dy)
        sample_xy = base_xy.unsqueeze(2).unsqueeze(3) + offsets  # (B, N, H, P, 2)

        # normalize sample coordinates to [-1, 1] for grid_sample (x normalized by w-1, y by h-1)
        norm = torch.tensor([(w - 1), (h - 1)], device=device, dtype=dtype).view(1, 1, 1, 1, 2)
        sample_norm = (sample_xy / norm) * 2.0 - 1.0  # (B, N, H, P, 2)

        # 6) Rearrange sample_norm for grid_sample:
        # desired ordering for sampling is (B, H, N*P, 2) -> then view to (B*H, N*P, 2)
        sample_norm = sample_norm.permute(0, 2, 1, 3, 4).contiguous()  # (B, H, N, P, 2)
        B_H = B * self.num_heads
        sample_norm = sample_norm.view(B_H, N * self.n_points, 2)  # (B*H, N*P, 2)

        if self.debug:
            print(f"[DeformableAttention] sample_norm shape (for grid_sample): {sample_norm.shape}")

        # 7) Use grid_sample to fetch features at sampled points (bilinear interpolation)
        # grid_sample expects (N_batch, C, h, w) and grid shaped (N_batch, H_out, W_out, 2),
        # and returns (N_batch, C, H_out, W_out).
        # We'll set H_out = N * P, W_out = 1
        grid_for_gs = sample_norm.view(B_H, N * self.n_points, 1, 2)  # (B*H, N*P, 1, 2)
        sampled_k = F.grid_sample(k_map_heads, grid_for_gs, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B*H, hd, N*P, 1)
        sampled_v = F.grid_sample(v_map_heads, grid_for_gs, mode='bilinear', padding_mode='zeros', align_corners=True)

        # remove last dim and reshape to (B, H, hd, N, P)
        sampled_k = sampled_k.squeeze(-1).view(B, self.num_heads, self.head_dim, N, self.n_points)
        sampled_v = sampled_v.squeeze(-1).view(B, self.num_heads, self.head_dim, N, self.n_points)

        # permute to (B, H, N, P, hd)
        sampled_k = sampled_k.permute(0, 1, 3, 4, 2).contiguous()
        sampled_v = sampled_v.permute(0, 1, 3, 4, 2).contiguous()

        if self.debug:
            print(f"[DeformableAttention] sampled_k shape: {sampled_k.shape}, sampled_v shape: {sampled_v.shape}")

        # 8) Compute attention logits and weights
        # q: (B, H, N, hd) ; sampled_k: (B, H, N, P, hd)
        # compute dot product along hd -> (B, H, N, P)
        # use einsum for clarity
        attn_logits = torch.einsum('bhnd,bhnpd->bhnp', q, sampled_k) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, N, P)

        # 9) Weighted sum of sampled_v -> (B, H, N, hd)
        out_heads = torch.einsum('bhnp,bhnpd->bhnd', attn_weights, sampled_v)  # (B, H, N, hd)

        # 10) Merge heads and project
        out = self._merge_heads(out_heads)  # (B, N, D)
        out = self.out_proj(out)  # (B, N, D)

        if self.debug:
            print(f"[DeformableAttention] out shape: {out.shape}")

        return out
    
class DLMWA(nn.Module):
    """
    Local mixing block wrapping DeformableAttention with LayerNorm and MLP (residuals).
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, n_points: int = 9, offset_scale: float = 4.0, debug: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DeformableAttention(dim, num_heads, n_points=n_points, offset_scale=offset_scale, debug=debug)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio=mlp_ratio)
        self.debug = debug

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.debug:
            print(f"[DLMWA] input: {x.shape}")
        res = x
        x = self.norm1(x)
        x = self.attn(x, h, w)
        x = res + x
        res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + x
        if self.debug:
            print(f"[DLMWA] output: {x.shape}")
        return x

class LocalMixingConv(nn.Module):
    """
    A fast local mixing implemented by depthwise convolution followed by pointwise projection.
    Works as a local mixing alternative to window attention (useful for speed).
    """
    def __init__(self, dim, kernel_size=(7,11), mlp_ratio=2.0, drop=0.0):
        super().__init__()
        kh, kw = kernel_size
        self.norm1 = nn.LayerNorm(dim) # Layer normalization before depthwise convolution
        # Depthwise convolution: applies a separate convolution to each input channel
        self.dw = nn.Conv2d(dim, dim, kernel_size=(kh, kw), padding=(kh//2, kw//2), groups=dim)
        # Pointwise convolution: 1x1 convolution to mix information across channels
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(dim) # Layer normalization before MLP
        self.mlp = MLP(dim, ratio=mlp_ratio, drop=drop) # MLP block

    def forward(self, x, h, w):
        # x: (B, N, D) where B is batch size, N is sequence length (h*w), D is dimension
        B, N, D = x.shape
        assert N == h * w, "N must equal h*w" # Assert sequence length matches spatial dimensions
        res = x # Residual connection

        # Pre-normalization
        x = self.norm1(x)

        # Reshape to NCHW for convolution (B, D, h, w)
        x2 = x.transpose(1, 2).reshape(B, D, h, w)

        # Apply depthwise and pointwise convolutions
        x2 = self.dw(x2)
        x2 = self.pw(x2)

        # Reshape back to (B, N, D)
        x = x2.view(B, D, h * w).transpose(1, 2).contiguous()

        # Add residual connection after convolution
        x = res + x

        # Second residual connection
        res = x

        # Pre-normalization before MLP
        x = self.norm2(x)

        # Apply MLP and add residual connection
        x = res + self.mlp(x)

        return x # Output tensor with the same shape as input (B, N, D)

class Merging(nn.Module):
    """
    Merge reduces height by 2 (stride (2,1)) similar to SVTR merge layer.
    Input x: (B, N, D) with h,w provided.
    Returns x_new (B, N_new, D_out), and new (h, w).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(2,1), padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, h, w):
        B, N, D = x.shape
        x2 = x.transpose(1, 2).reshape(B, D, h, w)
        x2 = self.conv(x2)  # (B, out_dim, h//2, w)
        _, _, hp, wp = x2.shape
        x = x2.flatten(2).transpose(1, 2).contiguous()  # (B, hp*wp, out_dim)
        x = self.norm(x)
        return x, hp, wp
    
class Combining(nn.Module):
    """
    Combine across height: collapse height via mean and project channels to out_dim.
    """
    def __init__(self, in_dim, out_dim, drop=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        # x: (B, N, D) where N = h*w
        B, N, D = x.shape
        x2 = x.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)
        # collapse height by mean -> (B, D, 1, w)
        x2 = x2.mean(dim=2, keepdim=True)  # average over height
        x2 = x2.squeeze(2)  # (B, D, w)
        x2 = x2.transpose(1, 2).contiguous()  # (B, w, D)
        x2 = self.fc(x2)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        return x2  # (B, w, out_dim)

class SVTR(nn.Module):
    """
    SVTR-like model architecture (PyTorch implementation).

    Args:
        img_size (tuple): Input image size (height, width). Defaults to (64, 256).
        in_chans (int): Number of input image channels. Defaults to 3.
        vocab_size (int): Size of the output vocabulary (number of characters). Defaults to 100.
        embed_dims (tuple): Embedding dimensions for each stage. Defaults to (64, 128, 256).
        d3 (int): Output dimension of the combining layer before the head. Defaults to 192.
        heads (tuple): Number of attention heads for each stage. Defaults to (2, 4, 8).
        mlp_ratio (float): Ratio to determine hidden dimension in MLP. Defaults to 2.0.
        window_sizes (list(tuple)): Window sizes for window-based local attention. length must match the number of 'L' blocks in the pattern.
                             Also used as kernel size for LocalMixingConv.
        num_blocks (tuple): Number of blocks in each stage. Defaults to (3, 6, 3).
        pattern (list, optional): List of 'L' (local) or 'G' (global) specifying block types.
                                  If None, a default pattern is generated.
        local_type (list): List of local mixing types for each local block ('non_overlapping', 'swin', 'deformable', 'conv').
                           Length must match the number of 'L' blocks in the pattern.
        drop (float): Dropout rate. Defaults to 0.0.
        n_points (int): Number of sampling points for Deformable Attention. Defaults to 9.
        offset_scale (float): Scaling factor for Deformable Attention offsets. Defaults to 4.0.
    """
    def __init__(self,
                 img_size=(64, 256),
                 in_chans=3,
                 vocab_size=100,
                 embed_dims=(64, 128, 256),
                 d3=192,
                 heads=(2, 4, 8),
                 mlp_ratio=2.0,
                 window_sizes=[(7, 11)] * 12 + [(3,3)] * 6, # Default window sizes if not provided
                 num_blocks=(3, 6, 3),
                 pattern=None,
                 local_type=None,
                 drop=0.0,
                 n_points=9,
                 offset_scale=4.0):
        super().__init__()

        # pattern length must cover sum(num_blocks). L = local, G = global
        total_blocks = sum(num_blocks)
        assert pattern is not None and len(pattern) == total_blocks, f"Pattern must be a list of length {total_blocks} specifying 'L' or 'G' for each block."

        # Ensure local_type is a list and matches the number of local blocks in the pattern
        assert isinstance(local_type, list), "local_type must be a list specifying block types."
        num_local_blocks = pattern.count('L')
        assert len(local_type) >= num_local_blocks, f"Length of local_type list ({len(local_type)}) must match the number of local blocks in pattern ({num_local_blocks})."
        self.local_type_list = local_type

        # Ensure window_sizes list length matches the number of local blocks
        assert isinstance(window_sizes, list), "window_sizes must be a list of tuples."
        assert len(window_sizes) >= num_local_blocks, f"Length of window_sizes list ({len(window_sizes)}) must match the number of local blocks in pattern ({num_local_blocks})."
        self.window_sizes_list = window_sizes


        self.patch_embed = PatchEmbedSVTR(img_size, in_chans, embed_dim=embed_dims[0])
        self.patches_resolution = self.patch_embed.patches_resolution
        dims = list(embed_dims) # Convert to list
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dims[0]))  # learnable positional embedding
        self.n_points = n_points # Store for Deformable Attention
        self.offset_scale = offset_scale # Store for Deformable Attention
        self.drop_rate = drop # Store dropout rate

        # Build the model stages based on the pattern and local_type
        cur_pattern_idx = 0
        cur_local_type_idx = 0
        cur_window_size_idx = 0

        # Stage 1
        self.stage1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            tp = pattern[cur_pattern_idx]
            if tp == 'L':
                current_local_type = self.local_type_list[cur_local_type_idx]
                current_window_size = self.window_sizes_list[cur_window_size_idx]
                if current_local_type == 'non_overlapping':
                    blk = NOLMWA(dims[0], heads[0], mlp_ratio, current_window_size, self.drop_rate)
                elif current_local_type == 'swin':
                    # Alternate shift for SWIN
                    do_shift = (i % 2 == 1)
                    shift_size = (current_window_size[0] // 2, current_window_size[1] // 2) if do_shift else (0, 0)
                    blk = SWLMWA(dims[0], heads[0], mlp_ratio, current_window_size, self.drop_rate, shift_size)
                elif current_local_type == 'deformable':
                    blk = DLMWA(dims[0], heads[0], mlp_ratio, self.n_points, self.offset_scale, debug=False) # debug=False by default
                elif current_local_type == 'conv':
                    blk = LocalMixingConv(dims[0], kernel_size=current_window_size, mlp_ratio=mlp_ratio, drop=self.drop_rate)
                else:
                    raise ValueError(f"Unknown local_type '{current_local_type}' for block {cur_pattern_idx}")
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else: # Global mixing
                blk = GlobalMixing(dims[0], heads[0], mlp_ratio, self.drop_rate)
            self.stage1.append(blk)
            cur_pattern_idx += 1

        self.merge1 = Merging(dims[0], dims[1])

        # Stage 2
        self.stage2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            tp = pattern[cur_pattern_idx]
            if tp == 'L':
                current_local_type = self.local_type_list[cur_local_type_idx]
                current_window_size = self.window_sizes_list[cur_window_size_idx]
                if current_local_type == 'non_overlapping':
                    blk = NOLMWA(dims[1], heads[1], mlp_ratio, current_window_size, self.drop_rate)
                elif current_local_type == 'swin':
                    do_shift = (i % 2 == 1)
                    shift_size = (current_window_size[0] // 2, current_window_size[1] // 2) if do_shift else (0, 0)
                    blk = SWLMWA(dims[1], heads[1], mlp_ratio, current_window_size, self.drop_rate, shift_size)
                elif current_local_type == 'deformable':
                    blk = DLMWA(dims[1], heads[1], mlp_ratio, self.n_points, self.offset_scale, debug=False)
                elif current_local_type == 'conv':
                    blk = LocalMixingConv(dims[1], kernel_size=current_window_size, mlp_ratio=mlp_ratio, drop=self.drop_rate)
                else:
                     raise ValueError(f"Unknown local_type '{current_local_type}' for block {cur_pattern_idx}")
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else: # Global mixing
                blk = GlobalMixing(dims[1], heads[1], mlp_ratio, self.drop_rate)
            self.stage2.append(blk)
            cur_pattern_idx += 1

        self.merge2 = Merging(dims[1], dims[2])

        # Stage 3
        self.stage3 = nn.ModuleList()
        for i in range(num_blocks[2]):
            tp = pattern[cur_pattern_idx]
            if tp == 'L':
                current_local_type = self.local_type_list[cur_local_type_idx]
                current_window_size = self.window_sizes_list[cur_window_size_idx]
                if current_local_type == 'non_overlapping':
                    blk = NOLMWA(dims[2], heads[2], mlp_ratio, current_window_size, self.drop_rate)
                elif current_local_type == 'swin':
                    do_shift = (i % 2 == 1)
                    shift_size = (current_window_size[0] // 2, current_window_size[1] // 2) if do_shift else (0, 0)
                    blk = SWLMWA(dims[2], heads[2], mlp_ratio, current_window_size, self.drop_rate, shift_size)
                elif current_local_type == 'deformable':
                    blk = DLMWA(dims[2], heads[2], mlp_ratio, self.n_points, self.offset_scale, debug=False)
                elif current_local_type == 'conv':
                    blk = LocalMixingConv(dims[2], kernel_size=current_window_size, mlp_ratio=mlp_ratio, drop=self.drop_rate)
                else:
                    raise ValueError(f"Unknown local_type '{current_local_type}' for block {cur_pattern_idx}")
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else: # Global mixing
                blk = GlobalMixing(dims[2], heads[2], mlp_ratio, self.drop_rate)
            self.stage3.append(blk)
            cur_pattern_idx += 1

        self.combine = Combining(dims[2], d3, drop=self.drop_rate)
        self.head = nn.Linear(d3, vocab_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        Uses Kaiming uniform initialization for convolutional and linear layers,
        constant initialization for batch norm and layer norm,
        and truncated normal for positional embeddings.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, PatchEmbedSVTR):
                        # Initialize convolution layers within PatchEmbedSVTR
                        for conv in [m.conv1, m.conv2]:  # Adjust based on actual structure
                            if isinstance(conv, nn.Conv2d):
                                nn.init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
                                if conv.bias is not None:
                                    nn.init.constant_(conv.bias, 0)

        # Initialize positional embedding with truncated normal distribution
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)


    def forward(self, x):
        """
        Forward pass of the SVTR model.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            torch.Tensor: Output logits for each character at each timestep (B, W, vocab_size).
        """
        # x: Input image tensor (B, C, H, W)
        B = x.shape[0]

        # Patch Embedding
        # Input shape: (B, C, H, W) e.g., (B, 3, 64, 512)
        x = self.patch_embed(x)
        # Add positional embedding
        x = x + self.pos_embed
        # Output shape: (B, N, D) where N is number of patches, D is embed_dim[0]
        # N = (H/4) * (W/4)
        h, w = self.patch_embed.patches_resolution # Spatial dimensions after patch embedding (H/4, W/4)

        # Build the model stages based on the pattern and local_type
        cur_pattern_idx = 0
        cur_local_type_idx = 0
        cur_window_size_idx = 0

        # Stage 1
        # Input shape: (B, N, embed_dims[0]) e.g., (B, 2048, 64) from patches_resolution (16, 128)
        for i in range(len(self.stage1)):
            blk = self.stage1[i]
            tp = config['pattern'][cur_pattern_idx] # Get pattern from config
            if tp == 'L':
                # Local mixing requires spatial dimensions h, w
                x = blk(x, h, w)
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else:
                # Global mixing operates on the sequence (B, N, D)
                x = blk(x)
            cur_pattern_idx += 1
            # Output shape after each block: (B, N, embed_dims[0])

        # Merge 1
        # Input shape: (B, N, embed_dims[0])
        x, h, w = self.merge1(x, h, w)
        # Output shape: (B, N_new, embed_dims[1]) and new spatial dimensions (h, w)
        # Merging reduces height by 2, keeps width (approximately due to convolution stride)
        # e.g., (B, (h//2)*w, embed_dims[1])

        # Stage 2
        # Input shape: (B, N_new, embed_dims[1])
        for i in range(len(self.stage2)):
            blk = self.stage2[i]
            tp = config['pattern'][cur_pattern_idx] # Get pattern from config
            if tp == 'L':
                x = blk(x, h, w)
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else:
                x = blk(x)
            cur_pattern_idx += 1
            # Output shape after each block: (B, N_new, embed_dims[1])

        # Merge 2
        # Input shape: (B, N_new, embed_dims[1])
        x, h, w = self.merge2(x, h, w)
        # Output shape: (B, N_new_new, embed_dims[2]) and new spatial dimensions (h, w)
        # Merging reduces height by 2, keeps width

        # Stage 3
        # Input shape: (B, N_new_new, embed_dims[2])
        for i in range(len(self.stage3)):
            blk = self.stage3[i]
            tp = config['pattern'][cur_pattern_idx] # Get pattern from config
            if tp == 'L':
                x = blk(x, h, w)
                cur_local_type_idx += 1
                cur_window_size_idx += 1
            else:
                x = blk(x)
            cur_pattern_idx += 1
            # Output shape after each block: (B, N_new_new, embed_dims[2])

        # Combine -> collapse height to sequence of width length
        # Input shape: (B, N_final, embed_dims[2]) where N_final is h*w after last merge
        x = self.combine(x, h, w)
        # Output shape: (B, w, d3)
        # After combining, sequence length is the final width (w), dimension is d3.

        # Final Linear Head (Classifier/Decoder)
        # Input shape: (B, w, d3)
        x = self.head(x)
        # Output shape: (B, w, vocab_size)
        # This is the output sequence of logits for each character at each timestep (width position).

        return x

if __name__ == "__main__":
    config = {
        # Dataset parameteres
        'on_colab': False,   # else Kaggle
        'use_drive': False, # else Kaggle
        # 'dataset_dir': '/content/Arabic_English_OCR_Dataset', # Kaggle
        'dataset_dir': '/kaggle/input/arabic-english-ocr-dataset/Arabic_English_OCR_Dataset', # Drive
        'ar_dir': 'ar',
        'en_dir': 'en',
        'labels_file': 'labels.txt',
        'max_samples': 80000,  # full is 200000
        'max_text_length': 32,              # Maximum text sequence length
        'permissible_chars': set(
                        " !\"#$%&'()*+,-./:;=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_`abcdefghijklmnopqrstuvwxyz{|}،؛؟٫٬٭"
                        "0123456789ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ٠١٢٣٤٥٦٧٨٩"
                        "‘’“”"
                    ),
        'vocab_size': 162,                  # Will be updated after dataset analysis
        'train_split_percentage': 0.8,      # Percentage for training set
        'val_split_percentage': 0.1,        # Percentage for validation set
        'test_split_percentage': 0.1,       # Percentage for test set
        'remove_bad_examples': True,        # remove examples with not permissible chars else remove these characters instead in the text
        'do_analysis': True,               # To do Analysis on the dataset (text --> lengths & chars, images --> shape)

        # Image parameters - SVTR standard
        'img_height': 64,  # SVTR uses 64 height
        'img_width': 512,  # SVTR uses 256 width, but seeding the mean width of dataset we use 512
        'channels': 3,

        # Model Parameters - SVTR Large
        'embed_dims': [128, 256, 384],
        'd3': 512,
        'heads': [4, 8, 12],                # heads chosen such that embed_dim / num_heads == 32 (nice head dim)
        'num_blocks': [3, 6, 3],            # [3, 12, 3]
        'pattern': ['L'] * 6 + ['G'] * 6,   #['L'] * 9 + ['G'] * 9
        #['non_overlapping', 'non_overlapping', 'deformable'] * 3 + ['conv'] * 9, #non_overlapping,swin,deformable,conv
        'local_type': ['non_overlapping', 'non_overlapping', 'deformable'] * 2 + ['conv'] * 6 ,
        'window_sizes': [(7,11)] * 6 + [(3,3)] * 6, # [(7,11)] * 9 + [(3,3)] * 9
        'mlp_ratio': 2,
        'dropout_rate': 0.1,
        'n_points': 9,
        'offset_scale': 4.0,


        # Training parameters
        'num_epochs': 20,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'warmup_epochs': 5,
        'gradient_clip': 1.0,

        # Augmentation parameters
        'aug_prob': 0.7,
        'rotation_limit': 5,
        'blur_limit': 3,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        # 'dataset_mean': [0.485, 0.456, 0.406],       #--> ImageNet Values
        # 'dataset_std': [0.229, 0.224, 0.225],        #--> ImageNet Values
        'dataset_mean': [0.615, 0.617, 0.616],       #--> this dataset Values
        'dataset_std': [0.271, 0.276, 0.273],        #--> this dataset Values

        # Other parameters
        'save_path_directory': './arabic_ocr_checkpoints',
        # 'load_model_path': kagglehub.model_download("ahmedkamal75/arabic_ppocr_1.0/pyTorch/default"),
        'beam_size': 4,  # For beam search during inference
        'SEED': 42,

        # DataLoader parameters
        'dataloader_params': {
            'batch_size': 32, # Increased batch size for visualization
            'num_workers': 4,
            'pin_memory': True, # Set to False when not using CUDA
            # Add other DataLoader parameters here if needed, e.g.,
            'persistent_workers': True, # Use with num_workers > 0
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    model = SVTR(
        img_size=(config['img_height'], config['img_width']),
        in_chans=config['channels'],
        vocab_size=config['vocab_size'],
        local_type=config['local_type'], # Use the list from config
        embed_dims=config['embed_dims'], # Use embed_dims from config
        heads=config['heads'], # Use heads from config
        mlp_ratio=config['mlp_ratio'], # Use mlp_ratio from config
        window_sizes=config['window_sizes'], # Use window_sizes from config
        num_blocks=config['num_blocks'], # Use num_blocks from config
        pattern=config['pattern'], # Use pattern from config
        drop=config['dropout_rate'], # Use dropout_rate from config
        n_points=config['n_points'], # Use n_points from config
        offset_scale=config['offset_scale'], # Use offset_scale from config
    ).to(config['device']).eval()

    dummy = torch.randn(2, config['channels'], config['img_height'], config['img_width'], device=config['device'])
    with torch.no_grad():
        out = model(dummy)

    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    # The expected output shape is (batch_size, final_width, vocab_size)
    # From the summary, the final width is 128.
    assert out.shape == torch.zeros((2, config['img_width'] // 4, config['vocab_size'])).shape, "Output shape does not match expected shape!"
    print("Test passed: Output shape matches expected shape.")

    summary(model, input_size=dummy.shape)






