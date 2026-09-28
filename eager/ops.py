# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
# Modifications: split operators and differentiable TND/BSND adapters.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extracted MiniMax M3 eager operations; see PROVENANCE.md for source.

Public inputs are TND. The floating-point operations retain the upstream
batched, padded implementation, including its main-attention rounding points.
The FP64 oracle is a separate implementation under tests/.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ..layout import pack_bs, sequence_lengths, unpack_tnd


def _check_block_size(block_size: int) -> None:
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")


def _check_qk(q: torch.Tensor, k: torch.Tensor, *, index: bool) -> None:
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("Q/K must be [T,H,D] with matching T and D")
    if min(q.shape[1:]) <= 0 or min(k.shape[1:]) <= 0:
        raise ValueError("head counts and head dimensions must be positive")
    if index and k.shape[1] != 1:
        raise ValueError("index K must have exactly one head")
    if q.dtype != k.dtype or q.device != k.device or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Q/K must have the same FP16, BF16 or FP32 dtype and device")


def index_score_bsnd(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    valid: torch.Tensor,
    *,
    block_size: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """Upstream FP32 QK then block ``amax``, returning [B,S,G,blocks]."""
    batch, length, groups, _ = index_q.shape
    q = index_q.transpose(1, 2)
    k = index_k.transpose(1, 2)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    if scale != 1.0:
        scores = scores * scale
    positions = torch.arange(length, device=q.device)
    keep = positions[None, :] <= positions[:, None]
    keep = keep[None, None, :, :] & valid[:, None, None, :] & valid[:, None, :, None]
    scores = scores.masked_fill(~keep, float("-inf"))
    blocks = (length + block_size - 1) // block_size
    scores = F.pad(scores, (0, blocks * block_size - length), value=float("-inf"))
    scores = scores.reshape(batch, groups, length, blocks, block_size).amax(-1)
    return scores.transpose(1, 2)


def m3_index_score(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int = 128,
    scale: float = 1.0,
) -> torch.Tensor:
    """Per-group causal block max scores, FP32 [T,G,ceil(max_seqlen/B)].

    Autograd follows torch.amax: ties within a block split gradients equally;
    masked tokens never receive gradients.
    """
    _check_block_size(block_size)
    _check_qk(index_q, index_k, index=True)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    q, valid = unpack_tnd(index_q, cu_seqlens, max_seqlen)
    k, _ = unpack_tnd(index_k, cu_seqlens, max_seqlen)
    scores = index_score_bsnd(q, k, valid, block_size=block_size, scale=scale)
    return pack_bs(scores, cu_seqlens)


@torch.no_grad()
def m3_topk(
    scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    block_size: int = 128,
    topk_blocks: int = 16,
    local_blocks: int = 1,
) -> torch.Tensor:
    """Discrete local-boost/top-k with fixed K slots and -1 right padding."""
    _check_block_size(block_size)
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [T,G,blocks]")
    if scores.shape[1] <= 0:
        raise ValueError("scores must have a positive group count")
    if scores.device != cu_seqlens.device:
        raise ValueError("scores and cu_seqlens must be on the same device")
    if not isinstance(topk_blocks, int) or topk_blocks < 0:
        raise ValueError("topk_blocks must be a nonnegative integer")
    if not isinstance(local_blocks, int) or not 0 <= local_blocks <= topk_blocks:
        raise ValueError("require 0 <= local_blocks <= topk_blocks")
    lengths = sequence_lengths(cu_seqlens, scores.shape[0])
    if scores.shape[-1] < (max(lengths) + block_size - 1) // block_size:
        raise ValueError("scores has too few block columns")
    positions = torch.cat([torch.arange(n, device=scores.device) for n in lengths])
    blocks = torch.arange(scores.shape[-1], device=scores.device)
    query_block = positions // block_size
    valid = blocks[None, None, :] <= query_block[:, None, None]
    boosted = scores.masked_fill(~valid, float("-inf"))
    if local_blocks:
        local = (blocks[None, None, :] > query_block[:, None, None] - local_blocks) & valid
        boosted = boosted.masked_fill(local, float("inf"))
    count = min(topk_blocks, scores.shape[-1])
    values, indices = boosted.topk(count, dim=-1)
    indices = indices.masked_fill(values == float("-inf"), -1).to(torch.int32)
    return F.pad(indices, (0, topk_blocks - count), value=-1)


def build_block_keep_mask(
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_query_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Expand [B,S,G,K] indices into a bool [B,Hq,S,S] attention mask."""
    batch, length, groups, _ = indices.shape
    blocks = (length + block_size - 1) // block_size
    indices = indices.transpose(1, 2).long()
    safe = indices.masked_fill((indices < 0) | (indices >= blocks), blocks)
    keep_blocks = torch.zeros((batch, groups, length, blocks + 1), dtype=torch.bool, device=indices.device)
    keep_blocks.scatter_(-1, safe, True)
    block_keep = keep_blocks[..., :blocks].repeat_interleave(block_size, dim=-1)[..., :length]
    block_keep = block_keep.repeat_interleave(num_query_heads // groups, dim=1)
    positions = torch.arange(length, device=indices.device)
    causal = positions[None, :] <= positions[:, None]
    return block_keep & causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]


def sparse_attention_bsnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native dtype QK/PV with FP32 softmax, preserving upstream rounding."""
    heads, groups = q.shape[2], k.shape[2]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(heads // groups, dim=1)
    v = v.transpose(1, 2).repeat_interleave(heads // groups, dim=1)
    keep = build_block_keep_mask(indices, valid, num_query_heads=heads, block_size=block_size)
    weights = torch.matmul(q, k.transpose(-1, -2)) * scale
    weights = weights.masked_fill(~keep, float("-inf"))
    # Avoid a NaN softmax graph for padding/all-invalid query rows. Their
    # probabilities, outputs, and all gradient contributions are exactly zero.
    has_keys = keep.any(-1, keepdim=True)
    safe_weights = torch.where(has_keys, weights, torch.zeros_like(weights))
    probs = F.softmax(safe_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    probs = probs.masked_fill(~has_keys, 0)
    out = torch.matmul(probs, v).transpose(1, 2).contiguous()
    lse = torch.logsumexp(safe_weights.float(), dim=-1)
    lse = lse.masked_fill(~has_keys.squeeze(-1), float("-inf"))
    return out, lse.transpose(1, 2).detach()


def m3_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int = 128,
    scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """TND wrapper for BSND sparse eager attention and its Q/K/V autograd."""
    _check_block_size(block_size)
    _check_qk(q, k, index=False)
    if v.shape != k.shape or v.dtype != q.dtype or v.device != q.device:
        raise ValueError("V must match K shape, dtype, and device")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query head count must be divisible by KV groups")
    if indices.ndim != 3 or indices.shape[:2] != k.shape[:2]:
        raise ValueError("indices must have shape [T,G,K]")
    if indices.dtype != torch.int32 or indices.device != q.device:
        raise ValueError("indices must be int32 and on the Q/K/V device")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    q_bs, valid = unpack_tnd(q, cu_seqlens, max_seqlen)
    k_bs, _ = unpack_tnd(k, cu_seqlens, max_seqlen)
    v_bs, _ = unpack_tnd(v, cu_seqlens, max_seqlen)
    ids_bs, _ = unpack_tnd(indices, cu_seqlens, max_seqlen, pad_value=-1)
    out, lse = sparse_attention_bsnd(q_bs, k_bs, v_bs, ids_bs, valid, block_size=block_size, scale=scale)
    out = pack_bs(out, cu_seqlens)
    return (out, pack_bs(lse, cu_seqlens)) if return_lse else out
