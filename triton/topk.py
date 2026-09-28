"""Native torch selection stage, including on NPU.

This stage deliberately uses torch.topk rather than a Triton kernel. It
preserves the independent score/selection/attention boundary and has no
backward because its only output is an integer index tensor.
"""

import torch


@torch.no_grad()
def m3_topk(scores, cu_seqlens, *, block_size=128, topk_blocks=16, local_blocks=1):
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [T,G,Bmax]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be an int32 vector with at least two entries")
    if cu_seqlens.device != scores.device:
        raise ValueError("scores and cu_seqlens must share a device")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(topk_blocks, int) or topk_blocks < 0:
        raise ValueError("topk_blocks must be a nonnegative integer")
    if not isinstance(local_blocks, int) or not 0 <= local_blocks <= topk_blocks:
        raise ValueError("local_blocks must satisfy 0 <= local_blocks <= topk_blocks")
    total, groups, blocks = scores.shape
    boundaries = cu_seqlens.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != total or any(b < a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must start at 0, be nondecreasing, and end at T")
    if groups <= 0 or blocks <= 0 or any(b - a > blocks * block_size for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("scores need positive G and enough block slots for every sequence")
    result = torch.full((total, groups, topk_blocks), -1, dtype=torch.int32, device=scores.device)
    if total == 0 or topk_blocks == 0:
        return result
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    starts = torch.repeat_interleave(cu_seqlens[:-1], lengths.to(torch.int64), output_size=total)
    positions = torch.arange(total, device=scores.device) - starts
    query_blocks = positions // block_size
    block_ids = torch.arange(blocks, device=scores.device)
    valid = block_ids[None, :] <= query_blocks[:, None]
    ranked = scores.masked_fill(~valid[:, None, :], -float("inf"))
    if local_blocks:
        local = valid & (block_ids[None, :] > query_blocks[:, None] - local_blocks)
        ranked = ranked.masked_fill(local[:, None, :], float("inf"))
    count = min(topk_blocks, blocks)
    values, selected = torch.topk(ranked, count, dim=-1, sorted=True)
    result[..., :count] = selected.masked_fill(values == -float("inf"), -1).to(torch.int32)
    return result
