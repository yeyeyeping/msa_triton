"""TND MiniMax index scores with a deterministic recomputation backward.

The portable kernels use FP32 arithmetic and no device-specific atomics. BF16
and FP16 inputs are widened exactly before launch, including on the target
device. This also makes the actual kernels executable in Triton's CPU
interpreter, which does not implement BF16 pointer arithmetic. There is no
eager fallback. GPU/NPU compilation and performance must be validated on the
target device separately from interpreter correctness.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from ._addressing import tnd_offset


@triton.jit
def _merge_pairs(a_hi, a_lo, b_hi, b_lo):
    """TwoSum merge retaining the residual used to disambiguate max winners."""
    summed = a_hi + b_hi
    virtual = summed - a_hi
    error = (a_hi - (summed - virtual)) + (b_hi - virtual)
    low = (a_lo + b_lo) + error
    high = summed + low
    low_virtual = high - summed
    remainder = (summed - (high - low_virtual)) + (low - low_virtual)
    return high, remainder


@triton.jit
def _reduce_dimension_pairs(high, low, ROW_TILE: tl.constexpr, HALF: tl.constexpr):
    positions = 2 * tl.arange(0, HALF)[None, :] + tl.zeros((ROW_TILE, HALF), tl.int32)
    left_hi = tl.gather(high, positions, 1)
    right_hi = tl.gather(high, positions + 1, 1)
    left_lo = tl.gather(low, positions, 1)
    right_lo = tl.gather(low, positions + 1, 1)
    return _merge_pairs(left_hi, left_lo, right_hi, right_lo)


@triton.jit
def _dot_pairs(left, right, ROW_TILE: tl.constexpr, DIM_TILE: tl.constexpr, LOG_DIM: tl.constexpr):
    """A fixed adjacent-pair reduction independent of the query/key tile size.

    Mantissa splitting gives the FP32 product residual without relying on FMA
    (the CPU interpreter does not fuse tl.fma). The launch must disable fusion:
    TwoSum/TwoProduct identities require each written FP32 rounding step.
    """
    high = left * right
    # Keep 12 significant bits in each high part. BF16/FP16 inputs are already
    # exact high parts; FP32 inputs additionally need the low-product residual.
    left_hi = (left.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    right_hi = (right.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    left_lo = left - left_hi
    right_lo = right - right_hi
    error1 = high - left_hi * right_hi
    error2 = error1 - left_lo * right_hi
    error3 = error2 - left_hi * right_lo
    low = left_lo * right_lo - error3
    for level in tl.static_range(0, LOG_DIM):
        high, low = _reduce_dimension_pairs(high, low, ROW_TILE, DIM_TILE >> (level + 1))
    return tl.sum(high, axis=1), tl.sum(low, axis=1)


@triton.jit
def _directed_dot(left, right, ROW_TILE: tl.constexpr, DIM_TILE: tl.constexpr,
                  SCALE: tl.constexpr, LOG_DIM: tl.constexpr):
    high, low = _dot_pairs(left, right, ROW_TILE, DIM_TILE, LOG_DIM)
    # Multiplication by a finite nonzero scale preserves ordering (or reverses
    # it when negative). Do not discard the residual before selecting winners.
    if SCALE < 0:
        high, low = -high, -low
    elif SCALE == 0:
        high, low = tl.full((ROW_TILE,), 0, tl.float32), tl.full((ROW_TILE,), 0, tl.float32)
    return high, low


@triton.jit
def _sequence_bounds(CU, token, N_SEQS: tl.constexpr, SEQ_TILE: tl.constexpr):
    seq = tl.arange(0, SEQ_TILE).to(tl.int64)
    # Keep every formed pointer inside CU, even for padded lanes. Reduce the
    # boundary values themselves rather than reloading CU through a scalar
    # index produced by a vector reduction. This is an Ascend lowering
    # simplification; the original masked-load semantics were valid Triton.
    safe_seq = tl.minimum(seq, N_SEQS - 1)
    starts = tl.load(CU + safe_seq, seq < N_SEQS, other=0).to(tl.int64)
    ends = tl.load(CU + safe_seq + 1, seq < N_SEQS, other=0).to(tl.int64)
    selected = (seq < N_SEQS) & (token >= starts)
    begin = tl.max(tl.where(selected, starts, 0), axis=0)
    end = tl.max(tl.where(selected, ends, 0), axis=0)
    return begin, end


@triton.jit
def _index_score_forward(
    Q, K, CU, SCORES, MAX_HI, MAX_LO, TIE_INFO,
    G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
    BLOCK: tl.constexpr, SCALE: tl.constexpr,
    N_SEQS: tl.constexpr, SEQ_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr, DIM_TILE: tl.constexpr, LOG_DIM: tl.constexpr,
    SAVE_STATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    token, group = row // G, row % G
    begin, end = _sequence_bounds(CU, token, N_SEQS, SEQ_TILE)
    output_offset = tnd_offset(token, group, G, NB) + block
    if block * BLOCK > token - begin:
        tl.store(SCORES + output_offset, -float("inf"))
        if SAVE_STATE:
            tl.store(MAX_HI + output_offset, -float("inf"))
            tl.store(MAX_LO + output_offset, 0)
            tl.store(TIE_INFO + output_offset, 0)
    else:
        key_offset = tl.arange(0, KEY_TILE).to(tl.int64)
        dim = tl.arange(0, DIM_TILE).to(tl.int64)
        safe_dim = tl.minimum(dim, D - 1)
        keys = begin + block * BLOCK + key_offset
        valid = (key_offset < BLOCK) & (keys < end) & (keys <= token)
        safe_keys = tl.minimum(keys, tl.minimum(end - 1, token))
        q = tl.load(Q + tnd_offset(token, group, G, D) + safe_dim, dim < D, other=0)
        k = tl.load(K + tnd_offset(safe_keys[:, None], 0, 1, D) + safe_dim[None, :],
                    valid[:, None] & (dim[None, :] < D), other=0)
        high, low = _directed_dot(k, q[None, :], KEY_TILE, DIM_TILE, SCALE, LOG_DIM)
        maximum_hi = tl.max(tl.where(valid, high, -float("inf")), axis=0)
        maximum_lo = tl.max(tl.where(valid & (high == maximum_hi), low, -float("inf")), axis=0)

        scale_magnitude: tl.constexpr = SCALE if SCALE >= 0 else -SCALE
        maximum = maximum_hi * scale_magnitude + maximum_lo * scale_magnitude
        tl.store(SCORES + output_offset, maximum)
        if SAVE_STATE:
            winners = valid & (high == maximum_hi) & (low == maximum_lo)
            count = tl.sum(winners.to(tl.int32), axis=0)
            first = tl.min(tl.where(winners, key_offset, KEY_TILE), axis=0)
            # Positive: unique winner offset + 1. Negative: number of tied maxima.
            info = tl.where(count == 1, first + 1, -count)
            tl.store(MAX_HI + output_offset, maximum_hi)
            tl.store(MAX_LO + output_offset, maximum_lo)
            tl.store(TIE_INFO + output_offset, info)


@triton.jit
def _index_score_backward_q(
    Q, K, CU, MAX_HI, MAX_LO, TIE_INFO, DS, DQ,
    G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
    BLOCK: tl.constexpr, SCALE: tl.constexpr,
    N_SEQS: tl.constexpr, SEQ_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr, DIM_TILE: tl.constexpr, LOG_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token, group = row // G, row % G
    begin, end = _sequence_bounds(CU, token, N_SEQS, SEQ_TILE)
    key_offset = tl.arange(0, KEY_TILE).to(tl.int64)
    dim = tl.arange(0, DIM_TILE).to(tl.int64)
    safe_dim = tl.minimum(dim, D - 1)
    q = tl.load(Q + tnd_offset(token, group, G, D) + safe_dim, dim < D, other=0)
    grad = tl.full((DIM_TILE,), 0, tl.float32)
    correction = tl.full((DIM_TILE,), 0, tl.float32)
    for block in range(tl.cdiv(token - begin + 1, BLOCK)):
        offset = tnd_offset(token, group, G, NB) + tl.cast(block, tl.int64)
        info = tl.load(TIE_INFO + offset)
        ds = tl.load(DS + offset)
        if info > 0:
            key = begin + tl.cast(block, tl.int64) * BLOCK + info.to(tl.int64) - 1
            winner_k = tl.load(K + tnd_offset(key, 0, 1, D) + safe_dim, dim < D, other=0)
            addition = winner_k * (ds * SCALE)
        else:
            keys = begin + tl.cast(block, tl.int64) * BLOCK + key_offset
            valid = (key_offset < BLOCK) & (keys < end) & (keys <= token)
            safe_keys = tl.minimum(keys, tl.minimum(end - 1, token))
            k = tl.load(K + tnd_offset(safe_keys[:, None], 0, 1, D) + safe_dim[None, :],
                        valid[:, None] & (dim[None, :] < D), other=0)
            high, low = _directed_dot(k, q[None, :], KEY_TILE, DIM_TILE, SCALE, LOG_DIM)
            maximum_hi = tl.load(MAX_HI + offset)
            maximum_lo = tl.load(MAX_LO + offset)
            winner = (high == maximum_hi) & (low == maximum_lo)
            weight = tl.where(valid & winner, ds * SCALE / tl.maximum(-info, 1), 0)
            addition = tl.sum(k * weight[:, None], axis=0)
        adjusted = addition - correction
        updated = grad + adjusted
        correction = (updated - grad) - adjusted
        grad = updated
    tl.store(DQ + tnd_offset(token, group, G, D) + safe_dim, grad, dim < D)


@triton.jit
def _index_score_backward_k_group(
    Q, K, CU, MAX_HI, MAX_LO, TIE_INFO, DS, DK_GROUP,
    G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
    BLOCK: tl.constexpr, SCALE: tl.constexpr,
    N_SEQS: tl.constexpr, SEQ_TILE: tl.constexpr,
    QUERY_TILE: tl.constexpr, DIM_TILE: tl.constexpr, LOG_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token, group = row // G, row % G
    begin, end = _sequence_bounds(CU, token, N_SEQS, SEQ_TILE)
    block = (token - begin) // BLOCK
    query_offset = tl.arange(0, QUERY_TILE).to(tl.int64)
    dim = tl.arange(0, DIM_TILE).to(tl.int64)
    safe_dim = tl.minimum(dim, D - 1)
    k = tl.load(K + tnd_offset(token, 0, 1, D) + safe_dim, dim < D, other=0)
    grad = tl.full((DIM_TILE,), 0, tl.float32)
    correction = tl.full((DIM_TILE,), 0, tl.float32)
    for start in range(token, end, QUERY_TILE):
        queries = tl.cast(start, tl.int64) + query_offset
        valid = queries < end
        safe_queries = tl.minimum(queries, end - 1)
        q = tl.load(Q + tnd_offset(safe_queries[:, None], group, G, D) + safe_dim[None, :],
                    valid[:, None] & (dim[None, :] < D), other=0)
        high, low = _directed_dot(q, k[None, :], QUERY_TILE, DIM_TILE, SCALE, LOG_DIM)
        offset = tnd_offset(safe_queries, group, G, NB) + block
        maximum_hi = tl.load(MAX_HI + offset, valid, other=float("inf"))
        maximum_lo = tl.load(MAX_LO + offset, valid, other=0)
        info = tl.load(TIE_INFO + offset, valid, other=0)
        count = tl.where(info > 0, 1, -info)
        ds = tl.load(DS + offset, valid, other=0)
        winner = tl.where(info > 0,
                          token - begin - block * BLOCK == info.to(tl.int64) - 1,
                          (high == maximum_hi) & (low == maximum_lo))
        weight = tl.where(valid & winner, ds * SCALE / tl.maximum(count, 1), 0)
        addition = tl.sum(q * weight[:, None], axis=0)
        adjusted = addition - correction
        updated = grad + adjusted
        correction = (updated - grad) - adjusted
        grad = updated
    tl.store(DK_GROUP + tnd_offset(token, group, G, D) + safe_dim, grad, dim < D)


@triton.jit
def _index_score_reduce_k(
    DK_GROUP, DK, G: tl.constexpr, D: tl.constexpr,
    GROUP_TILE: tl.constexpr, DIM_TILE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, GROUP_TILE).to(tl.int64)
    dim = tl.arange(0, DIM_TILE).to(tl.int64)
    safe_group = tl.minimum(group, G - 1)
    safe_dim = tl.minimum(dim, D - 1)
    values = tl.load(DK_GROUP + tnd_offset(token, safe_group[:, None], G, D) + safe_dim[None, :],
                     (group[:, None] < G) & (dim[None, :] < D), other=0)
    tl.store(DK + tnd_offset(token, 0, 1, D) + safe_dim, tl.sum(values, axis=0), dim < D)


class _IndexScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index_q, index_k, cu_seqlens, max_seqlen, block_size, scale, save_state):
        # Keep the same arithmetic and memory representation on CPU and device.
        q = index_q.to(torch.float32).contiguous()
        k = index_k.to(torch.float32).contiguous()
        cu = cu_seqlens.contiguous()
        total, groups, dim = q.shape
        blocks = triton.cdiv(max_seqlen, block_size)
        scores = torch.empty((total, groups, blocks), dtype=torch.float32, device=q.device)
        # The no-grad indexer path allocates only its public scores. Unused
        # state pointers are compiled away by SAVE_STATE, not read or written.
        tie_info = torch.empty_like(scores, dtype=torch.int32) if save_state else scores
        maximum_hi = torch.empty_like(scores) if save_state else scores
        maximum_lo = torch.empty_like(scores) if save_state else scores
        constants = dict(
            G=groups, D=dim, NB=blocks, BLOCK=block_size, SCALE=scale,
            N_SEQS=cu.numel() - 1, SEQ_TILE=triton.next_power_of_2(cu.numel() - 1),
            DIM_TILE=triton.next_power_of_2(dim), LOG_DIM=triton.next_power_of_2(dim).bit_length() - 1,
        )
        if total:
            _index_score_forward[(total * groups, blocks)](
                q, k, cu, scores, maximum_hi, maximum_lo, tie_info,
                KEY_TILE=triton.next_power_of_2(block_size), SAVE_STATE=save_state,
                enable_fp_fusion=False, **constants,
            )
        if save_state:
            ctx.save_for_backward(q, k, cu, maximum_hi, maximum_lo, tie_info)
        ctx.constants = constants
        ctx.q_dtype, ctx.k_dtype = index_q.dtype, index_k.dtype
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        q, k, cu, maximum_hi, maximum_lo, tie_info = ctx.saved_tensors
        total, groups, dim = q.shape
        ds = grad_scores.to(torch.float32).contiguous()
        dq = torch.empty_like(q) if ctx.needs_input_grad[0] else None
        dk = torch.empty_like(k) if ctx.needs_input_grad[1] else None
        if total and dq is not None:
            _index_score_backward_q[(total * groups,)](
                q, k, cu, maximum_hi, maximum_lo, tie_info, ds, dq,
                KEY_TILE=triton.next_power_of_2(ctx.constants["BLOCK"]),
                enable_fp_fusion=False, **ctx.constants,
            )
        if total and dk is not None:
            dk_group = torch.empty_like(q)
            _index_score_backward_k_group[(total * groups,)](
                q, k, cu, maximum_hi, maximum_lo, tie_info, ds, dk_group,
                QUERY_TILE=32, enable_fp_fusion=False, **ctx.constants,
            )
            _index_score_reduce_k[(total,)](
                dk_group, dk, G=groups, D=dim,
                GROUP_TILE=triton.next_power_of_2(groups),
                DIM_TILE=triton.next_power_of_2(dim),
            )
        return (
            None if dq is None else dq.to(ctx.q_dtype),
            None if dk is None else dk.to(ctx.k_dtype),
            None, None, None, None, None,
        )


def m3_index_score(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int = 128,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return per-group causal block maxima as FP32 ``[T, G, Bmax]``.

    ``index_q`` is ``[T,G,Di]`` and ``index_k`` is ``[T,1,Di]``.
    Blocks are relative to the start of each packed sequence. Future and
    out-of-sequence blocks return ``-inf`` and have zero derivative. Max ties
    share the gradient equally, matching ``torch.amax``. Winner comparisons use
    compensated FP32 dot products before the public FP32 score is rounded, so
    equal public scores need not denote a mathematical max tie. The discrete top-k
    stage is deliberately separate.

    Backward supports first-order gradients. CPU execution requires
    ``TRITON_INTERPRET=1`` to be set before importing this module. Inputs and
    unmasked dot products must be finite; NaN/Inf behavior is undefined.
    Two-component FP32 arithmetic reduces cancellation errors, but is not exact
    FP64 arithmetic: overflow, underflow, or arbitrary dimensions and dynamic
    ranges are not guaranteed to reproduce FP64 winner sets.
    Metadata validation currently synchronizes device boundaries to the host.
    """
    if index_q.ndim != 3 or index_k.ndim != 3:
        raise ValueError("index_q and index_k must be rank-3 TND tensors")
    total, groups, dim = index_q.shape
    if groups < 1 or dim < 1 or index_k.shape != (total, 1, dim):
        raise ValueError("expected index_q[T,G,Di] and index_k[T,1,Di] with G, Di > 0")
    if index_q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("index_q must have BF16, FP16, or FP32 dtype")
    if index_k.dtype != index_q.dtype or index_k.device != index_q.device:
        raise ValueError("index_q and index_k must share dtype and device")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be a rank-1 int32 tensor with at least two entries")
    if cu_seqlens.device != index_q.device:
        raise ValueError("cu_seqlens must be on the input device")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    if total > 2147483647 or cu_seqlens.numel() - 1 > 2147483647:
        raise ValueError("packed token and sequence counts must fit int32 metadata")
    if block_size > 2147483647 or max_seqlen > 2147483647:
        raise ValueError("block_size and max_seqlen must fit int32 metadata")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if index_q.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("CPU Triton execution requires TRITON_INTERPRET=1 before import")
    boundaries = cu_seqlens.detach().cpu().tolist()
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if boundaries[0] != 0 or boundaries[-1] != total or any(n < 0 for n in lengths):
        raise ValueError("cu_seqlens must span T with nondecreasing boundaries starting at zero")
    if max(lengths) > max_seqlen:
        raise ValueError("max_seqlen is smaller than a packed sequence")
    save_state = torch.is_grad_enabled() and (index_q.requires_grad or index_k.requires_grad)
    return _IndexScore.apply(index_q, index_k, cu_seqlens, max_seqlen, block_size, float(scale), save_state)
