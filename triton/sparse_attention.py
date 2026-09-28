"""TND sparse GQA attention implemented with Triton.

One forward program owns a query/head and streams its selected blocks. Backward
recomputes probabilities from saved FP32 maxima/denominators; query programs own
dQ and key programs own dK/dV through a block-to-query CSR transpose. This avoids
floating-point atomics and dense attention workspaces. This is
a correctness-first implementation; A3 compilation and tuning are separate
from CPU interpreter validation.  No dense attention matrix is materialized.

Inputs are widened to FP32 before launch, including in interpreter mode.  This
avoids relying on the interpreter's BF16 pointer support and keeps forward and
backward arithmetic consistent.  Only the public output and input gradients
are cast back to the input dtype.  The extra storage is linear in input size.
"""

from __future__ import annotations

import functools
import math
import os

import torch

from msa_triton.layout import sequence_lengths

# Populated only when a Triton operation is requested.  Importing this module
# therefore does not require Triton or a GPU/NPU runtime.
tl = None
_merge_pairs = None
_reduce_pairs = None
_tnd_offset = None
_advance_offset = None


def _forward_kernel(
    Q, K, V, INDICES, CU, OUT, LSE, NORMALIZERS,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    TOPK: tl.constexpr, NSEQ: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_META: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    group = head // (H_Q // H_KV)
    meta_offsets = tl.arange(0, BLOCK_META).to(tl.int64)
    boundaries = tl.load(CU + meta_offsets, meta_offsets < NSEQ, other=0).to(tl.int64)
    seq_start = tl.max(tl.where((meta_offsets < NSEQ) & (boundaries <= token), boundaries, 0), 0)
    ds = tl.arange(0, BLOCK_D).to(tl.int64)
    ns = tl.arange(0, BLOCK_N).to(tl.int64)
    query = tl.load(Q + _tnd_offset(token, head, H_Q, D) + ds, ds < D, other=0)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in range(TOPK):
        block = tl.load(INDICES + (token * H_KV + group) * TOPK + slot).to(tl.int64)
        if (block >= 0) & (block <= (token - seq_start) // BLOCK_SIZE):
            for tile in range(tl.cdiv(BLOCK_SIZE, BLOCK_N)):
                block_offsets = tl.cast(tile, tl.int64) * BLOCK_N + ns
                keys = seq_start + block * BLOCK_SIZE + block_offsets
                valid = (block_offsets < BLOCK_SIZE) & (keys <= token)
                key = tl.load(
                    K + _tnd_offset(keys[:, None], group, H_KV, D) + ds[None, :],
                    valid[:, None] & (ds[None, :] < D), other=0,
                )
                value = tl.load(
                    V + _tnd_offset(keys[:, None], group, H_KV, D) + ds[None, :],
                    valid[:, None] & (ds[None, :] < D), other=0,
                )
                logits = tl.sum(key * query[None, :], 1) * SCALE
                logits = tl.where(valid, logits, float("-inf"))
                next_maximum = tl.maximum(maximum, tl.max(logits, 0))
                finite_maximum = tl.where(next_maximum == float("-inf"), 0.0, next_maximum)
                rescale = tl.exp(maximum - finite_maximum)
                probability = tl.exp(logits - finite_maximum)
                numerator = numerator * rescale + tl.sum(probability[:, None] * value, 0)
                denominator = denominator * rescale + tl.sum(probability, 0)
                maximum = next_maximum
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    output = numerator / safe_denominator
    logsumexp = tl.where(denominator > 0.0, maximum + tl.log(safe_denominator), float("-inf"))
    tl.store(OUT + _tnd_offset(token, head, H_Q, D) + ds, output, ds < D)
    tl.store(LSE + token * H_Q + head, logsumexp)
    tl.store(NORMALIZERS + (token * H_Q + head) * 2, maximum)
    tl.store(NORMALIZERS + (token * H_Q + head) * 2 + 1, denominator)


def _backward_kernel(
    Q, K, V, INDICES, CU, OUT, NORMALIZERS, DOUT, DQ, DK, DV, DK_CORRECTION, DV_CORRECTION, STATS,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    TOPK: tl.constexpr, NSEQ: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_META: tl.constexpr, ACCUMULATE_KV: tl.constexpr,
    COMPUTE_DQ: tl.constexpr, COMPUTE_DK: tl.constexpr,
    COMPUTE_DV: tl.constexpr, WRITE_STATS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    group = head // (H_Q // H_KV)
    meta_offsets = tl.arange(0, BLOCK_META).to(tl.int64)
    boundaries = tl.load(CU + meta_offsets, meta_offsets < NSEQ, other=0).to(tl.int64)
    seq_start = tl.max(tl.where((meta_offsets < NSEQ) & (boundaries <= token), boundaries, 0), 0)
    ds = tl.arange(0, BLOCK_D).to(tl.int64)
    ns = tl.arange(0, BLOCK_N).to(tl.int64)
    query_offsets = _tnd_offset(token, head, H_Q, D) + ds
    query = tl.load(Q + query_offsets, ds < D, other=0)
    if (COMPUTE_DQ or COMPUTE_DK) or (ACCUMULATE_KV and COMPUTE_DV):
        grad_output = tl.load(DOUT + query_offsets, ds < D, other=0)
    maximum = tl.load(NORMALIZERS + (token * H_Q + head) * 2)
    denominator = tl.load(NORMALIZERS + (token * H_Q + head) * 2 + 1)
    center = tl.full((), 0.0, tl.float32)
    if COMPUTE_DQ or COMPUTE_DK:
        output = tl.load(OUT + query_offsets, ds < D, other=0)
        center = tl.sum(output * grad_output, 0)
    centered_delta = tl.full((), 0.0, tl.float32)
    probability_mass = tl.full((), 1.0, tl.float32)
    grad_query = tl.full((BLOCK_D,), 0.0, tl.float32)
    correction = tl.full((BLOCK_D,), 0.0, tl.float32)
    if denominator > 0.0:
        # Refine the softmax derivative's center using selected probabilities.
        # Directly subtracting sum(O*dO) from large, close dP values loses
        # precision even when O is retained in FP32. The centered reduction
        # computes the same derivative while avoiding that cancellation.
        centered_delta = tl.full((), 0.0, tl.float32)
        delta_correction = tl.full((), 0.0, tl.float32)
        probability_mass = tl.full((), 0.0, tl.float32)
        mass_correction = tl.full((), 0.0, tl.float32)
        for slot in range(TOPK):
            block = tl.load(INDICES + (token * H_KV + group) * TOPK + slot).to(tl.int64)
            if (block >= 0) & (block <= (token - seq_start) // BLOCK_SIZE):
                for tile in range(tl.cdiv(BLOCK_SIZE, BLOCK_N)):
                    block_offsets = tl.cast(tile, tl.int64) * BLOCK_N + ns
                    keys = seq_start + block * BLOCK_SIZE + block_offsets
                    valid = (block_offsets < BLOCK_SIZE) & (keys <= token)
                    kv_offsets = _tnd_offset(keys[:, None], group, H_KV, D) + ds[None, :]
                    kv_mask = valid[:, None] & (ds[None, :] < D)
                    key = tl.load(K + kv_offsets, kv_mask, other=0)
                    logits = tl.sum(key * query[None, :], 1) * SCALE
                    probability = tl.exp(tl.where(valid, logits - maximum, float("-inf"))) / denominator
                    if COMPUTE_DQ or COMPUTE_DK:
                        value = tl.load(V + kv_offsets, kv_mask, other=0)
                        grad_probability = tl.sum(value * grad_output[None, :], 1)
                        addition = tl.sum(probability * (grad_probability - center), 0)
                        adjusted = addition - delta_correction
                        updated = centered_delta + adjusted
                        delta_correction = (updated - centered_delta) - adjusted
                        centered_delta = updated
                    mass_addition = tl.sum(probability, 0)
                    mass_adjusted = mass_addition - mass_correction
                    mass_updated = probability_mass + mass_adjusted
                    mass_correction = (mass_updated - probability_mass) - mass_adjusted
                    probability_mass = mass_updated
        if COMPUTE_DQ or COMPUTE_DK:
            centered_delta = centered_delta / probability_mass
        if COMPUTE_DQ or ACCUMULATE_KV:
            for slot in range(TOPK):
                block = tl.load(INDICES + (token * H_KV + group) * TOPK + slot).to(tl.int64)
                if (block >= 0) & (block <= (token - seq_start) // BLOCK_SIZE):
                    for tile in range(tl.cdiv(BLOCK_SIZE, BLOCK_N)):
                        block_offsets = tl.cast(tile, tl.int64) * BLOCK_N + ns
                        keys = seq_start + block * BLOCK_SIZE + block_offsets
                        valid = (block_offsets < BLOCK_SIZE) & (keys <= token)
                        kv_offsets = _tnd_offset(keys[:, None], group, H_KV, D) + ds[None, :]
                        kv_mask = valid[:, None] & (ds[None, :] < D)
                        key = tl.load(K + kv_offsets, kv_mask, other=0)
                        logits = tl.sum(key * query[None, :], 1) * SCALE
                        # Keep the mass refinement even for V-only gradients.
                        probability = tl.exp(tl.where(valid, logits - maximum, float("-inf"))) / denominator / probability_mass
                        if COMPUTE_DQ or COMPUTE_DK:
                            value = tl.load(V + kv_offsets, kv_mask, other=0)
                            grad_probability = tl.sum(value * grad_output[None, :], 1)
                            grad_logits = probability * ((grad_probability - center) - centered_delta) * SCALE
                        if COMPUTE_DQ:
                            addition = tl.sum(grad_logits[:, None] * key, 0)
                            adjusted = addition - correction
                            updated = grad_query + adjusted
                            correction = (updated - grad_query) - adjusted
                            grad_query = updated
                        if ACCUMULATE_KV and COMPUTE_DK:
                            # Retained only for the private regression reference.
                            grad_key = grad_logits[:, None] * query[None, :]
                            previous_key = tl.atomic_add(DK + kv_offsets, grad_key, kv_mask)
                            updated_key = previous_key + grad_key
                            virtual_key = updated_key - previous_key
                            error_key = (previous_key - (updated_key - virtual_key)) + (grad_key - virtual_key)
                            tl.atomic_add(DK_CORRECTION + kv_offsets, error_key, kv_mask)
                        if ACCUMULATE_KV and COMPUTE_DV:
                            grad_value = probability[:, None] * grad_output[None, :]
                            previous_value = tl.atomic_add(DV + kv_offsets, grad_value, kv_mask)
                            updated_value = previous_value + grad_value
                            virtual_value = updated_value - previous_value
                            error_value = (previous_value - (updated_value - virtual_value)) + (grad_value - virtual_value)
                            tl.atomic_add(DV_CORRECTION + kv_offsets, error_value, kv_mask)
    if COMPUTE_DQ:
        tl.store(DQ + query_offsets, grad_query, ds < D)
    if WRITE_STATS:
        tl.store(STATS + (token * H_Q + head) * 3, center)
        tl.store(STATS + (token * H_Q + head) * 3 + 1, centered_delta)
        tl.store(STATS + (token * H_Q + head) * 3 + 2, probability_mass)


def _merge_pairs_impl(a_hi, a_lo, b_hi, b_lo):
    """Merge two FP32 expansions with TwoSum; no device-specific intrinsics."""
    summed = a_hi + b_hi
    virtual = summed - a_hi
    error = (a_hi - (summed - virtual)) + (b_hi - virtual)
    low = (a_lo + b_lo) + error
    high = summed + low
    low_virtual = high - summed
    remainder = (summed - (high - low_virtual)) + (low - low_virtual)
    return high, remainder


def _reduce_pairs_impl(high, low, HALF: tl.constexpr, BLOCK_D: tl.constexpr):
    """Reduce adjacent query rows, keeping both FP32 parts of each sum."""
    # Preserve the exact (row 0 + row 1), (row 2 + row 3), ... tree without
    # materializing gather indices at every level. This gives the compiler
    # fixed tensor structure for the D=128 backward specialization that
    # aborts in the reported Ascend compiler. Target validation is still
    # required to confirm this avoids that failure; arithmetic is unchanged.
    high_pairs = tl.permute(tl.reshape(high, (HALF, 2, BLOCK_D)), (0, 2, 1))
    low_pairs = tl.permute(tl.reshape(low, (HALF, 2, BLOCK_D)), (0, 2, 1))
    left_hi, right_hi = tl.split(high_pairs)
    left_lo, right_lo = tl.split(low_pairs)
    return _merge_pairs(left_hi, left_lo, right_hi, right_lo)


def _backward_kv_kernel(
    Q, K, V, CU, CU_BLOCKS, ROW_PTR, QUERIES, NORMALIZERS, STATS, DOUT, DK, DV,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    NSEQ: tl.constexpr, BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_META: tl.constexpr,
    COMPUTE_DK: tl.constexpr, COMPUTE_DV: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    meta_offsets = tl.arange(0, BLOCK_META).to(tl.int64)
    starts = tl.load(CU + meta_offsets, meta_offsets < NSEQ, other=0).to(tl.int64)
    sequence = tl.max(tl.where((meta_offsets < NSEQ) & (starts <= token), meta_offsets, 0), 0)
    seq_start = tl.load(CU + sequence).to(tl.int64)
    block = tl.load(CU_BLOCKS + sequence).to(tl.int64) + (token - seq_start) // BLOCK_SIZE
    row = block * H_KV + group
    row_start = tl.load(ROW_PTR + row).to(tl.int64)
    row_end = tl.load(ROW_PTR + row + 1).to(tl.int64)
    ds = tl.arange(0, BLOCK_D).to(tl.int64)
    qs = tl.arange(0, 32).to(tl.int64)
    kv_offsets = _tnd_offset(token, group, H_KV, D) + ds
    key = tl.load(K + kv_offsets, ds < D, other=0)
    if COMPUTE_DK:
        value = tl.load(V + kv_offsets, ds < D, other=0)
    key_hi = tl.full((BLOCK_D,), 0.0, tl.float32)
    key_lo = tl.full((BLOCK_D,), 0.0, tl.float32)
    value_hi = tl.full((BLOCK_D,), 0.0, tl.float32)
    value_lo = tl.full((BLOCK_D,), 0.0, tl.float32)
    offset = row_start
    while offset < row_end:
        csr_offsets = _advance_offset(offset, qs)
        query_tokens = tl.load(QUERIES + csr_offsets, csr_offsets < row_end, other=0).to(tl.int64)
        valid = (csr_offsets < row_end) & (query_tokens >= token)
        for local_head in range(H_Q // H_KV):
            head = group * (H_Q // H_KV) + local_head
            head_offsets = query_tokens * H_Q + head
            query_offsets = _tnd_offset(query_tokens[:, None], head, H_Q, D) + ds[None, :]
            query_mask = valid[:, None] & (ds[None, :] < D)
            query = tl.load(Q + query_offsets, query_mask, other=0)
            grad_output = tl.load(DOUT + query_offsets, query_mask, other=0)
            maximum = tl.load(NORMALIZERS + head_offsets * 2, valid, other=0)
            denominator = tl.load(NORMALIZERS + head_offsets * 2 + 1, valid, other=1)
            if COMPUTE_DK:
                center = tl.load(STATS + head_offsets * 3, valid, other=0)
                centered_delta = tl.load(STATS + head_offsets * 3 + 1, valid, other=0)
            probability_mass = tl.load(STATS + head_offsets * 3 + 2, valid, other=1)
            logits = tl.sum(query * key[None, :], 1) * SCALE
            probability = tl.exp(tl.where(valid, logits - maximum, float("-inf"))) / denominator / probability_mass
            zeros = tl.full((32, BLOCK_D), 0.0, tl.float32)
            # An explicit pairwise tree also runs efficiently in the CPU
            # interpreter, unlike a generic tuple-reduction callback.
            if COMPUTE_DK:
                grad_probability = tl.sum(grad_output * value[None, :], 1)
                grad_logits = probability * ((grad_probability - center) - centered_delta) * SCALE
                grad_key = grad_logits[:, None] * query
                kh, kl = _reduce_pairs(grad_key, zeros, 16, BLOCK_D)
                kh, kl = _reduce_pairs(kh, kl, 8, BLOCK_D)
                kh, kl = _reduce_pairs(kh, kl, 4, BLOCK_D)
                kh, kl = _reduce_pairs(kh, kl, 2, BLOCK_D)
                kh, kl = _reduce_pairs(kh, kl, 1, BLOCK_D)
                key_hi, key_lo = _merge_pairs(key_hi, key_lo, tl.sum(kh, 0), tl.sum(kl, 0))
            if COMPUTE_DV:
                grad_value = probability[:, None] * grad_output
                vh, vl = _reduce_pairs(grad_value, zeros, 16, BLOCK_D)
                vh, vl = _reduce_pairs(vh, vl, 8, BLOCK_D)
                vh, vl = _reduce_pairs(vh, vl, 4, BLOCK_D)
                vh, vl = _reduce_pairs(vh, vl, 2, BLOCK_D)
                vh, vl = _reduce_pairs(vh, vl, 1, BLOCK_D)
                value_hi, value_lo = _merge_pairs(value_hi, value_lo, tl.sum(vh, 0), tl.sum(vl, 0))
        offset += 32
    if COMPUTE_DK:
        tl.store(DK + kv_offsets, key_hi + key_lo, ds < D)
    if COMPUTE_DV:
        tl.store(DV + kv_offsets, value_hi + value_lo, ds < D)


def _finish_gradients_kernel(
    DK, DV, DK_CORRECTION, DV_CORRECTION, SIZE: tl.constexpr, TILE: tl.constexpr,
    COMPUTE_DK: tl.constexpr, COMPUTE_DV: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE).to(tl.int64)
    mask = offsets < SIZE
    if COMPUTE_DK:
        dk = tl.load(DK + offsets, mask, other=0)
        dk_error = tl.load(DK_CORRECTION + offsets, mask, other=0)
        tl.store(DK + offsets, dk + dk_error, mask)
    if COMPUTE_DV:
        dv = tl.load(DV + offsets, mask, other=0)
        dv_error = tl.load(DV_CORRECTION + offsets, mask, other=0)
        tl.store(DV + offsets, dv + dv_error, mask)


@functools.lru_cache(maxsize=1)
def _kernels():
    global tl, _merge_pairs, _reduce_pairs, _tnd_offset, _advance_offset
    try:
        import triton
        import triton.language as language
    except ImportError as exc:
        raise ImportError("Install triton-ascend on NPU, or Triton for CPU interpreter tests.") from exc
    from ._addressing import tnd_offset, advance_offset

    tl = language
    _tnd_offset, _advance_offset = tnd_offset, advance_offset
    _merge_pairs = triton.jit(_merge_pairs_impl)
    _reduce_pairs = triton.jit(_reduce_pairs_impl)
    return (
        triton, triton.jit(_forward_kernel), triton.jit(_backward_kernel),
        triton.jit(_backward_kv_kernel), triton.jit(_finish_gradients_kernel),
    )


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, cu_seqlens, block_size, scale):
        triton, forward_kernel, _, _, _ = _kernels()
        q32, k32, v32 = (x.to(torch.float32).contiguous() for x in (q, k, v))
        indices = indices.contiguous()
        cu_seqlens = cu_seqlens.contiguous()
        output32 = torch.empty_like(q32)
        lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        normalizers = torch.empty((*q.shape[:2], 2), device=q.device, dtype=torch.float32)
        launch = dict(
            H_Q=q.shape[1], H_KV=k.shape[1], D=q.shape[2], TOPK=indices.shape[2],
            NSEQ=cu_seqlens.numel() - 1, BLOCK_SIZE=block_size, SCALE=scale,
            BLOCK_D=triton.next_power_of_2(q.shape[2]),
            BLOCK_N=min(32, triton.next_power_of_2(block_size)),
            BLOCK_META=triton.next_power_of_2(cu_seqlens.numel() - 1),
        )
        if q.shape[0]:
            forward_kernel[(q.shape[0], q.shape[1])](
                q32, k32, v32, indices, cu_seqlens, output32, lse, normalizers, **launch,
            )
        ctx.save_for_backward(q32, k32, v32, indices, cu_seqlens, output32, normalizers)
        ctx.launch = launch
        ctx.input_dtype = q.dtype
        ctx.mark_non_differentiable(lse)
        return output32.to(q.dtype), lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        from .k2q import build_k2q_csr

        _, _, backward_kernel, kv_kernel, _ = _kernels()
        q, k, v, indices, cu_seqlens, output, normalizers = ctx.saved_tensors
        grad_output = grad_output.to(torch.float32).contiguous()
        need_q, need_k, need_v = ctx.needs_input_grad[:3]
        need_kv = need_k or need_v
        dq = torch.empty_like(q) if need_q else None
        dk = torch.empty_like(k) if need_k else None
        dv = torch.empty_like(v) if need_v else None
        # Unused pointers are never read/written by the constexpr-disabled
        # branches; reuse existing storage instead of allocating dummy outputs.
        dq_ptr, dk_ptr, dv_ptr = dq if need_q else q, dk if need_k else k, dv if need_v else v
        if q.shape[0]:
            # The query prepass is also required without dQ: it supplies the
            # compensated mass/center used by dK/dV. A Q-only backward needs
            # neither this stats buffer nor an adjacency transpose.
            stats = torch.empty((*q.shape[:2], 3), device=q.device, dtype=torch.float32) if need_kv else q
            backward_kernel[(q.shape[0], q.shape[1])](
                q, k, v, indices, cu_seqlens, output, normalizers, grad_output, dq_ptr, dk_ptr, dv_ptr,
                dk_ptr, dv_ptr, stats, ACCUMULATE_KV=False, COMPUTE_DQ=need_q,
                COMPUTE_DK=need_k, COMPUTE_DV=need_v, WRITE_STATS=need_kv, **ctx.launch,
            )
            if need_kv:
                csr = build_k2q_csr(indices, cu_seqlens, block_size=ctx.launch["BLOCK_SIZE"])
                kv_constants = {name: value for name, value in ctx.launch.items() if name not in ("TOPK", "BLOCK_N")}
                kv_kernel[(k.shape[0], k.shape[1])](
                    q, k, v, cu_seqlens, csr.cu_block_lens, csr.row_ptr, csr.query_indices,
                    normalizers, stats, grad_output, dk_ptr, dv_ptr,
                    COMPUTE_DK=need_k, COMPUTE_DV=need_v, **kv_constants,
                )
        grads = tuple(x.to(ctx.input_dtype) if x is not None else None for x in (dq, dk, dv))
        return *grads, None, None, None, None


class _SparseAttentionAtomicReference(_SparseAttention):
    """Private regression reference for the previous query-owned backward."""

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        triton, _, backward_kernel, _, finish_kernel = _kernels()
        q, k, v, indices, cu_seqlens, output, normalizers = ctx.saved_tensors
        grad_output = grad_output.to(torch.float32).contiguous()
        need_q, need_k, need_v = ctx.needs_input_grad[:3]
        dq = torch.empty_like(q) if need_q else None
        dk = torch.zeros_like(k) if need_k else None
        dv = torch.zeros_like(v) if need_v else None
        dk_correction = torch.zeros_like(k) if need_k else k
        dv_correction = torch.zeros_like(v) if need_v else v
        dq_ptr, dk_ptr, dv_ptr = dq if need_q else q, dk if need_k else k, dv if need_v else v
        if q.shape[0]:
            backward_kernel[(q.shape[0], q.shape[1])](
                q, k, v, indices, cu_seqlens, output, normalizers, grad_output, dq_ptr, dk_ptr, dv_ptr,
                dk_correction, dv_correction, q, ACCUMULATE_KV=need_k or need_v,
                COMPUTE_DQ=need_q, COMPUTE_DK=need_k, COMPUTE_DV=need_v,
                WRITE_STATS=False, **ctx.launch,
            )
            if need_k or need_v:
                finish_kernel[(triton.cdiv(k.numel(), 256),)](
                    dk_ptr, dv_ptr, dk_correction, dv_correction, SIZE=k.numel(), TILE=256,
                    COMPUTE_DK=need_k, COMPUTE_DV=need_v,
                )
        grads = tuple(x.to(ctx.input_dtype) if x is not None else None for x in (dq, dk, dv))
        return *grads, None, None, None, None


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
):
    """Apply causal block sparse attention to packed TND tensors.

    ``indices[T, G, K]`` contains distinct sequence-local block IDs, with ``-1``
    for padding.  Each contiguous group of ``Hq / G`` query heads shares a
    selection.  Empty selections produce zero output and zero input gradients.
    Optional natural-log LSE is FP32 and explicitly non-differentiable.

    CPU execution requires ``TRITON_INTERPRET=1`` before the first Triton
    import.  It executes these kernels in the interpreter; there is no eager
    attention fallback.  NPU execution requires a matching triton-ascend build.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Expected q[T,Hq,D] and matching k/v[T,G,D].")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q/k/v must have matching token and head dimensions.")
    if k.shape[1] <= 0 or q.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be a positive multiple of G.")
    if q.shape[2] <= 0:
        raise ValueError("Head dimension must be positive.")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q/k/v must share a BF16, FP16, or FP32 dtype.")
    if indices.ndim != 3 or indices.shape[:2] != (q.shape[0], k.shape[1]) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [T,G,K].")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32 [num_sequences + 1].")
    if any(x.device != q.device for x in (k, v, indices, cu_seqlens)):
        raise ValueError("All inputs must reside on the same device.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer.")
    # Correctness-first metadata validation also applies on accelerators. This
    # introduces one host synchronization; production metadata caching belongs
    # in the future integration layer, outside this standalone implementation.
    lengths = sequence_lengths(cu_seqlens, q.shape[0])
    if max(lengths) > max_seqlen:
        raise ValueError("max_seqlen is smaller than an actual sequence.")
    if q.device.type == "cpu":
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise RuntimeError("CPU Triton execution requires TRITON_INTERPRET=1 before importing Triton.")
    effective_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(effective_scale):
        raise ValueError("scale must be finite.")
    output, lse = _SparseAttention.apply(q, k, v, indices, cu_seqlens, block_size, effective_scale)
    return (output, lse) if return_lse else output
