"""Frozen 6d47514 KV tree for historical compiler diagnostics, never acceptance.

Keep this small snapshot separate from the production kernel. Temporary variants
replace only KV backward; forward and the query prepass use the current module.
Historical launches deliberately retain the backend's default FP fusion setting.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import torch


tl = None
_merge_pairs = None
_reduce_pairs = None
_tnd_offset = None
_advance_offset = None


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


def _tree_kernel(HIGH, LOW, OUT_HIGH, OUT_LOW, BLOCK_D: tl.constexpr):
    rows = tl.arange(0, 32)
    dims = tl.arange(0, BLOCK_D)
    offsets = rows[:, None] * BLOCK_D + dims[None, :]
    high = tl.load(HIGH + offsets)
    low = tl.load(LOW + offsets)
    high, low = _reduce_pairs(high, low, 16, BLOCK_D)
    high, low = _reduce_pairs(high, low, 8, BLOCK_D)
    high, low = _reduce_pairs(high, low, 4, BLOCK_D)
    high, low = _reduce_pairs(high, low, 2, BLOCK_D)
    high, low = _reduce_pairs(high, low, 1, BLOCK_D)
    tl.store(OUT_HIGH + dims, tl.sum(high, 0))
    tl.store(OUT_LOW + dims, tl.sum(low, 0))


def _reference_tree(high, low):
    # NumPy FP32 ufuncs round each operation, independently of Triton lowering.
    # Slices define row pairing directly, without reshape/permute/split.
    while len(high) > 1:
        a_hi, b_hi = high[0::2], high[1::2]
        a_lo, b_lo = low[0::2], low[1::2]
        summed = a_hi + b_hi
        virtual = summed - a_hi
        error = (a_hi - (summed - virtual)) + (b_hi - virtual)
        residual = (a_lo + b_lo) + error
        high = summed + residual
        low_virtual = high - summed
        low = (summed - (high - low_virtual)) + (residual - low_virtual)
    return high[0], low[0]


def _bind(module):
    import triton
    import triton.language as language
    from msa_triton.triton._addressing import tnd_offset, advance_offset

    module.tl = language
    module._tnd_offset, module._advance_offset = tnd_offset, advance_offset
    module._merge_pairs = triton.jit(module._merge_pairs_impl)
    module._reduce_pairs = triton.jit(module._reduce_pairs_impl)
    return triton


class _HistoricalLaunch:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        original = self.kernel[grid]

        def launch(*args, **kwargs):
            # Production now protects compensation with fusion disabled. The
            # historical comparison keeps its original backend default instead.
            options = dict(kwargs)
            options.pop("enable_fp_fusion", None)
            return original(*args, **options)

        return launch


def load_legacy_kv(directory, *, transform=None):
    """Persist a frozen or transformed KV source and return its launch wrapper."""
    source = "from __future__ import annotations\n\n"
    source += "\n\n\n".join(inspect.getsource(function) for function in (
        _merge_pairs_impl, _reduce_pairs_impl, _backward_kv_kernel,
    ))
    if transform is not None:
        source = transform(source)
    path = Path(directory) / "legacy_attention_kv.py"
    path.write_text(source)
    name = "msa_triton.tests._temporary_legacy_attention_kv"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load historical KV source")
    module = importlib.util.module_from_spec(spec)
    # Source stays available to Triton and the evidence collector until the
    # caller's temporary directory is removed; no production module is copied.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    triton = _bind(module)
    print("HISTORICAL_KV=6d47514 enable_fp_fusion=backend_default", flush=True)
    return _HistoricalLaunch(triton.jit(module._backward_kv_kernel))


def check_legacy_tree(device, dimension=128):
    """Exact historical high/low check; intentionally outside test discovery."""
    triton = _bind(sys.modules[__name__])
    rng = np.random.default_rng(512)
    high = rng.standard_normal((32, dimension)).astype(np.float32)
    low = (rng.standard_normal(high.shape) * 2.0**-26).astype(np.float32)
    high[:4, 0] = [2.0**24, 1.0, -(2.0**24), 2.0**-24]
    low[:4, 0] = [2.0**-26, 0.0, 0.0, 0.0]
    expected = _reference_tree(high, low)
    assert np.count_nonzero(expected[1]) > 0
    inputs = [torch.from_numpy(x).to(device) for x in (high, low)]
    outputs = [torch.empty(dimension, dtype=torch.float32, device=device) for _ in range(2)]
    print("HISTORICAL_TREE=6d47514 enable_fp_fusion=backend_default", flush=True)
    triton.jit(_tree_kernel)[(1,)](*inputs, *outputs, BLOCK_D=dimension)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(reference), atol=0, rtol=0)
