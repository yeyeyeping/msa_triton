"""Protect row pairing and low components when changing KV tree lowering."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch


DEVICE = os.environ.get("MSA_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU kernel tests require TRITON_INTERPRET=1.",
)
tl = None
_reduce_pairs = None


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


@pytest.mark.parametrize("dimension", [1, 8, 32, 128])
def test_adjacent_pair_tree_preserves_low_components(dimension):
    global tl, _reduce_pairs
    triton = pytest.importorskip("triton")
    if DEVICE.startswith("npu"):
        pytest.importorskip("torch_npu")
    module = importlib.import_module("msa_triton.triton.sparse_attention")
    module._kernels()
    tl, _reduce_pairs = module.tl, module._reduce_pairs

    rng = np.random.default_rng(512)
    high = rng.standard_normal((32, dimension)).astype(np.float32)
    low = (rng.standard_normal(high.shape) * 2.0**-26).astype(np.float32)
    # Cancellation produces a nonzero low component that an FP32 sum drops.
    high[:4, 0] = [2.0**24, 1.0, -(2.0**24), 2.0**-24]
    low[:4, 0] = [2.0**-26, 0.0, 0.0, 0.0]
    expected = _reference_tree(high, low)
    assert np.count_nonzero(expected[1]) > 0
    inputs = [torch.from_numpy(x).to(DEVICE) for x in (high, low)]
    outputs = [torch.empty(dimension, dtype=torch.float32, device=DEVICE) for _ in range(2)]
    triton.jit(_tree_kernel)[(1,)](*inputs, *outputs, BLOCK_D=dimension)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(reference), atol=0, rtol=0)
