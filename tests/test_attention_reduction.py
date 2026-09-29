"""Validate streaming single FP32 contributions against an independent sum."""

from __future__ import annotations

import importlib
import math
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
_merge_pairs = None


def _stream_kernel(INPUT, OUT_HIGH, OUT_LOW, BLOCK_D: tl.constexpr):
    dims = tl.arange(0, BLOCK_D).to(tl.int64)
    high = tl.full((BLOCK_D,), 0, tl.float32)
    low = tl.full((BLOCK_D,), 0, tl.float32)
    for row in range(32):
        offsets = tl.cast(row, tl.int64) * BLOCK_D + dims
        addition = tl.load(INPUT + offsets)
        high, low = _merge_pairs(high, low, addition, 0.0)
    tl.store(OUT_HIGH + dims, high)
    tl.store(OUT_LOW + dims, low)


@pytest.mark.parametrize("dimension", [1, 8, 32, 128])
def test_streaming_pairs_preserve_cancellation(dimension):
    global tl, _merge_pairs
    triton = pytest.importorskip("triton")
    if DEVICE.startswith("npu"):
        pytest.importorskip("torch_npu")
    module = importlib.import_module("msa_triton.triton.sparse_attention")
    module._kernels()
    tl, _merge_pairs = module.tl, module._merge_pairs

    rng = np.random.default_rng(512)
    high = rng.standard_normal((32, dimension)).astype(np.float32)
    # Production feeds individual FP32 gradient contributions (b_lo=0);
    # the accumulator's low component is produced by the merge itself.
    # An exact first lane plus independently summed random lanes exercise
    # cancellation without mirroring the production TwoSum algorithm.
    high[:, 0] = 0
    high[:4, 0] = [2.0**24, 1.0, -(2.0**24), 2.0**-24]
    expected = torch.tensor([
        math.fsum(float(v) for v in high[:, d])
        for d in range(dimension)
    ], dtype=torch.float64)
    assert expected[0].item() == 1.0 + 2.0**-24
    values = torch.from_numpy(high).to(DEVICE)
    outputs = [torch.empty(dimension, dtype=torch.float32, device=DEVICE) for _ in range(2)]
    triton.jit(_stream_kernel)[(1,)](
        values, *outputs, BLOCK_D=dimension, enable_fp_fusion=False,
    )
    actual_hi, actual_lo = (x.cpu().double() for x in outputs)
    assert actual_lo[0].item() != 0
    torch.testing.assert_close(actual_hi + actual_lo, expected, atol=1e-12, rtol=1e-12)
