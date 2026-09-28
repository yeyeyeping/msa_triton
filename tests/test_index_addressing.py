"""Exercise production offset helpers across int32 limits without huge buffers."""

import os

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

from msa_triton.triton._addressing import advance_offset, tnd_offset
from msa_triton.triton.index_score import _sequence_bounds


@triton.jit
def _address_probe(TOKENS, OFFSETS, OUT, HEADS: tl.constexpr, DIM: tl.constexpr):
    lane = tl.program_id(0)
    token = tl.load(TOKENS + lane)
    cursor = tl.load(OFFSETS + lane)
    tl.store(OUT + lane * 3, tnd_offset(token, HEADS - 1, HEADS, DIM) + DIM - 1)
    tl.store(OUT + lane * 3 + 1, advance_offset(cursor, 31))
    tl.store(OUT + lane * 3 + 2, advance_offset(cursor, 32))


@triton.jit
def _bounds_probe(CU, OUT, N_SEQS: tl.constexpr, SEQ_TILE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    begin, end = _sequence_bounds(CU, token, N_SEQS, SEQ_TILE)
    tl.store(OUT + token * 2, begin)
    tl.store(OUT + token * 2 + 1, end)


def _device():
    name = os.environ.get("MSA_TEST_DEVICE", "cpu")
    if name == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU kernel execution requires TRITON_INTERPRET=1")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return name


@pytest.mark.parametrize("heads,dim", [(64, 128), (4, 4096), (4, 16)])
def test_offsets_and_csr_cursor_cross_int32_limit(heads, dim):
    name = _device()
    tokens_cpu = torch.tensor([262143, 262144, 2**31 - 2], dtype=torch.int32)
    cursors_cpu = torch.tensor([0, 2**31 - 32, 2**31 - 1], dtype=torch.int32)
    tokens, cursors = tokens_cpu.to(name), cursors_cpu.to(name)
    actual = torch.empty((3, 3), dtype=torch.int64, device=name)
    _address_probe[(3,)](tokens, cursors, actual, HEADS=heads, DIM=dim)
    expected = torch.tensor([
        [(int(t) * heads + heads - 1) * dim + dim - 1, int(c) + 31, int(c) + 32]
        for t, c in zip(tokens_cpu, cursors_cpu)
    ], dtype=torch.int64)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [
    [0, 3, 0, 2, 0],
    [0, 0, 1, 0, 0],
    [2, 0, 4, 0, 1, 0, 0, 2, 0],
])
def test_sequence_boundaries_with_empty_sequences(lengths):
    device = _device()
    bounds, expected = [0], []
    for length in lengths:
        begin = bounds[-1]
        end = begin + length
        expected.extend([(begin, end)] * length)
        bounds.append(end)
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    actual = torch.empty((bounds[-1], 2), dtype=torch.int64, device=device)
    _bounds_probe[(bounds[-1],)](
        cu, actual, N_SEQS=len(lengths), SEQ_TILE=triton.next_power_of_2(len(lengths)),
    )
    torch.testing.assert_close(actual.cpu(), torch.tensor(expected, dtype=torch.int64), rtol=0, atol=0)
