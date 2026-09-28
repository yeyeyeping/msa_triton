"""Standalone reverse sparse-map tests against a Python edge inversion."""

from __future__ import annotations

from collections import Counter
import os

import pytest
import torch

from msa_triton.triton.k2q import build_k2q_csr


def _device():
    name = os.environ.get("MSA_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(name)


def _invert_python(indices, lengths, block_size):
    """Construct rows using Python lists only; no production helpers."""
    tokens, groups, slots = indices.shape
    values = indices.tolist()
    block_prefix = [0]
    for length in lengths:
        block_prefix.append(block_prefix[-1] + (length + block_size - 1) // block_size)
    rows = [[] for _ in range(block_prefix[-1] * groups)]
    valid_edges = []
    start = 0
    for sequence, length in enumerate(lengths):
        nblocks = (length + block_size - 1) // block_size
        for local_query in range(length):
            query = start + local_query
            for group in range(groups):
                for slot in range(slots):
                    block = values[query][group][slot]
                    if block < 0 or block >= nblocks or block * block_size > local_query:
                        continue
                    row = (block_prefix[sequence] + block) * groups + group
                    rows[row].append((query, slot))
                    valid_edges.append((row, query, slot))
        start += length
    row_ptr, queries, selection_slots = [0], [], []
    for row in rows:
        for query, slot in sorted(row):
            queries.append(query)
            selection_slots.append(slot)
        row_ptr.append(len(queries))
    padding = tokens * groups * slots - len(queries)
    return row_ptr, queries + [-1] * padding, selection_slots + [-1] * padding, block_prefix, valid_edges


def _check(indices_cpu, lengths, block_size):
    device = _device()
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    indices = indices_cpu.to(device)
    original = indices.clone()
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    result = build_k2q_csr(indices, cu, block_size=block_size)
    ptr, queries, slots, block_prefix, edges = _invert_python(indices_cpu, lengths, block_size)
    for actual, expected in (
        (result.row_ptr, ptr), (result.query_indices, queries),
        (result.slot_indices, slots), (result.cu_block_lens, block_prefix),
    ):
        assert actual.dtype == torch.int32 and actual.device == indices.device
        assert not actual.requires_grad
        torch.testing.assert_close(actual.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(indices, original, rtol=0, atol=0)

    # Independently reconstruct the valid edge multiset from CSR segments.
    reconstructed = []
    actual_ptr = result.row_ptr.cpu().tolist()
    actual_queries = result.query_indices.cpu().tolist()
    actual_slots = result.slot_indices.cpu().tolist()
    for row, (low, high) in enumerate(zip(actual_ptr[:-1], actual_ptr[1:])):
        for edge in range(low, high):
            reconstructed.append((row, actual_queries[edge], actual_slots[edge]))
    assert Counter(reconstructed) == Counter(edges)
    count = actual_ptr[-1]
    assert all(value == -1 for value in actual_queries[count:])
    assert all(value == -1 for value in actual_slots[count:])
    return result


@pytest.mark.parametrize(
    "lengths,groups,slots,block_size",
    [([1], 1, 4, 128), ([3, 5], 3, 5, 2), ([0, 3, 0, 2, 0], 2, 4, 2),
     ([127, 128, 129], 4, 5, 128), ([0, 0], 3, 5, 128), ([4, 2], 3, 0, 2)],
)
def test_k2q_python_inversion(lengths, groups, slots, block_size):
    generator = torch.Generator().manual_seed(951)
    max_blocks = (max(lengths) + block_size - 1) // block_size
    ids = torch.randint(-2, max_blocks + 4, (sum(lengths), groups, slots),
                        generator=generator, dtype=torch.int32)
    _check(ids, lengths, block_size)


def test_k2q_preserves_duplicate_slots_and_resets_blocks_per_sequence():
    ids = torch.tensor(
        [[[0, 0, 1], [1, 0, -1]],
         [[0, -1, 8], [0, 0, 1]],
         [[1, 0, 1], [0, 1, -2]],
         [[0, 0, 1], [0, -1, 9]],
         [[0, 0, -1], [0, 0, -1]]], dtype=torch.int32,
    )
    result = _check(ids, [3, 2], block_size=2)
    assert result.cu_block_lens.cpu().tolist() == [0, 2, 3]
    # Sequence 1's local block 0 is global block 2, hence rows 4 and 5.
    ptr = result.row_ptr.cpu().tolist()
    assert set(result.query_indices[ptr[4]:ptr[5]].cpu().tolist()) == {3, 4}


def test_k2q_local_block_hotspots_and_tail_block():
    lengths, groups, slots, block_size = [9, 5], 4, 3, 4
    ids = torch.full((sum(lengths), groups, slots), -1, dtype=torch.int32)
    start = 0
    for length in lengths:
        for local in range(length):
            ids[start + local, :, 0] = local // block_size
            ids[start + local, :, 1] = 0
            if local // block_size > 0:
                ids[start + local, :, 2] = local // block_size - 1
        start += length
    _check(ids, lengths, block_size)


def test_k2q_all_invalid_has_zero_pointers_and_minus_one_tail():
    ids = torch.full((7, 2, 4), 99, dtype=torch.int32)
    ids[:, :, 0] = -1
    result = _check(ids, [3, 4], block_size=2)
    assert result.row_ptr.count_nonzero().item() == 0
    assert (result.query_indices == -1).all().item()


def test_k2q_noncontiguous_input():
    base = torch.tensor([[[0, 1, 2, -1], [0, -1, 9, 0]]] * 5, dtype=torch.int32)
    indices = base.transpose(1, 2)
    assert not indices.is_contiguous()
    _check(indices, [2, 3], block_size=2)


def test_k2q_rejects_invalid_metadata_and_capacity_overflow():
    ids = torch.zeros(3, 2, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="nondecreasing"):
        build_k2q_csr(ids, torch.tensor([0, 2, 1, 3], dtype=torch.int32), block_size=2)
    with pytest.raises(ValueError, match="span T"):
        build_k2q_csr(ids, torch.tensor([0, 2], dtype=torch.int32), block_size=2)
    with pytest.raises(ValueError, match="block_size"):
        build_k2q_csr(ids, torch.tensor([0, 3], dtype=torch.int32), block_size=0)
    # A stride-zero view exercises validation without allocating billions of edges.
    oversized = torch.tensor(0, dtype=torch.int32).expand(1, 2, 2**30)
    with pytest.raises(OverflowError, match="edge capacity"):
        build_k2q_csr(oversized, torch.tensor([0, 1], dtype=torch.int32), block_size=1)
