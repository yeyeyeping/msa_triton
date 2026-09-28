"""Private block-to-query CSR metadata for KV-owned sparse backward work.

This is a native PyTorch implementation, independent of Triton imports. It
materializes O(T * G * K + number_of_block_rows) metadata and never constructs
a token-by-token mask. Sequence-boundary validation currently reads metadata
on the host; sorting, counting, and edge construction execute on the input
device. The public three-operator API is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class K2QCSR:
    """Reverse sparse edges, with every tensor on the indices device.

    ``row_ptr`` has ``total_blocks * G + 1`` int32 elements. Row
    ``global_block * G + group`` contains global packed query token IDs in
    ``query_indices`` and their original selection slots in ``slot_indices``.
    Both edge arrays have capacity ``T * G * K``; entries beyond
    ``row_ptr[-1]`` are ``-1``. ``cu_block_lens`` is the int32 prefix sum of
    each sequence's block count. Within each row, edges are ordered by query
    token and then slot. Duplicate selected blocks retain separate slot edges.
    """

    row_ptr: torch.Tensor
    query_indices: torch.Tensor
    slot_indices: torch.Tensor
    cu_block_lens: torch.Tensor


@torch.no_grad()
def build_k2q_csr(
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    block_size: int,
) -> K2QCSR:
    """Invert ``indices[T,G,K]`` into deterministic block/group CSR rows.

    Blocks use sequence-local numbering. Negative, future, and out-of-sequence
    block IDs do not produce edges. A query selecting its current block does
    produce an edge; a consuming attention kernel must still apply the token
    causal mask within that block. Empty sequences, T=0, and K=0 are supported.
    All outputs are int32 and non-differentiable.
    """
    if indices.ndim != 3 or indices.dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [T,G,K]")
    total, groups, slots = indices.shape
    if groups <= 0:
        raise ValueError("indices must have at least one group")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a rank-1 int32 tensor with at least two entries")
    if cu_seqlens.device != indices.device:
        raise ValueError("cu_seqlens and indices must be on the same device")
    capacity = total * groups * slots
    # This limit protects int32 CSR cumulative counts as well as the sort key.
    # Changing the sorting algorithm must not remove the metadata bound.
    if capacity > _INT32_MAX:
        raise OverflowError("T * G * K exceeds int32 edge capacity")

    bounds = cu_seqlens.detach().cpu().tolist()
    lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:])]
    if bounds[0] != 0 or bounds[-1] != total or any(length < 0 for length in lengths):
        raise ValueError("cu_seqlens must span T with nondecreasing boundaries starting at zero")
    block_counts = [(length + block_size - 1) // block_size for length in lengths]
    block_bounds = [0]
    for count in block_counts:
        block_bounds.append(block_bounds[-1] + count)
    num_rows = block_bounds[-1] * groups
    if num_rows >= _INT32_MAX:
        raise OverflowError("block/group row pointer length exceeds int32 indexing capacity")

    device = indices.device
    cu_block_lens = torch.tensor(block_bounds, dtype=torch.int32, device=device)
    row_ptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    if capacity == 0:
        return K2QCSR(
            row_ptr=row_ptr,
            query_indices=torch.empty(0, dtype=torch.int32, device=device),
            slot_indices=torch.empty(0, dtype=torch.int32, device=device),
            cu_block_lens=cu_block_lens,
        )

    repeats = torch.tensor(lengths, dtype=torch.int64, device=device)
    query = torch.arange(total, dtype=torch.int64, device=device)
    sequence_starts = torch.repeat_interleave(cu_seqlens[:-1].long(), repeats, output_size=total)
    block_starts = torch.repeat_interleave(cu_block_lens[:-1].long(), repeats, output_size=total)
    sequence_blocks = torch.repeat_interleave(
        torch.tensor(block_counts, dtype=torch.int64, device=device), repeats, output_size=total
    )
    current_blocks = (query - sequence_starts) // block_size
    selected = indices.long()
    valid = (
        (selected >= 0)
        & (selected < sequence_blocks[:, None, None])
        & (selected <= current_blocks[:, None, None])
    ).reshape(-1)
    group = torch.arange(groups, dtype=torch.int64, device=device)
    rows = ((block_starts[:, None, None] + selected) * groups + group[None, :, None]).reshape(-1)
    safe_rows = rows.masked_fill(~valid, 0)

    # Avoid bincount, whose backend support differs across NPU software stacks.
    counts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    counts.scatter_add_(0, safe_rows, valid.to(torch.int32))
    row_ptr[1:] = counts.cumsum(dim=0, dtype=torch.int32)

    # The flat edge ID orders (query, group, slot). Group is fixed within a
    # CSR row, so this gives the requested (query, slot) ordering. Valid keys
    # are unique; even an unstable device sort therefore has a stable result.
    # rows <= T*G <= capacity and capacity <= INT32_MAX bound keys below 2**63.
    edge = torch.arange(capacity, dtype=torch.int64, device=device)
    keys = safe_rows * capacity + edge
    keys = keys.masked_fill(~valid, torch.iinfo(torch.int64).max)
    order = torch.argsort(keys)
    ordered_valid = valid[order]
    query_indices = (order // (groups * slots)).to(torch.int32).masked_fill(~ordered_valid, -1)
    slot_indices = (order % slots).to(torch.int32).masked_fill(~ordered_valid, -1)
    return K2QCSR(row_ptr, query_indices, slot_indices, cu_block_lens)
