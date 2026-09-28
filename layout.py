"""Differentiable right-padding adapters for packed TND tensors.

These deliberately use host-visible sequence lengths: they are reference-path
utilities, not helpers to call inside a production Triton kernel launch loop.
"""

from __future__ import annotations

import torch


def sequence_lengths(cu_seqlens: torch.Tensor, total_tokens: int) -> list[int]:
    """Validate packed boundaries and return sequence lengths on the host."""
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a vector with at least two entries")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be int32")
    boundaries = cu_seqlens.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at zero and end at total_tokens")
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if min(lengths) < 0:
        raise ValueError("cu_seqlens must be nondecreasing")
    return lengths


def unpack_tnd(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    pad_value: float | int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right-padded ``[B,S,...]`` data and a ``[B,S]`` valid mask.

    The name is conventional: any number of dimensions after T is supported,
    including score tensors and integer block indices.
    """
    if tensor.ndim < 1:
        raise ValueError("tensor must have a leading token dimension")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    if tensor.device != cu_seqlens.device:
        raise ValueError("tensor and cu_seqlens must be on the same device")
    lengths = sequence_lengths(cu_seqlens, tensor.shape[0])
    if max_seqlen < max(lengths):
        raise ValueError("max_seqlen is smaller than an actual sequence")
    rows = []
    start = 0
    for length in lengths:
        # F.pad rejects zero trailing dimensions, including valid indices[T,G,0].
        # Concatenating padding also retains the input's gradient connection for
        # empty packed sequences and preserves token ordering exactly.
        padding = tensor.new_full((max_seqlen - length, *tensor.shape[1:]), pad_value)
        rows.append(torch.cat([tensor[start : start + length], padding], dim=0))
        start += length
    data = torch.stack(rows)
    valid = torch.arange(max_seqlen, device=tensor.device)[None, :] < torch.tensor(
        lengths, device=tensor.device
    )[:, None]
    return data, valid


def pack_bs(tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Remove right padding from ``[B,S,...]`` without detaching gradients."""
    if tensor.ndim < 2:
        raise ValueError("tensor must have batch and sequence dimensions")
    if tensor.device != cu_seqlens.device:
        raise ValueError("tensor and cu_seqlens must be on the same device")
    total = int(cu_seqlens[-1].item())
    lengths = sequence_lengths(cu_seqlens, total)
    if tensor.shape[0] != len(lengths) or tensor.shape[1] < max(lengths):
        raise ValueError("tensor batch or padded sequence dimension is invalid")
    return torch.cat([tensor[b, :length] for b, length in enumerate(lengths)])
