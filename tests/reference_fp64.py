"""Independent, intentionally slow CPU FP64 MSA oracle.

No production eager, layout, or Triton helpers are imported. All arithmetic
remains double until the caller rounds final outputs/gradients for comparison.
"""

from __future__ import annotations

import torch


def _double_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cpu", dtype=torch.float64)


def index_score_fp64(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int = 128,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute each causal block separately without building a padded batch."""
    q, k = _double_cpu(index_q), _double_cpu(index_k)
    bounds = cu_seqlens.detach().cpu().tolist()
    num_blocks = (max_seqlen + block_size - 1) // block_size
    sequences = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        length = end - start
        columns = []
        for block in range(num_blocks):
            lower, upper = block * block_size, min((block + 1) * block_size, length)
            if lower >= upper:
                columns.append(q[start:end, :, 0] * 0 + float("-inf"))
                continue
            logits = torch.einsum("tgd,kd->tgk", q[start:end], k[start + lower : start + upper, 0]) * scale
            allowed = torch.arange(lower, upper)[None, :] <= torch.arange(length)[:, None]
            columns.append(logits.masked_fill(~allowed[:, None], float("-inf")).amax(-1))
        sequences.append(torch.stack(columns, dim=-1))
    result = torch.cat(sequences)
    # An empty batch evaluates no QK products. Keep K connected with its
    # mathematical zero derivative so backward returns a shaped empty tensor.
    return result + k.sum() * 0 if q.shape[0] == 0 else result


def sparse_attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    *,
    block_size: int = 128,
    scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Per-sequence and per-group dense double oracle with an index-built mask."""
    q, k, v = _double_cpu(q), _double_cpu(k), _double_cpu(v)
    ids = indices.detach().cpu()
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    bounds = cu_seqlens.detach().cpu().tolist()
    ratio = q.shape[1] // k.shape[1]
    outputs, normalizers = [], []
    for start, end in zip(bounds[:-1], bounds[1:]):
        length = end - start
        group_out, group_lse = [], []
        token_pos = torch.arange(length)
        for group in range(k.shape[1]):
            # [S,R,S], where R is the contiguous query heads in this KV group.
            logits = torch.einsum("trd,kd->trk", q[start:end, group * ratio : (group + 1) * ratio], k[start:end, group]) * scale
            selected = ((token_pos[None, :, None] // block_size) == ids[start:end, group, None, :]).any(-1)
            allowed = selected & (token_pos[None, :] <= token_pos[:, None])
            has_keys = allowed.any(-1)
            masked = logits.masked_fill(~allowed[:, None, :], float("-inf"))
            safe = torch.where(has_keys[:, None, None], masked, torch.zeros_like(masked))
            probs = torch.softmax(safe, dim=-1).masked_fill(~has_keys[:, None, None], 0)
            group_out.append(torch.einsum("trk,kd->trd", probs, v[start:end, group]))
            group_lse.append(torch.logsumexp(safe, dim=-1).masked_fill(~has_keys[:, None], float("-inf")))
        outputs.append(torch.cat(group_out, dim=1))
        normalizers.append(torch.cat(group_lse, dim=1))
    output, lse = torch.cat(outputs), torch.cat(normalizers)
    return (output, lse) if return_lse else output
