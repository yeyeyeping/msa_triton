"""TND operators for Triton / Triton-Ascend, loaded only when called."""


def m3_index_score(index_q, index_k, cu_seqlens, max_seqlen, *, block_size=128, scale=1.0):
    from .index_score import m3_index_score as implementation

    return implementation(index_q, index_k, cu_seqlens, max_seqlen, block_size=block_size, scale=scale)


def m3_topk(scores, cu_seqlens, *, block_size=128, topk_blocks=16, local_blocks=1):
    from .topk import m3_topk as implementation

    return implementation(
        scores, cu_seqlens, block_size=block_size, topk_blocks=topk_blocks, local_blocks=local_blocks
    )


def m3_sparse_attention(
    q, k, v, indices, cu_seqlens, max_seqlen, *, block_size=128, scale=None, return_lse=False
):
    from .sparse_attention import m3_sparse_attention as implementation

    return implementation(
        q, k, v, indices, cu_seqlens, max_seqlen,
        block_size=block_size, scale=scale, return_lse=return_lse,
    )


__all__ = ["m3_index_score", "m3_topk", "m3_sparse_attention"]
