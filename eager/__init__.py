"""Transformers-derived BSND eager math behind the common TND interfaces."""

from .ops import m3_index_score, m3_sparse_attention, m3_topk

__all__ = ["m3_index_score", "m3_topk", "m3_sparse_attention"]
