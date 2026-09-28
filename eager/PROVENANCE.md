# Transformers eager migration

Source: Hugging Face Transformers, commit
[`f324707307757d9c0b8dac1c4462eceff911fa2f`](https://github.com/huggingface/transformers/blob/f324707307757d9c0b8dac1c4462eceff911fa2f/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py)
(2026-09-23; source-file revision verified through the official GitHub API on 2026-09-28).
License: Apache-2.0; original copyright belongs to the MiniMax AI and HuggingFace teams.

The migrated operations are the numerical portions of `MiniMaxM3VLIndexer.forward`,
`MiniMaxM3VLIndexer.build_block_mask`, `repeat_kv`, and `eager_attention_forward`.
Projection, QK normalization, RoPE, caches, and model configuration remain outside
these operators. Transformers is not a runtime dependency.

## Preserved behavior

- The eager numerical path uses padded BSND tensors internally (transposed to
  BHSD for matrix multiplication); packing does not shift sequence-relative blocks.
- Indexer QK is explicitly FP32, unscaled by default, with causal masking before
  a separate maximum over the tokens of each block. Groups remain independent.
- Local blocks occupy top-k slots. Selection is discrete and group-specific.
- Main attention performs QK in the input dtype, applies the scale in that dtype,
  computes softmax in FP32, casts probabilities back to the input dtype, then
  computes PV in the input dtype. Native autograd preserves those rounding points.
- Each KV group serves a contiguous range of query heads.

## Explicit adapter changes

- Public inputs and outputs are TND; differentiable unpack/pack uses right padding.
- Padding keys and queries are masked. All-invalid query rows produce zero output,
  zero gradients and `-inf` LSE instead of an undefined all-masked softmax.
- Masks use `-inf` in the adapter rather than the upstream additive minimum finite
  value. With at least one valid key and finite logits the allowed probabilities
  are equivalent; all-invalid rows follow the explicit new contract above.
- Score, selection and attention are separate calls. `scale=1.0` is exposed for
  score without changing its default; no backend selector is provided.
- Local boosting uses a new tensor rather than modifying public scores in place.
- Selection always returns exactly K int32 slots, filling missing entries with
  `-1`; upstream bounds its returned slot count by the available block count.
- Attention optionally returns detached FP32 natural-log LSE. It is diagnostic
  teacher state, not a differentiable extra output.
- Self-attention and zero dropout are the initial contract. Empty packed sequences
  are accepted without contributing scores, outputs, or gradients.

## Accuracy interpretation

The native BF16 attention path rounds intermediate QK, scaled QK, and probabilities.
It is therefore a compatibility reference, not a FP64 oracle. The independent
`tests/reference_fp64.py` retains double precision throughout and rounds only at
the comparison boundary. Native-vs-oracle diagnostics report failures of the
unchanged `atol=rtol=1e-4` criterion; they are not used to silently relax optimized
kernel accuracy requirements.
