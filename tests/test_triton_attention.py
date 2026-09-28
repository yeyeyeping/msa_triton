"""Strict sparse-attention tests for the interpreter and real accelerators.

CPU: TRITON_INTERPRET=1 python -m pytest msa_triton/tests/test_triton_attention.py
NPU: MSA_TEST_DEVICE=npu python -m pytest msa_triton/tests/test_triton_attention.py
"""

from __future__ import annotations

import os
import importlib

import numpy as np
import pytest
import torch

from msa_triton.tests.reference_fp64 import sparse_attention_fp64
from msa_triton.triton.sparse_attention import _SparseAttentionAtomicReference, m3_sparse_attention


DEVICE = os.environ.get("MSA_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU kernel tests require TRITON_INTERPRET=1; eager/oracle tests remain available.",
)


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("triton")
    if DEVICE.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(DEVICE)


def _inputs(lengths, groups, ratio, dimension, block_size, topk, device):
    generator = torch.Generator().manual_seed(1234)
    total = sum(lengths)
    q = torch.randn(total, groups * ratio, dimension, generator=generator).to(torch.bfloat16)
    k = torch.randn(total, groups, dimension, generator=generator).to(torch.bfloat16)
    v = torch.randn(total, groups, dimension, generator=generator).to(torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    indices = torch.full((total, groups, topk), -1, dtype=torch.int32)
    start = 0
    for length in lengths:
        for offset in range(length):
            for group in range(groups):
                # Distinct selections per group, including causal partial blocks.
                current_block = offset // block_size
                candidates = [current_block]
                candidates += [b for b in range(current_block) if (b + group) % 2 == 0]
                chosen = candidates[:topk]
                if chosen:
                    indices[start + offset, group, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        start += length
    return tuple(x.to(device).detach().requires_grad_() for x in (q, k, v)), indices.to(device), cu.to(device)


def _assert_lse(actual, expected):
    actual = actual.detach().cpu()
    expected = expected.detach().float().cpu()
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(expected), atol=0, rtol=0)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "lengths,groups,ratio,dimension,block_size,topk,scale",
    [
        ([1], 1, 1, 5, 128, 1, None),
        ([7, 3], 2, 2, 8, 3, 3, None),
        ([17, 5], 3, 1, 16, 8, 2, 0.2),
        ([129, 1], 1, 2, 8, 128, 3, None),
        ([127, 128, 257], 1, 1, 8, 128, 3, None),
        ([5, 1, 9], 2, 4, 32, 4, 4, None),
        ([5, 3], 2, 2, 128, 4, 2, None),
        ([0, 3, 0, 2, 0], 2, 2, 8, 2, 2, None),
    ],
)
def test_forward_backward_against_fp64(lengths, groups, ratio, dimension, block_size, topk, scale, device):
    tensors, indices, cu = _inputs(lengths, groups, ratio, dimension, block_size, topk, device)
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, expected_lse = sparse_attention_fp64(
        *reference_inputs, indices.cpu(), cu.cpu(), max(lengths), block_size=block_size, scale=scale,
    )
    actual, actual_lse = m3_sparse_attention(
        *tensors, indices, cu, max(lengths), block_size=block_size, scale=scale, return_lse=True,
    )
    assert actual.dtype == torch.bfloat16
    assert actual_lse.dtype == torch.float32
    assert not actual_lse.requires_grad
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    _assert_lse(actual_lse, expected_lse)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(5678)).bfloat16()
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference in zip(tensors, reference_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("topk", [0, 3])
def test_empty_selection_has_zero_output_and_gradients(topk, device):
    tensors, indices, cu = _inputs([3, 2], 2, 2, 7, 2, topk, device)
    indices.fill_(-1)
    output, lse = m3_sparse_attention(*tensors, indices, cu, 3, block_size=2, return_lse=True)
    assert torch.count_nonzero(output) == 0
    assert torch.isneginf(lse).all()
    output.sum().backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


def test_noncontiguous_inputs_and_masked_queries(device):
    tensors, indices, cu = _inputs([5, 2], 2, 2, 5, 3, 2, device)
    tensors = tuple(x.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_() for x in tensors)
    indices[1] = -1
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, expected_lse = sparse_attention_fp64(
        *reference_inputs, indices.cpu(), cu.cpu(), 5, block_size=3,
    )
    actual, actual_lse = m3_sparse_attention(*tensors, indices, cu, 5, block_size=3, return_lse=True)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    _assert_lse(actual_lse, expected_lse)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(73)).bfloat16()
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference in zip(tensors, reference_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(tensors[0].grad[1]) == 0


def test_default_return_is_output_tensor(device):
    tensors, indices, cu = _inputs([2], 1, 1, 4, 2, 1, device)
    output = m3_sparse_attention(*tensors, indices, cu, 2, block_size=2)
    assert isinstance(output, torch.Tensor)
    assert output.shape == tensors[0].shape


def test_future_only_selected_blocks_are_safely_masked(device):
    tensors, indices, cu = _inputs([5, 2], 1, 2, 8, 3, 1, device)
    indices.fill_(1)
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, expected_lse = sparse_attention_fp64(
        *reference_inputs, indices.cpu(), cu.cpu(), 5, block_size=3,
    )
    actual, actual_lse = m3_sparse_attention(*tensors, indices, cu, 5, block_size=3, return_lse=True)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    _assert_lse(actual_lse, expected_lse)
    actual.sum().backward()
    expected.sum().backward()
    for tensor, reference in zip(tensors, reference_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)


def test_empty_packed_input(device):
    tensors, indices, cu = _inputs([0, 0], 1, 2, 8, 4, 1, device)
    output, lse = m3_sparse_attention(*tensors, indices, cu, 4, block_size=4, return_lse=True)
    assert output.shape == (0, 2, 8)
    assert lse.shape == (0, 2)
    output.sum().backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_scale(scale, device):
    tensors, indices, cu = _inputs([2], 1, 1, 4, 2, 1, device)
    with pytest.raises(ValueError, match="scale must be finite"):
        m3_sparse_attention(*tensors, indices, cu, 2, block_size=2, scale=scale)


def test_rejects_invalid_sequence_metadata(device):
    tensors, indices, cu = _inputs([2], 1, 1, 4, 2, 1, device)
    with pytest.raises(ValueError, match="positive integer"):
        m3_sparse_attention(*tensors, indices, cu, 0, block_size=2)
    with pytest.raises(ValueError, match="smaller"):
        m3_sparse_attention(*tensors, indices, cu, 1, block_size=2)
    cu[-1] = 1
    with pytest.raises(ValueError, match="end at total_tokens"):
        m3_sparse_attention(*tensors, indices, cu, 2, block_size=2)


def test_long_key_accumulation_bf16_rounding_regression(device):
    """A gradient close to a BF16 rounding boundary exposed FP32 atomic drift."""
    rng = np.random.default_rng(1234)
    # Preserve the exact inputs that exposed the error in a packed [127,128,257]
    # stress case; only the last, independent sequence is needed to reproduce it.
    tensors = tuple(
        torch.from_numpy(rng.standard_normal((512, 1, 8))).float().bfloat16()[255:]
        .to(device).detach().requires_grad_()
        for _ in range(3)
    )
    grad = torch.from_numpy(rng.standard_normal((512, 1, 8))).float().bfloat16()[255:]
    indices = torch.full((257, 1, 3), -1, dtype=torch.int32)
    for token in range(257):
        current = token // 128
        chosen = [current] + [b for b in range(current) if b % 2 == 0]
        indices[token, 0, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    cu = torch.tensor([0, 257], dtype=torch.int32, device=device)
    indices = indices.to(device)
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, _ = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), 257)
    actual = m3_sparse_attention(*tensors, indices, cu, 257)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference in zip(tensors, reference_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)


def test_kv_owner_matches_query_owner_and_fp64_with_hotspots(device):
    lengths, groups, ratio, dim, block_size = [41, 5], 2, 4, 16, 8
    tensors, indices, cu = _inputs(lengths, groups, ratio, dim, block_size, 2, device)
    selected = torch.full(indices.shape, -1, dtype=torch.int32)
    start = 0
    for length in lengths:
        for position in range(length):
            # Group 0 creates a highly shared first block; group 1 selects
            # only its current block. Many other group-0 CSR rows stay empty.
            selected[start + position, 0, 0] = 0
            selected[start + position, 1, 0] = position // block_size
        start += length
    indices = selected.to(device)
    previous_inputs = tuple(x.detach().clone().requires_grad_() for x in tensors)
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, _ = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), max(lengths), block_size=block_size)
    actual = m3_sparse_attention(*tensors, indices, cu, max(lengths), block_size=block_size)
    previous, _ = _SparseAttentionAtomicReference.apply(
        *previous_inputs, indices, cu, block_size, dim**-0.5,
    )
    torch.testing.assert_close(actual, previous, atol=0, rtol=0)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(819)).bfloat16()
    actual.backward(grad.to(device))
    previous.backward(grad.to(device))
    expected.backward(grad.double())
    for current, prior, reference in zip(tensors, previous_inputs, reference_inputs):
        torch.testing.assert_close(current.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(current.grad, prior.grad, atol=1e-4, rtol=1e-4)


def test_selected_block_order_preserves_output_and_gradients(device):
    tensors, indices, cu = _inputs([9, 3], 2, 2, 7, 3, 3, device)
    reordered = torch.full(indices.shape, -1, dtype=torch.int32)
    host_indices = indices.cpu()
    for token in range(indices.shape[0]):
        for group in range(indices.shape[1]):
            valid = host_indices[token, group]
            valid = valid[valid >= 0].flip(0)
            reordered[token, group, :valid.numel()] = valid
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, _ = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), 9, block_size=3)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(127)).bfloat16()
    expected.backward(grad.double())
    for selected in (indices, reordered.to(device)):
        current = tuple(x.detach().clone().requires_grad_() for x in tensors)
        actual = m3_sparse_attention(*current, selected, cu, 9, block_size=3)
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
        actual.backward(grad.to(device))
        for tensor, reference in zip(current, reference_inputs):
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("needs_grad", [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (True, False, True), (False, True, True),
    (True, True, True),
])
def test_independent_input_gradients_and_q_only_skips_csr(needs_grad, device, monkeypatch):
    """KV-only still needs query statistics; Q-only must not build k2q."""
    tensors, indices, cu = _inputs([0, 5, 0, 3, 0], 2, 2, 8, 3, 2, device)
    tensors = tuple(x.detach().requires_grad_(need) for x, need in zip(tensors, needs_grad))
    saved_inputs = tuple(x.detach().clone() for x in tensors)
    indices[2] = -1
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_(need) for x, need in zip(tensors, needs_grad))
    k2q_module = importlib.import_module("msa_triton.triton.k2q")
    original_build = k2q_module.build_k2q_csr
    csr_calls = []

    def checked_build(*args, **kwargs):
        assert needs_grad[1] or needs_grad[2], "Q-only backward must not build CSR"
        csr_calls.append(True)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(k2q_module, "build_k2q_csr", checked_build)
    expected, _ = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), 5, block_size=3)
    actual = m3_sparse_attention(*tensors, indices, cu, 5, block_size=3)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(190)).bfloat16()
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    assert len(csr_calls) == int(needs_grad[1] or needs_grad[2])
    for tensor, saved in zip(tensors, saved_inputs):
        # Disabled output pointers reuse input storage and must never write it.
        torch.testing.assert_close(tensor.detach(), saved, atol=0, rtol=0)
    for tensor, reference, need in zip(tensors, reference_inputs, needs_grad):
        if need:
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)
        else:
            assert tensor.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dtype_and_gqa16_forward_backward_against_fp64(dtype, device):
    """Exercise non-BF16 public casts and a large contiguous GQA group."""
    tensors, indices, cu = _inputs([5, 3], 1, 16, 8, 3, 2, device)
    generator = torch.Generator().manual_seed(320)
    tensors = tuple(
        torch.randn(x.shape, generator=generator).to(dtype=dtype, device=device).requires_grad_()
        for x in tensors
    )
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in tensors)
    expected, expected_lse = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), 5, block_size=3)
    actual, actual_lse = m3_sparse_attention(*tensors, indices, cu, 5, block_size=3, return_lse=True)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().to(dtype), atol=1e-4, rtol=1e-4)
    _assert_lse(actual_lse, expected_lse)
    grad = torch.randn(actual.shape, generator=generator).to(dtype)
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference in zip(tensors, reference_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("needs_grad", [(False, True, False), (False, False, True)])
def test_private_atomic_reference_independent_kv_gradients(needs_grad, device):
    tensors, indices, cu = _inputs([3, 2], 1, 2, 8, 2, 2, device)
    tensors = tuple(x.detach().requires_grad_(need) for x, need in zip(tensors, needs_grad))
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_(need) for x, need in zip(tensors, needs_grad))
    expected, _ = sparse_attention_fp64(*reference_inputs, indices.cpu(), cu.cpu(), 3, block_size=2)
    actual, _ = _SparseAttentionAtomicReference.apply(*tensors, indices, cu, 2, 8**-0.5)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(801)).bfloat16()
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference, need in zip(tensors, reference_inputs, needs_grad):
        if need:
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)
        else:
            assert tensor.grad is None
