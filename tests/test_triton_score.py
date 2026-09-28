"""Run with TRITON_INTERPRET=1, or MSA_TEST_DEVICE=npu on target hardware."""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("triton")
from msa_triton.eager import m3_index_score as eager_index_score
from msa_triton.triton.index_score import m3_index_score
from msa_triton.tests.reference_fp64 import index_score_fp64


def _device():
    name = os.environ.get("MSA_TEST_DEVICE", "cpu")
    if name.startswith("cpu") and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU kernel execution requires TRITON_INTERPRET=1 before import")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(name)


def _oracle(q, k, lengths, block_size, max_seqlen, scale):
    """The oracle keeps all intermediates and autograd computations in FP64."""
    assert q.dtype == k.dtype == torch.float64
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    return index_score_fp64(q, k, cu, max_seqlen, block_size=block_size, scale=scale)


def _assert_scores(actual, expected):
    actual = actual.detach().cpu()
    expected = expected.detach().to(torch.float32)
    assert actual.dtype == torch.float32
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert not torch.isnan(actual).any()
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "lengths,groups,dim,block_size,scale",
    [([1], 1, 1, 4, 1.0),
     ([0, 0], 2, 7, 4, 1.0),
     ([0, 3, 0, 2, 0], 2, 7, 4, 1.0),
     ([3, 7], 2, 7, 4, 1.0),
     ([7, 11], 4, 16, 8, 0.5),
     ([9], 2, 64, 4, 1.0),
     ([5], 4, 128, 128, 1.0)],
)
def test_score_forward_backward_fp64(lengths, groups, dim, block_size, scale):
    device = _device()
    generator = torch.Generator().manual_seed(731)
    q_cpu = torch.randn(sum(lengths), groups, dim, generator=generator).to(torch.bfloat16)
    k_cpu = torch.randn(sum(lengths), 1, dim, generator=generator).to(torch.bfloat16)
    q = q_cpu.to(device).clone().requires_grad_()
    k = k_cpu.to(device).clone().requires_grad_()
    qr = q_cpu.double().requires_grad_()
    kr = k_cpu.double().requires_grad_()
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                      dtype=torch.int32, device=device)
    max_seqlen = max(lengths) + block_size  # Includes entirely invalid output blocks.
    actual = m3_index_score(q, k, cu, max_seqlen, block_size=block_size, scale=scale)
    expected = _oracle(qr, kr, lengths, block_size, max_seqlen, scale)
    _assert_scores(actual, expected)
    upstream = torch.randn(actual.shape, generator=generator).to(torch.bfloat16).float()
    actual.backward(upstream.to(device))
    expected.backward(upstream.double())
    torch.testing.assert_close(q.grad.cpu(), qr.grad.to(q.dtype), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad.cpu(), kr.grad.to(k.dtype), atol=1e-4, rtol=1e-4)
    qe, ke = q_cpu.to(device).clone().requires_grad_(), k_cpu.to(device).clone().requires_grad_()
    eager = eager_index_score(qe, ke, cu, max_seqlen, block_size=block_size, scale=scale)
    _assert_scores(actual, eager.cpu())
    eager.backward(upstream.to(device))
    torch.testing.assert_close(q.grad, qe.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad, ke.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("lengths", [[127], [128], [129], [1, 129, 257]])
def test_score_causal_block_and_packing_boundaries(lengths):
    device = _device()
    generator = torch.Generator().manual_seed(97)
    q = torch.randn(sum(lengths), 2, 4, generator=generator).to(torch.bfloat16)
    k = torch.randn(sum(lengths), 1, 4, generator=generator).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                      dtype=torch.int32, device=device)
    actual = m3_index_score(q.to(device), k.to(device), cu, max(lengths), block_size=128)
    expected = _oracle(q.double(), k.double(), lengths, 128, max(lengths), 1.0)
    _assert_scores(actual, expected)


@pytest.mark.parametrize("scale", [1.0, 0.0, -1.0])
def test_score_ties_split_backward_and_invalid_blocks_have_zero_gradient(scale):
    device = _device()
    q_cpu = torch.ones(5, 2, 3, dtype=torch.bfloat16)
    k_cpu = torch.ones(5, 1, 3, dtype=torch.bfloat16)
    q, k = q_cpu.to(device).clone().requires_grad_(), k_cpu.to(device).clone().requires_grad_()
    qr, kr = q_cpu.double().requires_grad_(), k_cpu.double().requires_grad_()
    cu = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    actual = m3_index_score(q, k, cu, 8, block_size=4, scale=scale)
    expected = _oracle(qr, kr, [2, 3], 4, 8, scale)
    _assert_scores(actual, expected)
    # Nonzero gradients in the invalid blocks must contribute exactly zero.
    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(q.grad.cpu(), qr.grad.to(q.dtype), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad.cpu(), kr.grad.to(k.dtype), atol=1e-4, rtol=1e-4)


def test_score_groups_remain_independent():
    device = _device()
    q = torch.tensor([[[1., 0.], [0., 1.]]] * 4, dtype=torch.bfloat16, device=device)
    k = torch.tensor([[[9., 1.]], [[1., 8.]], [[0., 0.]], [[0., 0.]]],
                     dtype=torch.bfloat16, device=device)
    cu = torch.tensor([0, 4], dtype=torch.int32, device=device)
    scores = m3_index_score(q, k, cu, 4, block_size=1)
    assert scores.shape == (4, 2, 4)
    assert scores[-1, 0].argmax().item() == 0
    assert scores[-1, 1].argmax().item() == 1


def test_score_rejects_invalid_metadata():
    device = _device()
    q = torch.ones(3, 2, 4, device=device)
    k = torch.ones(3, 1, 4, device=device)
    with pytest.raises(ValueError, match="nondecreasing"):
        m3_index_score(q, k, torch.tensor([0, 2, 1, 3], dtype=torch.int32, device=device), 3)
    with pytest.raises(ValueError, match="max_seqlen"):
        m3_index_score(q, k, torch.tensor([0, 3], dtype=torch.int32, device=device), 2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, -1.0, 0.0])
@pytest.mark.parametrize("case", ["cancellation_unique", "cancellation_tie", "rounded_score_tie"])
def test_score_winners_retain_low_bits(case, scale, dtype):
    """Accurate values alone do not guarantee a correct max subgradient."""
    device = _device()
    small = 2**-12
    if case.startswith("cancellation"):
        query = [1.0, small, 1.0]
        key0 = [1.0, small, -1.0]
        key1 = [0.0, small if case.endswith("tie") else 0.0, 0.0]
    else:
        # Both public FP32 scores round to 1, but the exact winner is unique.
        query, key0, key1 = [1.0, 1.0], [1.0, 2**-24], [1.0, 0.0]
    q_cpu = torch.tensor([[query], [query]], dtype=dtype)
    k_cpu = torch.tensor([[key0], [key1]], dtype=dtype)
    q, k = [x.to(device).requires_grad_() for x in (q_cpu, k_cpu)]
    qr, kr = [x.detach().double().requires_grad_() for x in (q_cpu, k_cpu)]
    cu = torch.tensor([0, 2], dtype=torch.int32, device=device)
    actual = m3_index_score(q, k, cu, 2, block_size=2, scale=scale)
    expected = index_score_fp64(qr, kr, cu.cpu(), 2, block_size=2, scale=scale)
    _assert_scores(actual, expected)
    grad = torch.zeros_like(actual)
    grad[1] = 1
    actual.backward(grad)
    expected.backward(grad.cpu().double())
    for tensor, reference in zip((q, k), (qr, kr)):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("exact_tie", [False, True])
def test_fp32_product_residual_changes_winner(exact_tie):
    """Recover multiplication roundoff as well as summation cancellation."""
    device = _device()
    query = [1 + 2**-12, 1.0]
    q_cpu = torch.tensor([[query], [query]], dtype=torch.float32)
    k_cpu = torch.tensor([[[1 + 2**-12, -1 - 2**-11]],
                          [[0.0, 2**-24 if exact_tie else 0.0]]], dtype=torch.float32)
    q, k = [x.to(device).requires_grad_() for x in (q_cpu, k_cpu)]
    qr, kr = [x.detach().double().requires_grad_() for x in (q_cpu, k_cpu)]
    cu = torch.tensor([0, 2], dtype=torch.int32, device=device)
    actual = m3_index_score(q, k, cu, 2, block_size=2)
    expected = index_score_fp64(qr, kr, cu.cpu(), 2, block_size=2)
    _assert_scores(actual, expected)
    grad = torch.zeros_like(actual)
    grad[1] = 1
    actual.backward(grad)
    expected.backward(grad.cpu().double())
    for tensor, reference in zip((q, k), (qr, kr)):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_score_dtype_and_packed_backward(dtype):
    device = _device()
    generator = torch.Generator().manual_seed(928)
    q_cpu = torch.randn(6, 2, 7, generator=generator).to(dtype)
    k_cpu = torch.randn(6, 1, 7, generator=generator).to(dtype)
    q, k = [x.to(device).requires_grad_() for x in (q_cpu, k_cpu)]
    qr, kr = [x.detach().double().requires_grad_() for x in (q_cpu, k_cpu)]
    cu = torch.tensor([0, 2, 2, 6], dtype=torch.int32, device=device)
    actual = m3_index_score(q, k, cu, 8, block_size=4, scale=0.3)
    expected = index_score_fp64(qr, kr, cu.cpu(), 8, block_size=4, scale=0.3)
    _assert_scores(actual, expected)
    grad = torch.randn(actual.shape, generator=generator)
    actual.backward(grad.to(device))
    expected.backward(grad.double())
    for tensor, reference in zip((q, k), (qr, kr)):
        torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype), atol=1e-4, rtol=1e-4)


def test_no_grad_with_trainable_inputs_preserves_score_values():
    device = _device()
    q = torch.tensor([[[1.0, 2**-12, 1.0]]] * 2, dtype=torch.bfloat16,
                     device=device, requires_grad=True)
    k = torch.tensor([[[1.0, 2**-12, -1.0]], [[0.0, 0.0, 0.0]]],
                     dtype=torch.bfloat16, device=device, requires_grad=True)
    cu = torch.tensor([0, 2], dtype=torch.int32, device=device)
    trainable = m3_index_score(q, k, cu, 4, block_size=2)
    with torch.no_grad():
        inference = m3_index_score(q, k, cu, 4, block_size=2)
    assert trainable.requires_grad and not inference.requires_grad
    torch.testing.assert_close(trainable, inference, atol=0, rtol=0)
