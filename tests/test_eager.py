"""CPU reference/layout regressions and explicit native BF16 diagnostics."""

from __future__ import annotations

import json

import pytest
import torch

from msa_triton.eager import m3_index_score, m3_sparse_attention, m3_topk
from msa_triton.layout import pack_bs, unpack_tnd
from msa_triton.tests.reference_fp64 import index_score_fp64, sparse_attention_fp64


def _cu(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)


def _close(actual, expected):
    expected = expected.to(actual.dtype)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("lengths,max_len", [([1], 1), ([0, 0], 2), ([0, 3, 0, 7], 9), ([1, 3, 7], 9), ([127, 128, 129], 129)])
def test_pack_unpack_values_and_gradient(lengths, max_len):
    x = torch.randn(sum(lengths), 2, 3, requires_grad=True)
    cu = _cu(lengths)
    padded, valid = unpack_tnd(x, cu, max_len, pad_value=-9)
    assert valid.sum() == x.shape[0]
    assert (padded[~valid] == -9).all()
    packed = pack_bs(padded, cu)
    torch.testing.assert_close(x, packed, atol=0, rtol=0)
    upstream = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(packed, x, upstream)[0], upstream, atol=0, rtol=0)


@pytest.mark.parametrize("lengths,groups,dim,block_size", [([1], 1, 3, 4), ([3, 9], 2, 7, 4), ([127, 129], 3, 16, 128), ([257], 2, 8, 128)])
def test_index_score_fp64_forward_backward(lengths, groups, dim, block_size):
    torch.manual_seed(19)
    cu = _cu(lengths)
    q = torch.randn(sum(lengths), groups, dim).to(torch.bfloat16).requires_grad_()
    k = torch.randn(sum(lengths), 1, dim).to(torch.bfloat16).requires_grad_()
    q_ref, k_ref = [x.detach().double().requires_grad_() for x in (q, k)]
    actual = m3_index_score(q, k, cu, max(lengths), block_size=block_size)
    expected = index_score_fp64(q_ref, k_ref, cu, max(lengths), block_size=block_size)
    _close(actual, expected)
    grad = torch.randn_like(actual).masked_fill(~torch.isfinite(actual), 0)
    actual_grads = torch.autograd.grad(actual, (q, k), grad)
    expected_grads = torch.autograd.grad(expected, (q_ref, k_ref), grad.double())
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        _close(actual_grad, expected_grad)


def test_score_ties_split_gradient_and_future_mask_is_zero():
    q = torch.ones(3, 1, 1, requires_grad=True)
    k = torch.ones(3, 1, 1, requires_grad=True)
    score = m3_index_score(q, k, _cu([3]), 4, block_size=4)
    score[-1].sum().backward()
    torch.testing.assert_close(q.grad[:, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(k.grad[:, 0, 0], torch.full((3,), 1 / 3))
    q.grad = k.grad = None
    m3_index_score(q, k, _cu([3]), 4, block_size=4)[0].sum().backward()
    torch.testing.assert_close(k.grad[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_topk_groups_local_budget_and_no_mutation():
    scores = torch.tensor([[[10., 2., 0.], [1., 9., 0.]]]).expand(5, -1, -1).clone()
    original = scores.clone()
    ids = m3_topk(scores, _cu([5]), block_size=2, topk_blocks=2, local_blocks=1)
    assert set(ids[-1, 0].tolist()) == {0, 2}
    assert set(ids[-1, 1].tolist()) == {1, 2}
    assert ids[0, 0].tolist() == [0, -1]
    assert ids.dtype == torch.int32 and not ids.requires_grad
    torch.testing.assert_close(scores, original, atol=0, rtol=0)
    for t in range(5):
        for group in range(2):
            row = ids[t, group].tolist()
            valid = [b for b in row if b >= 0]
            assert len(valid) == len(set(valid))
            assert row == valid + [-1] * (2 - len(valid))
            assert all(b <= t // 2 for b in valid)


@pytest.mark.parametrize("k,local", [(0, 0), (1, 0), (8, 1), (8, 8)])
def test_topk_short_sequence_fixed_slot_count(k, local):
    scores = torch.zeros(2, 2, 1)
    ids = m3_topk(scores, _cu([1, 1]), block_size=4, topk_blocks=k, local_blocks=local)
    assert ids.shape == (2, 2, k)
    if k:
        assert (ids[..., 0] == 0).all()
        assert (ids[..., 1:] == -1).all()


def _inputs(lengths, groups, ratio, dim, dtype=torch.float32):
    total = sum(lengths)
    return [torch.randn(total, heads, dim).to(dtype).requires_grad_() for heads in (groups * ratio, groups, groups)]


def _all_blocks(lengths, groups, block_size):
    slots = (max(lengths) + block_size - 1) // block_size
    return m3_topk(torch.zeros(sum(lengths), groups, slots), _cu(lengths), block_size=block_size, topk_blocks=slots, local_blocks=0)


@pytest.mark.parametrize("lengths,groups,ratio,dim,block_size", [([1], 1, 1, 3, 4), ([3, 9], 2, 3, 7, 4), ([127, 128, 129], 2, 2, 16, 128), ([257], 1, 2, 8, 128)])
def test_attention_float32_fp64_forward_backward(lengths, groups, ratio, dim, block_size):
    torch.manual_seed(71)
    qkv = _inputs(lengths, groups, ratio, dim)
    refs = [x.detach().double().requires_grad_() for x in qkv]
    cu, indices = _cu(lengths), _all_blocks(lengths, groups, block_size)
    out, lse = m3_sparse_attention(*qkv, indices, cu, max(lengths), block_size=block_size, return_lse=True)
    expected, expected_lse = sparse_attention_fp64(*refs, indices, cu, max(lengths), block_size=block_size)
    _close(out, expected)
    _close(lse, expected_lse)
    assert not lse.requires_grad
    grad = torch.randn_like(out)
    for actual_grad, expected_grad in zip(torch.autograd.grad(out, qkv, grad), torch.autograd.grad(expected, refs, grad.double())):
        _close(actual_grad, expected_grad)


def test_all_invalid_attention_rows_and_padding_have_zero_gradient():
    torch.manual_seed(7)
    qkv = _inputs([2, 5], 2, 2, 3)
    indices = torch.full((7, 2, 3), -1, dtype=torch.int32)
    out, lse = m3_sparse_attention(*qkv, indices, _cu([2, 5]), 8, block_size=4, return_lse=True)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()
    for grad in torch.autograd.grad(out, qkv, torch.randn_like(out)):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("lengths", [[2, 5], [0, 2, 0, 5], [0, 0]])
def test_zero_selection_slots_have_zero_output_and_gradients(lengths):
    torch.manual_seed(9)
    qkv = _inputs(lengths, 2, 2, 3, torch.bfloat16)
    indices = torch.empty(sum(lengths), 2, 0, dtype=torch.int32)
    out, lse = m3_sparse_attention(
        *qkv, indices, _cu(lengths), 8, block_size=4, return_lse=True,
    )
    assert out.shape == qkv[0].shape and lse.shape == qkv[0].shape[:2]
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert not lse.requires_grad
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()
    for tensor, grad in zip(qkv, torch.autograd.grad(out, qkv, torch.randn_like(out))):
        assert grad.shape == tensor.shape and grad.dtype == tensor.dtype
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


def test_no_cross_sequence_values_or_gradients():
    torch.manual_seed(3)
    lengths, block_size = [3, 5], 2
    qkv = _inputs(lengths, 2, 2, 5)
    ids = _all_blocks(lengths, 2, block_size)
    out = m3_sparse_attention(*qkv, ids, _cu(lengths), 5, block_size=block_size)
    grads = torch.autograd.grad(out[:3].sum(), qkv)
    assert all(torch.count_nonzero(grad[3:]) == 0 for grad in grads)
    changed = [torch.cat([x[:3], x[3:] + 1000]) for x in qkv]
    again = m3_sparse_attention(*changed, ids, _cu(lengths), 5, block_size=block_size)
    torch.testing.assert_close(out[:3], again[:3], atol=0, rtol=0)


def _upstream_padded_attention(q, k, v, ids, lengths, block_size):
    """Literal upstream math on BSND rows; independent adapter/mask building."""
    batch, length, heads = q.shape[:3]
    ratio = heads // k.shape[2]
    query = q.transpose(1, 2)
    key = k.transpose(1, 2).repeat_interleave(ratio, dim=1)
    value = v.transpose(1, 2).repeat_interleave(ratio, dim=1)
    mask = torch.zeros(batch, heads, length, length, dtype=q.dtype)
    for b, real_length in enumerate(lengths):
        for t in range(length):
            for head in range(heads):
                selected = ids[b, t, head // ratio].tolist()
                for j in range(length):
                    if t >= real_length or j >= real_length or j > t or j // block_size not in selected:
                        mask[b, head, t, j] = torch.finfo(q.dtype).min
    weights = torch.matmul(query, key.transpose(2, 3)) * (q.shape[-1] ** -0.5)
    weights = weights + mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tnd_adapter_matches_original_bsnd_math_and_gradients(dtype):
    torch.manual_seed(97)
    lengths = [3, 7]
    qkv = _inputs(lengths, 2, 2, 8, dtype)
    native = [x.detach().clone().requires_grad_() for x in qkv]
    indices = _all_blocks(lengths, 2, 4)
    # Force different group selections on late queries while retaining self.
    indices[7:, 0, 0] = -1
    actual = m3_sparse_attention(*qkv, indices, _cu(lengths), 7, block_size=4)
    # Build the padded native input without using the implementation's layout
    # helpers, so this checks the wrapper as well as the numerical operations.
    padded = []
    for x in [*native, indices]:
        rows, start = [], 0
        for length in lengths:
            fill = -1 if x is indices else 0
            pad = x.new_full((max(lengths) - length, *x.shape[1:]), fill)
            rows.append(torch.cat([x[start:start + length], pad]))
            start += length
        padded.append(torch.stack(rows))
    expected_bs = _upstream_padded_attention(*padded, lengths, 4)
    expected = torch.cat([expected_bs[b, :n] for b, n in enumerate(lengths)])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    for a, e in zip(torch.autograd.grad(actual, qkv, upstream), torch.autograd.grad(expected, native, upstream)):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_native_bfloat16_fp64_diagnostic(capsys):
    """Report native rounding vs strict FP64; this is not the kernel accuracy gate."""
    torch.manual_seed(211)
    lengths = [3, 17]
    qkv = _inputs(lengths, 2, 2, 16, torch.bfloat16)
    refs = [x.detach().double().requires_grad_() for x in qkv]
    ids = _all_blocks(lengths, 2, 4)
    out, lse = m3_sparse_attention(*qkv, ids, _cu(lengths), 17, block_size=4, return_lse=True)
    expected, expected_lse = sparse_attention_fp64(*refs, ids, _cu(lengths), 17, block_size=4)
    upstream = torch.randn_like(out)
    gradients = torch.autograd.grad(out, qkv, upstream)
    ref_gradients = torch.autograd.grad(expected, refs, upstream.double())
    report = {}
    for name, actual, ref in zip(["out", "lse", "dq", "dk", "dv"], [out, lse, *gradients], [expected, expected_lse, *ref_gradients]):
        rounded = ref.to(actual.dtype)
        error = (actual.float() - rounded.float()).abs()
        tolerance = 1e-4 + 1e-4 * rounded.float().abs()
        report[name] = {"max_abs": error.max().item(), "strict_failures": (error > tolerance).sum().item(), "elements": actual.numel()}
        assert torch.isfinite(actual).all()
    with capsys.disabled():
        print("native_bfloat16_fp64_diagnostic=" + json.dumps(report, sort_keys=True))
