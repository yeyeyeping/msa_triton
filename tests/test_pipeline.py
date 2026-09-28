"""Three-stage composition and a small trainable attention module."""

import os

import pytest
import torch

from msa_triton import eager
from msa_triton import triton as optimized
from msa_triton.tests.reference_fp64 import index_score_fp64, sparse_attention_fp64

DEVICE = os.environ.get("MSA_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU Triton execution requires TRITON_INTERPRET=1",
)


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("triton")
    if DEVICE.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(DEVICE)


@pytest.mark.parametrize("lengths,groups,ratio,dim", [([1, 3], 1, 1, 8), ([7, 5], 2, 2, 16), ([17, 9], 4, 1, 8)])
def test_composed_pipeline_fp64_and_native_diagnostic(lengths, groups, ratio, dim, device, capsys):
    gen = torch.Generator().manual_seed(51)
    total = sum(lengths)
    block, topk = 4, 2
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    iq = torch.randn(total, groups, dim, generator=gen).bfloat16().to(device)
    ik = torch.randn(total, 1, dim, generator=gen).bfloat16().to(device)
    q, k, v = [torch.randn(total, heads, dim, generator=gen).bfloat16().to(device).requires_grad_()
               for heads in (groups * ratio, groups, groups)]
    score = optimized.m3_index_score(iq, ik, cu, max(lengths), block_size=block)
    reference_score = index_score_fp64(iq, ik, cu, max(lengths), block_size=block).float()
    torch.testing.assert_close(score.cpu(), reference_score, atol=1e-4, rtol=1e-4)
    ids = optimized.m3_topk(score, cu, block_size=block, topk_blocks=topk)
    out = optimized.m3_sparse_attention(q, k, v, ids, cu, max(lengths), block_size=block)
    ref_inputs = [x.detach().cpu().double().requires_grad_() for x in (q, k, v)]
    ref_out, _ = sparse_attention_fp64(*ref_inputs, ids, cu, block_size=block)
    torch.testing.assert_close(out.cpu(), ref_out.bfloat16(), atol=1e-4, rtol=1e-4)
    grad = torch.randn(out.shape, generator=gen).bfloat16()
    out.backward(grad.to(device))
    ref_out.backward(grad.double())
    for actual, reference in zip((q, k, v), ref_inputs):
        torch.testing.assert_close(actual.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4)

    # Different native eager rounding is reported, never used to relax the oracle gate.
    cpu_qkv = [x.detach().cpu() for x in (q, k, v)]
    native = eager.m3_sparse_attention(*cpu_qkv, ids.cpu(), cu.cpu(), max(lengths), block_size=block)
    difference = (native.float() - out.detach().cpu().float()).abs()
    with capsys.disabled():
        print(f"native_eager_vs_triton lengths={lengths} G={groups} R={ratio} D={dim}: "
              f"max_abs={difference.max().item():.8g}, mean_abs={difference.mean().item():.8g}")


def test_small_module_parameter_gradients(device):
    """Same TND API inside projection -> MSA -> output projection, with frozen indexer."""
    torch.manual_seed(54)
    total, hidden, groups, ratio, dim = 8, 12, 2, 2, 4
    base = [torch.randn(total, hidden), torch.randn(hidden, groups * ratio * dim) * 0.2,
            torch.randn(hidden, groups * dim) * 0.2, torch.randn(hidden, groups * dim) * 0.2,
            torch.randn(groups * ratio * dim, hidden) * 0.2]
    cu = torch.tensor([0, 5, total], dtype=torch.int32, device=device)
    index_q_weight = torch.randn(hidden, groups * dim, device=device, requires_grad=True)
    index_k_weight = torch.randn(hidden, dim, device=device, requires_grad=True)
    results = []
    for implementation in (eager, optimized):
        x, wq, wk, wv, wo = [a.to(device).detach().clone().requires_grad_() for a in base]
        with torch.no_grad():
            iq = (x @ index_q_weight).reshape(total, groups, dim)
            ik = (x @ index_k_weight).reshape(total, 1, dim)
            score = implementation.m3_index_score(iq, ik, cu, 5, block_size=2)
            ids = implementation.m3_topk(score, cu, block_size=2, topk_blocks=2)
        q = (x @ wq).reshape(total, groups * ratio, dim)
        k, v = [(x @ w).reshape(total, groups, dim) for w in (wk, wv)]
        out = implementation.m3_sparse_attention(q, k, v, ids, cu, 5, block_size=2)
        output = out.reshape(total, -1) @ wo
        output.square().mean().backward()
        results.append([output.detach(), *[a.grad for a in (x, wq, wk, wv, wo)]])
    for native, actual in zip(*results):
        torch.testing.assert_close(actual, native, atol=1e-4, rtol=1e-4)
    assert index_q_weight.grad is None
    assert index_k_weight.grad is None
