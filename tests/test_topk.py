"""Selection contracts independently checked against a Python ranking oracle."""

import os

import pytest
import torch

from msa_triton.eager import m3_topk as eager_topk
from msa_triton.triton import m3_topk as native_topk


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("MSA_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(name)


@pytest.mark.parametrize("implementation", [eager_topk, native_topk])
@pytest.mark.parametrize("lengths,block,k,local", [
    ([1, 7, 3], 2, 3, 1), ([127, 128, 129], 128, 16, 1),
    ([17, 0, 9], 4, 2, 0), ([8, 5], 2, 3, 3), ([2], 1, 0, 0),
])
def test_topk_matches_independent_ranking(implementation, lengths, block, k, local, device):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    groups, blocks = 3, (max(lengths) + block - 1) // block
    scores_cpu = torch.randn(sum(lengths), groups, blocks, generator=torch.Generator().manual_seed(82))
    scores = scores_cpu.to(device)
    saved = scores.clone()
    selected = implementation(scores, cu, block_size=block, topk_blocks=k, local_blocks=local)
    assert selected.dtype == torch.int32
    assert selected.shape == (sum(lengths), groups, k)
    torch.testing.assert_close(scores, saved, atol=0, rtol=0)
    selected = selected.cpu()
    start = 0
    for length in lengths:
        for position in range(length):
            current = position // block
            forced = set(range(max(0, current - local + 1), current + 1)) if local else set()
            for group in range(groups):
                candidates = [b for b in range(current + 1) if b not in forced]
                candidates.sort(key=lambda b: -float(scores_cpu[start + position, group, b]))
                expected = forced | set(candidates[:k - len(forced)])
                actual = selected[start + position, group].tolist()
                kept = [b for b in actual if b != -1]
                assert len(kept) == len(set(kept))
                assert set(kept) == expected
                assert actual == kept + [-1] * (k - len(kept))
        start += length


@pytest.mark.parametrize("implementation", [eager_topk, native_topk])
def test_equal_scores_local_and_no_gradient(implementation, device):
    scores = torch.zeros(7, 2, 4, requires_grad=True, device=device)
    cu = torch.tensor([0, 7], dtype=torch.int32, device=device)
    out = implementation(scores, cu, block_size=2, topk_blocks=2, local_blocks=1)
    assert not out.requires_grad
    out = out.cpu()
    for token in range(7):
        for group in range(2):
            valid = out[token, group][out[token, group] >= 0].tolist()
            assert token // 2 in valid
            assert len(valid) == min(2, token // 2 + 1)
            assert len(set(valid)) == len(valid)
            assert all(0 <= b <= token // 2 for b in valid)
