"""Explicit NPU diagnostics; this file is excluded from normal test discovery.

Each case must run in a fresh process on NPU. Run with the isolated probe:

    python -m msa_triton.tests.probe_npu_isolated \
        msa_triton/tests/probe_attention_cases.py --output-dir /tmp/msa-attention-cases

CPU validation requires MSA_TEST_DEVICE=cpu and TRITON_INTERPRET=1. The two
experiments modify only this process: separate launches retain the original
KV kernel, while tile-sum-kahan loads an ablated copy from a temporary file.
Neither experiment changes the production implementation or acceptance gate.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from msa_triton.tests.probe_attention_ablation import ablate
from msa_triton.tests.reference_fp64 import sparse_attention_fp64
from msa_triton.tests.test_triton_attention import _assert_lse, _inputs


CASES = (
    "tree-d128", "qkv-d128", "qk-d128", "qv-d128", "split-kv-d128",
    "tile-sum-kahan-d128",
)


def _stage(name):
    print(f"STAGE={name}", flush=True)


def _synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.fixture
def device():
    # A missing dependency/device is a failed probe, never a successful skip.
    name = os.environ.get("MSA_TEST_DEVICE", "npu")
    if name.startswith("npu"):
        importlib.import_module("torch_npu")
    importlib.import_module("triton")
    result = torch.device(name)
    if result.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("CPU probes require TRITON_INTERPRET=1 before importing Triton")
    if result.type != "cpu" and os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Accelerator probes must unset TRITON_INTERPRET")
    return result


class _SplitKVKernel:
    """Intercept only KV launch flags, keeping every pointer and other argument."""

    def __init__(self, kernel, device):
        self.kernel = kernel
        self.device = device
        self.calls = []

    def __getitem__(self, grid):
        original_launch = self.kernel[grid]

        def launch(*args, **kwargs):
            assert kwargs["COMPUTE_DK"] and kwargs["COMPUTE_DV"]
            for need_k, need_v, label in ((True, False, "dk"), (False, True, "dv")):
                constants = dict(kwargs, COMPUTE_DK=need_k, COMPUTE_DV=need_v)
                self.calls.append((need_k, need_v))
                _stage(f"split_kv_{label}_launch")
                original_launch(*args, **constants)
                _stage(f"split_kv_{label}_synchronize")
                _synchronize(self.device)
                _stage(f"split_kv_{label}_complete")

        return launch


class _TracingKernel:
    """Attribute launch/compile and asynchronous failures without changing math."""

    def __init__(self, kernel, device, label):
        self.kernel = kernel
        self.device = device
        self.label = label

    def __getitem__(self, grid):
        original_launch = self.kernel[grid]

        def launch(*args, **kwargs):
            _stage(f"{self.label}_launch")
            result = original_launch(*args, **kwargs)
            _stage(f"{self.label}_synchronize")
            _synchronize(self.device)
            _stage(f"{self.label}_complete")
            return result

        return launch


def _temporary_ablation(module, tmp_path, monkeypatch):
    source = ablate(Path(module.__file__).read_text(), "tile-sum-kahan")
    path = tmp_path / "attention_tile_sum_kahan.py"
    path.write_text(source)
    name = "msa_triton.triton._probe_attention_tile_sum_kahan"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the temporary attention ablation")
    copy = importlib.util.module_from_spec(spec)
    # Keep a package-qualified module name so the copy's relative imports and
    # Triton source inspection work. monkeypatch removes it after this case.
    import sys

    monkeypatch.setitem(sys.modules, name, copy)
    spec.loader.exec_module(copy)
    return copy


@pytest.mark.parametrize("case", CASES)
def test_probe(case, device, monkeypatch, tmp_path):
    _stage(f"case_{case}")
    module = importlib.import_module("msa_triton.triton.sparse_attention")
    if case == "tree-d128":
        tree = importlib.import_module("msa_triton.tests.test_attention_reduction")
        monkeypatch.setattr(tree, "DEVICE", str(device))
        _stage("tree_launch_and_exact_hi_lo_check")
        tree.test_adjacent_pair_tree_preserves_low_components(128)
        _synchronize(device)
        _stage("complete")
        return

    needs_grad = {
        "qk-d128": (True, True, False),
        "qv-d128": (True, False, True),
    }.get(case, (True, True, True))
    # Q remains enabled for both partial-gradient probes, retaining the same
    # query center/mass prepass as the original failing Q/K/V case.
    _stage("input_transfer")
    tensors, indices, cu = _inputs([5, 3], 2, 2, 128, 4, 2, device)
    tensors = tuple(x.detach().requires_grad_(need) for x, need in zip(tensors, needs_grad))
    saved_inputs = tuple(x.detach().clone() for x in (*tensors, indices, cu))
    _synchronize(device)

    _stage("fp64_reference")
    reference_inputs = tuple(
        x.detach().cpu().double().requires_grad_(need)
        for x, need in zip(tensors, needs_grad)
    )
    expected, expected_lse = sparse_attention_fp64(
        *reference_inputs, indices.cpu(), cu.cpu(), 5, block_size=4, scale=None,
    )

    if case == "tile-sum-kahan-d128":
        module = _temporary_ablation(module, tmp_path, monkeypatch)
    kernels = module._kernels()
    split_kernel = None
    kv_kernel = kernels[3]
    if case == "split-kv-d128":
        split_kernel = _SplitKVKernel(kernels[3], device)
        # The original query backward sees all three requested gradients. Both
        # KV launches receive its unchanged stats and the same allocated DK/DV.
        kv_kernel = split_kernel
    patched = (
        kernels[0],
        _TracingKernel(kernels[1], device, "attention_forward"),
        _TracingKernel(kernels[2], device, "query_backward"),
        _TracingKernel(kv_kernel, device, "kv_backward"),
        kernels[4],
    )
    monkeypatch.setattr(module, "_kernels", lambda: patched)
    k2q_module = importlib.import_module("msa_triton.triton.k2q")
    original_build = k2q_module.build_k2q_csr

    def traced_build(*args, **kwargs):
        _stage("k2q_build_start")
        result = original_build(*args, **kwargs)
        _stage("k2q_build_synchronize")
        _synchronize(device)
        _stage("k2q_build_complete")
        return result

    monkeypatch.setattr(k2q_module, "build_k2q_csr", traced_build)

    _stage("forward_launch")
    actual, actual_lse = module.m3_sparse_attention(
        *tensors, indices, cu, 5, block_size=4, scale=None, return_lse=True,
    )
    _stage("forward_synchronize")
    _synchronize(device)
    _stage("forward_check")
    assert actual.dtype == torch.bfloat16
    assert actual_lse.dtype == torch.float32
    assert not actual_lse.requires_grad
    torch.testing.assert_close(
        actual.detach().cpu(), expected.detach().bfloat16(), atol=1e-4, rtol=1e-4,
    )
    _assert_lse(actual_lse, expected_lse)
    grad = torch.randn(actual.shape, generator=torch.Generator().manual_seed(5678)).bfloat16()
    expected.backward(grad.double())
    _stage("gradient_transfer")
    device_grad = grad.to(device)
    _synchronize(device)
    _stage("backward_launch")
    actual.backward(device_grad)
    _stage("backward_synchronize")
    _synchronize(device)
    _stage("backward_check")
    for tensor, reference, need in zip(tensors, reference_inputs, needs_grad):
        if need:
            assert tensor.grad is not None
            torch.testing.assert_close(
                tensor.grad.cpu(), reference.grad.bfloat16(), atol=1e-4, rtol=1e-4,
            )
        else:
            assert tensor.grad is None
    for tensor, saved in zip((*tensors, indices, cu), saved_inputs):
        torch.testing.assert_close(tensor.detach().cpu(), saved.cpu(), atol=0, rtol=0)
    if split_kernel is not None:
        assert split_kernel.calls == [(True, False), (False, True)]
    _stage("complete")
