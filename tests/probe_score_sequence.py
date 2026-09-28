"""Stress score kernels in one process, stopping at the first failed stage.

Run from the package's parent directory, for example::

    python -m msa_triton.tests.probe_score_sequence --device npu --repeats 20
    python -m msa_triton.tests.probe_score_sequence --device cpu --repeats 1

The CPU option enables the Triton interpreter before importing Torch/Triton.
NPU execution requires a working device and TRITON_INTERPRET to be unset.
JSON lines identify each action before it starts, including synchronizations.
This is a diagnostic, not a collected pytest test or a performance benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


# Keep this order identical to test_score_forward_backward_fp64: the report
# observed failures only after previous shapes had run in the same process.
CASES = (
    ([1], 1, 1, 4, 1.0),
    ([0, 0], 2, 7, 4, 1.0),
    ([0, 3, 0, 2, 0], 2, 7, 4, 1.0),
    ([3, 7], 2, 7, 4, 1.0),
    ([7, 11], 4, 16, 8, 0.5),
    ([9], 2, 64, 4, 1.0),
    ([5], 4, 128, 128, 1.0),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _git_value(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _emit(context: dict, stage: str, **details) -> None:
    context["stage"] = stage
    print(json.dumps({**context, **details}, sort_keys=True, allow_nan=False), flush=True)


def _run(args, context: dict) -> None:
    _emit(context, "configure_environment", device=args.device)
    if args.device == "cpu":
        os.environ["TRITON_INTERPRET"] = "1"
    elif args.device == "npu" or (args.device.startswith("npu:") and args.device[4:].isdigit()):
        if os.environ.get("TRITON_INTERPRET") == "1":
            raise RuntimeError("NPU validation requires TRITON_INTERPRET to be unset")
    else:
        raise ValueError("--device must be cpu, npu, or npu:<logical device index>")

    _emit(context, "import_runtime")
    import torch
    import triton

    npu_version = None
    if args.device.startswith("npu"):
        import torch_npu

        npu_version = torch_npu.__version__
        if not torch.npu.is_available():
            raise RuntimeError("NPU was requested, but torch.npu.is_available() is false")

    from msa_triton.tests.reference_fp64 import index_score_fp64
    from msa_triton.triton.index_score import m3_index_score

    _emit(context, "initialize_device")
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)

    def synchronize(stage: str) -> None:
        _emit(context, stage)
        if device.type == "npu":
            torch.npu.synchronize()

    synchronize("synchronize_initialization")
    package_dir = Path(__file__).resolve().parents[1]
    git_status = _git_value(package_dir, "status", "--porcelain")
    _emit(
        context, "environment", torch_version=torch.__version__,
        torch_npu_version=npu_version, triton_version=triton.__version__,
        python_version=sys.version, package_path=str(package_dir),
        triton_path=str(Path(triton.__file__).resolve()),
        git_revision=_git_value(package_dir, "rev-parse", "HEAD"),
        git_dirty=None if git_status is None else bool(git_status),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        ascend_launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        triton_interpret=os.environ.get("TRITON_INTERPRET"),
        device=str(device), dtype=args.dtype, repeats=args.repeats,
        forward_only=args.forward_only, atol=1e-4, rtol=1e-4,
    )
    dtype = {"BF16": torch.bfloat16, "FP16": torch.float16, "FP32": torch.float32}[args.dtype]
    needs_grad = not args.forward_only

    for iteration in range(args.repeats):
        for case_index, (lengths, groups, dim, block_size, scale) in enumerate(CASES):
            # Iteration zero reproduces the original pytest data for every
            # shape; later iterations vary it without changing case order.
            seed = 731 + iteration
            context.update(
                iteration=iteration, case_index=case_index, seed=seed,
                shape={"lengths": lengths, "groups": groups, "dimension": dim,
                       "block_size": block_size, "scale": scale},
            )
            _emit(context, "prepare_cpu_inputs")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            q_cpu = torch.randn(sum(lengths), groups, dim, generator=generator).to(dtype)
            k_cpu = torch.randn(sum(lengths), 1, dim, generator=generator).to(dtype)
            cu_cpu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)
            max_seqlen = max(lengths) + block_size
            blocks = (max_seqlen + block_size - 1) // block_size
            upstream_cpu = torch.randn(sum(lengths), groups, blocks, generator=generator).to(dtype).float()
            qr = q_cpu.double().requires_grad_(needs_grad)
            kr = k_cpu.double().requires_grad_(needs_grad)

            _emit(context, "oracle_forward")
            expected = index_score_fp64(
                qr, kr, cu_cpu, max_seqlen, block_size=block_size, scale=scale,
            )
            if needs_grad:
                _emit(context, "oracle_backward")
                expected.backward(upstream_cpu.double())

            _emit(context, "transfer_inputs")
            q = q_cpu.to(device).clone().requires_grad_(needs_grad)
            k = k_cpu.to(device).clone().requires_grad_(needs_grad)
            cu = cu_cpu.to(device)
            upstream = upstream_cpu.to(device) if needs_grad else None
            synchronize("synchronize_transfer")

            _emit(context, "forward")
            with torch.set_grad_enabled(needs_grad):
                actual = m3_index_score(q, k, cu, max_seqlen, block_size=block_size, scale=scale)
            synchronize("synchronize_forward")

            _emit(context, "compare_forward")
            actual_cpu = actual.detach().cpu()
            expected_cpu = expected.detach().float()
            if actual_cpu.dtype != torch.float32:
                raise AssertionError(f"public score dtype must be FP32, got {actual_cpu.dtype}")
            torch.testing.assert_close(actual_cpu, expected_cpu, atol=1e-4, rtol=1e-4)

            if needs_grad:
                _emit(context, "backward")
                actual.backward(upstream)
                synchronize("synchronize_backward")
                _emit(context, "compare_backward")
                torch.testing.assert_close(q.grad.cpu(), qr.grad.to(dtype), atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(k.grad.cpu(), kr.grad.to(dtype), atol=1e-4, rtol=1e-4)
            _emit(context, "passed", checked_gradients=needs_grad)
            # Release this case before allocating the next one, allowing the
            # device allocator to reuse storage across the changing shapes.
            del q, k, cu, upstream, actual, actual_cpu, expected, expected_cpu, qr, kr

    _emit(context, "complete", passed_cases=args.repeats * len(CASES))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu", help="npu (default), npu:<index>, or cpu")
    parser.add_argument("--repeats", type=_positive_int, default=20)
    parser.add_argument("--dtype", type=str.upper, choices=("BF16", "FP16", "FP32"), default="BF16")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    context = {"iteration": None, "case_index": None, "shape": None, "stage": "startup"}
    try:
        _run(args, context)
    except Exception as error:
        # Do not perform additional device operations or attempt another case:
        # an illegal access may already have poisoned the NPU context.
        failed_stage = context["stage"]
        _emit(context, "failed", failed_stage=failed_stage,
              error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
