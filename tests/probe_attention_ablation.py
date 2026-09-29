"""Run explicit diagnostic kernel ablations, leaving production files unchanged.

TRITON_INTERPRET=1 python -m msa_triton.tests.probe_attention_ablation \
    --mode no-center --case dim128

No-center modifies the current query prepass. Tile-sum-Kahan modifies only the
frozen 6d47514 KV tree; it is a historical compiler comparison with known failures
on the cancellation tests, not an accepted implementation. This script forwards
pytest's exit code and is excluded from normal test discovery.
"""

import argparse
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile

import pytest


def replace_region(source, first, last, replacement):
    if source.count(first) != 1 or source.count(last) != 1:
        raise RuntimeError("Reviewed source changed; re-check the ablation before running it")
    start, end = source.index(first), source.index(last)
    if start >= end:
        raise RuntimeError("Ablation source markers are out of order")
    return source[:start] + replacement + source[end:]


def ablate(source, mode):
    if mode == "no-center":
        return replace_region(
            source,
            "        # Refine the softmax derivative's center using selected probabilities.\n",
            "        if COMPUTE_DQ or ACCUMULATE_KV:\n",
            "",
        )
    if mode != "tile-sum-kahan":
        raise ValueError(f"Unknown attention ablation: {mode}")
    replacement = """            if COMPUTE_DK:
                grad_probability = tl.sum(grad_output * value[None, :], 1)
                grad_logits = probability * ((grad_probability - center) - centered_delta) * SCALE
                grad_key = grad_logits[:, None] * query
                key_addition = tl.sum(grad_key, 0)
                key_adjusted = key_addition - key_lo
                key_updated = key_hi + key_adjusted
                key_lo = (key_updated - key_hi) - key_adjusted
                key_hi = key_updated
            if COMPUTE_DV:
                grad_value = probability[:, None] * grad_output
                value_addition = tl.sum(grad_value, 0)
                value_adjusted = value_addition - value_lo
                value_updated = value_hi + value_adjusted
                value_lo = (value_updated - value_hi) - value_adjusted
                value_hi = value_updated
"""
    source = replace_region(
        source,
        "            zeros = tl.full((32, BLOCK_D), 0.0, tl.float32)\n",
        "        offset += 32\n",
        replacement,
    )
    # Standard Kahan's correction is not the low component of a TwoSum pair.
    for name in ("key", "value"):
        source = source.replace(f"{name}_hi + {name}_lo, ds < D)", f"{name}_hi, ds < D)", 1)
    return source


def _query_ablation(module, directory):
    source = "from __future__ import annotations\n\n"
    source += ablate(inspect.getsource(module._backward_kernel), "no-center")
    path = Path(directory) / "attention_query_no_center.py"
    path.write_text(source)
    name = "msa_triton.tests._temporary_attention_query_no_center"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load query-prepass ablation")
    copy = importlib.util.module_from_spec(spec)
    sys.modules[name] = copy
    spec.loader.exec_module(copy)
    copy.tl, copy._tnd_offset = module.tl, module._tnd_offset
    return module._kernels()[0].jit(copy._backward_kernel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no-center", "tile-sum-kahan"), required=True)
    parser.add_argument("--case", choices=("dim128", "matrix", "rounding", "cancellation"), default="dim128")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    expression = {
        "dim128": "forward_backward_against_fp64 and lengths6",
        "matrix": "forward_backward_against_fp64",
        "rounding": "long_key_accumulation_bf16_rounding_regression",
        "cancellation": "kv_within_tile_cancellation_against_fp64",
    }[args.case]
    module = importlib.import_module("msa_triton.triton.sparse_attention")
    with tempfile.TemporaryDirectory(prefix="msa-review-") as directory, pytest.MonkeyPatch.context() as patch:
        kernels = list(module._kernels())
        if args.mode == "tile-sum-kahan":
            from msa_triton.tests._attention_tree_reference import load_legacy_kv

            kernels[3] = load_legacy_kv(directory, transform=lambda source: ablate(source, args.mode))
        else:
            kernels[2] = _query_ablation(module, directory)
        patched = tuple(kernels)
        patch.setattr(module, "_kernels", lambda: patched)
        return pytest.main(
            [str(package / "tests" / "test_triton_attention.py"), "-q", "-k", expression],
        )


if __name__ == "__main__":
    raise SystemExit(main())
