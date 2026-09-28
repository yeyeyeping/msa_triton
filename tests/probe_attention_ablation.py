"""Run review ablations in temporary source copies, leaving kernels unchanged.

TRITON_INTERPRET=1 python -m msa_triton.tests.probe_attention_ablation \
    --mode no-center --case dim128

The no-center dim128 probe currently fails the unchanged FP64 accuracy test.
This script forwards pytest's exit code; it is not part of normal collection.
The source transforms are specific to the reviewed implementation and should
be re-reviewed if that implementation changes.
"""

import argparse
import importlib.util
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no-center", "tile-sum-kahan"), required=True)
    parser.add_argument("--case", choices=("dim128", "matrix", "rounding"), default="dim128")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    source = (package / "triton" / "sparse_attention.py").read_text()
    source = ablate(source, args.mode)
    expression = {
        "dim128": "forward_backward_against_fp64 and lengths6",
        "matrix": "forward_backward_against_fp64",
        "rounding": "long_key_accumulation_bf16_rounding_regression",
    }[args.case]
    with tempfile.TemporaryDirectory(prefix="msa-review-") as directory:
        path = Path(directory) / "attention_ablation.py"
        path.write_text(source)
        name = "msa_triton.triton._review_attention_ablation"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        class OverrideImplementation:
            def pytest_collection_modifyitems(self, items):
                for item in items:
                    if item.module.__name__.endswith("test_triton_attention"):
                        item.module.m3_sparse_attention = module.m3_sparse_attention

        return pytest.main(
            [str(package / "tests" / "test_triton_attention.py"), "-q", "-k", expression],
            plugins=[OverrideImplementation()],
        )


if __name__ == "__main__":
    raise SystemExit(main())
