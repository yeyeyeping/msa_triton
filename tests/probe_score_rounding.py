"""Review diagnostic for the score-rounding counterexamples and their fixes.

Run from the workspace root:
  TRITON_INTERPRET=1 python -m msa_triton.tests.probe_score_rounding

This is deliberately not a collected pytest test: the corresponding accuracy
assertions now live in test_triton_score.py; this probe prints detailed values.
Exit status zero means the diagnostic ran, not that the accuracy gate passed.
"""

import json

import torch

from msa_triton.tests.reference_fp64 import index_score_fp64
from msa_triton.triton import m3_index_score


def probe(exact_reference_tie: bool) -> dict:
    small = 2**-12
    q = torch.tensor([[[1.0, small, 1.0]]] * 2, dtype=torch.bfloat16, requires_grad=True)
    other_key = [0.0, small, 0.0] if exact_reference_tie else [0.0, 0.0, 0.0]
    k = torch.tensor([[[1.0, small, -1.0]], [other_key]], dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    qr, kr = [x.detach().double().requires_grad_() for x in (q, k)]
    actual = m3_index_score(q, k, cu, 2, block_size=2)
    reference = index_score_fp64(qr, kr, cu, 2, block_size=2)
    upstream = torch.zeros_like(actual)
    upstream[1] = 1
    actual.backward(upstream)
    reference.backward(upstream.double())
    report = {"case": "fp64_exact_tie" if exact_reference_tie else "fp64_unique_winner"}
    for name, value, expected in (
        ("score", actual, reference), ("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad)
    ):
        expected = expected.to(value.dtype)
        report[name] = {
            "actual": value.detach().tolist(),
            "reference": expected.detach().tolist(),
            "max_abs": (value.detach().float() - expected.detach().float()).abs().max().item(),
            "within_tolerance": bool(torch.allclose(value, expected, atol=1e-4, rtol=1e-4)),
        }
    return report


if __name__ == "__main__":
    for exact_reference_tie in (False, True):
        print(json.dumps(probe(exact_reference_tie), sort_keys=True))
