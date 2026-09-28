"""Synthetic stage timings; real accelerator runs are required for performance claims."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time

import torch

from .triton import m3_index_score, m3_sparse_attention, m3_topk
from .triton.k2q import build_k2q_csr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--index-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--local-blocks", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action="store_true", help="Also time score and attention forward+backward")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(args.device)
    if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        parser.error("CPU smoke timing requires TRITON_INTERPRET=1 (not a performance benchmark)")
    if args.repeats < 1 or args.warmup < 0 or any(n <= 0 for n in args.lengths):
        parser.error("lengths and repeats must be positive; warmup must be nonnegative")
    if args.q_heads % args.kv_heads:
        parser.error("q-heads must be divisible by kv-heads")
    runtime = getattr(torch, device.type) if device.type != "cpu" else None

    def synchronize():
        if runtime is not None:
            runtime.synchronize()

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        samples = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            fn()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    versions = {}
    for package in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    print(json.dumps({"environment": versions, "device": str(device),
                      "cpu_interpreter_only": device.type == "cpu"}, ensure_ascii=False))
    generator = torch.Generator().manual_seed(71)
    for length in args.lengths:
        def tensor(heads, dim):
            return torch.randn(length, heads, dim, generator=generator).bfloat16().to(device).requires_grad_(args.backward)

        iq, ik = tensor(args.kv_heads, args.index_dim), tensor(1, args.index_dim)
        q, k, v = tensor(args.q_heads, args.dim), tensor(args.kv_heads, args.dim), tensor(args.kv_heads, args.dim)
        cu = torch.tensor([0, length], dtype=torch.int32, device=device)
        def score():
            return m3_index_score(iq, ik, cu, length, block_size=args.block_size)

        def select(scores):
            return m3_topk(scores, cu, block_size=args.block_size, topk_blocks=args.topk, local_blocks=args.local_blocks)

        with torch.no_grad():
            saved_scores = score()
            ids = select(saved_scores)

        def attention():
            return m3_sparse_attention(q, k, v, ids, cu, length, block_size=args.block_size)

        def pipeline():
            with torch.no_grad():
                selected = select(score())
            return m3_sparse_attention(q, k, v, selected, cu, length, block_size=args.block_size)

        if runtime is not None:
            runtime.reset_peak_memory_stats()
        result = {"length": length, "q_heads": args.q_heads, "kv_heads": args.kv_heads,
                  "dim": args.dim, "block_size": args.block_size, "topk": args.topk}
        with torch.no_grad():
            result["score_fwd_ms"] = measure(score)
            result["topk_ms"] = measure(lambda: select(saved_scores))
            result["k2q_csr_ms"] = measure(lambda: build_k2q_csr(ids, cu, block_size=args.block_size))
            result["attention_fwd_ms"] = measure(attention)
            result["pipeline_fwd_ms"] = measure(pipeline)
        if args.backward:
            dscore = torch.ones_like(saved_scores)
            dout = torch.ones_like(q)

            def score_backward():
                iq.grad = ik.grad = None
                score().backward(dscore)

            def attention_backward():
                q.grad = k.grad = v.grad = None
                attention().backward(dout)

            result["score_fwd_bwd_ms"] = measure(score_backward)
            result["attention_fwd_bwd_ms"] = measure(attention_backward)
        if runtime is not None:
            result["peak_allocated_bytes"] = runtime.max_memory_allocated()
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
