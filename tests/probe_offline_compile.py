"""Compile portable kernels with upstream Triton's CUDA sm80 target on CPU.

Unset TRITON_INTERPRET before running this optional diagnostic. No GPU is
required or used. This checks Triton language lowering, not NPU compatibility,
hardware arithmetic, or performance; it is not part of pytest collection.
"""

from itertools import product
import os


def main():
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Unset TRITON_INTERPRET for offline compilation")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from msa_triton.triton import index_score as score
    from msa_triton.triton.sparse_attention import _kernels

    integer_pointers = {"INDICES", "CU", "CU_BLOCKS", "ROW_PTR", "QUERIES", "TIE_INFO"}
    compiled = []

    def check(fn, values, *, disable_fusion=False, unused_score_state=False):
        constants = {name: value for name, value in values.items() if name in fn.arg_names}
        signature = {
            name: "constexpr" if name in constants else "*i32" if name in integer_pointers else "*fp32"
            for name in fn.arg_names
        }
        if unused_score_state:
            # The production no-grad launch aliases TIE_INFO to FP32 scores.
            signature["TIE_INFO"] = "*fp32"
        result = triton.compile(
            ASTSource(fn, signature, constexprs=constants),
            target=GPUTarget("cuda", 80, 32),
            options={"enable_fp_fusion": False} if disable_fusion else {},
        )
        assert "ptx" in result.asm
        compiled.append(fn.__name__)

    for dim, groups, block, scale, num_blocks, num_sequences in [
        (3, 1, 2, 1.0, 4, 3), (16, 4, 16, -1.0, 4, 3), (128, 4, 128, 0.0, 4, 3),
        # The first runtime failure in the NPU report: [0,3,0,2,0], D=7.
        (7, 2, 4, 1.0, 2, 5),
    ]:
        dimension_tile = triton.next_power_of_2(dim)
        constants = dict(
            G=groups, D=dim, NB=num_blocks, BLOCK=block, SCALE=scale,
            N_SEQS=num_sequences, SEQ_TILE=triton.next_power_of_2(num_sequences),
            KEY_TILE=triton.next_power_of_2(block), QUERY_TILE=32,
            DIM_TILE=dimension_tile, LOG_DIM=dimension_tile.bit_length() - 1,
            GROUP_TILE=triton.next_power_of_2(groups),
        )
        for save_state in (True, False):
            check(score._index_score_forward, dict(constants, SAVE_STATE=save_state),
                  disable_fusion=True, unused_score_state=not save_state)
        check(score._index_score_backward_q, constants, disable_fusion=True)
        check(score._index_score_backward_k_group, constants, disable_fusion=True)
        check(score._index_score_reduce_k, constants)

    _, forward, backward, kv, finish = _kernels()
    constants = dict(
        H_Q=64, H_KV=4, D=128, TOPK=16, NSEQ=4, BLOCK_SIZE=128,
        SCALE=128**-0.5, BLOCK_D=128, BLOCK_N=32, BLOCK_META=4,
    )
    check(forward, constants)
    for dq, dk, dv in product((False, True), repeat=3):
        if dq or dk or dv:
            check(backward, dict(constants, COMPUTE_DQ=dq, COMPUTE_DK=dk,
                                 COMPUTE_DV=dv, WRITE_STATS=dk or dv, ACCUMULATE_KV=False))
    for dk, dv in [(True, False), (False, True), (True, True)]:
        flags = dict(COMPUTE_DK=dk, COMPUTE_DV=dv)
        check(kv, dict(constants, **flags), disable_fusion=True)
        check(finish, dict(SIZE=32768, TILE=256, **flags))
    for dk, dv in [(True, False), (False, True)]:
        check(backward, dict(constants, COMPUTE_DQ=False, COMPUTE_DK=dk,
                             COMPUTE_DV=dv, WRITE_STATS=False, ACCUMULATE_KV=True))
    # Exact reported attention specialization, not only the larger GQA case.
    reported = dict(constants, H_Q=4, H_KV=2, TOPK=2, NSEQ=2,
                    BLOCK_SIZE=4, BLOCK_N=4, BLOCK_META=2)
    check(forward, reported)
    check(backward, dict(reported, COMPUTE_DQ=True, COMPUTE_DK=True,
                         COMPUTE_DV=True, WRITE_STATS=True, ACCUMULATE_KV=False))
    for dk, dv in [(True, False), (False, True), (True, True)]:
        check(kv, dict(reported, COMPUTE_DK=dk, COMPUTE_DV=dv), disable_fusion=True)
    print(f"{len(compiled)} CUDA sm80 offline compile variants passed; "
          "no hardware execution or NPU compilation")


if __name__ == "__main__":
    main()
