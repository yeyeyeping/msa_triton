"""Private wide address arithmetic shared by the score/attention kernels.

Public sequence and CSR metadata remain int32. Widen the leading index before
multiplication/addition: converting an already-overflowed offset cannot fix it.
"""

import triton
import triton.language as tl


@triton.jit
def tnd_offset(token, head, heads: tl.constexpr, dim: tl.constexpr):
    return (token.to(tl.int64) * heads + head) * dim


@triton.jit
def advance_offset(offset, delta):
    return offset.to(tl.int64) + delta
