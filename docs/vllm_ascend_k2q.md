# vLLM Ascend 的 MSA k2q 实现与训练适配

核对日期：2026-09-28。参考版本固定为 vLLM Ascend
[`a99c3a30737fc5e84ddaee8fbfc5274b789bd99b`](https://github.com/vllm-project/vllm-ascend/commit/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b)
（2026-09-23）。本文区分已核实的上游推理实现和本项目的训练适配；不表示完成了 NPU 性能验收。

## 1. k2q 改变的是访问方向

`q2k` 回答“query 选择了哪些 KV blocks”；`k2q` 回答“某个 KV block 被哪些 queries 选择”。它反转同一组稀疏边的索引方向，不交换 Q/K 张量的数学角色，也不把 softmax 改成沿 query 归一化。

```text
q2k: query 0 → blocks [0, 2]
     query 1 → blocks [1, 2]

k2q: block 0 → [(query 0, slot 0)]
     block 1 → [(query 1, slot 0)]
     block 2 → [(query 0, slot 1), (query 1, slot 1)]
```

这是稀疏图转置。每条边仍然要应用 token 级 causal mask；反转 block 索引不等于 block 内所有 tokens 都可见。转换的 query 与 slot 写入语义可直接核对 [AscendC scatter 实现](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_mc.h)。

## 2. 上游的真实接口和布局

令 `G` 为 KV/index groups，`T` 为 packed query tokens，`K` 为 top-k 容量，
`R = sum_b ceil(kv_length_b / block_size)` 为所有请求的逻辑 KV blocks 数。

| 参数 | 上游形状/语义 |
|---|---|
| `q2k` | int32 `[G,T,K]`，存序列内逻辑 block id，空位 `-1` |
| `cu_seqlens` | int32 `[B+1]`，packed query 边界 |
| `cu_block_lens` | int32 `[B+1]`，各请求逻辑 KV block 数的累加和 |
| `row_ptr` | int32 `[G,R+1]`，每个 group 的 CSR 行指针 |
| `q_ind` | int32 `[G,T*K]`，边容量缓冲区；有效区间由 row_ptr 确定 |
| `slot` | int32 `[G,T*K]`，该边在原 query top-k 数组中的槽位 |

`q_global_offset=1` 时 `q_ind` 为整个 packed Q 中的 token 下标；为 0 时为请求内下标。无效容量填 `-1`。
`order_method=0` 按请求依次连接 KV blocks；为 1 时按 block 层级轮询请求。官方 prefill 调用使用 `order_method=1`、全局 query 下标。
完整张量分配和参数定义见 [C++ adapter](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/csrc/attention/k2q_csr/k2q_csr_torch_adpt.h)，行顺序见 [Meta kernel](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_stage_meta.h)。

因此，本项目的公共 `indices[T,G,K]` 不能直接当成上游 `q2k[G,T,K]` 传入。即使转置后形状匹配，也仍需区分 CSR 行顺序、分页缓存与 packed KV 的寻址方式。

## 3. 上游推理链路：哪些是 Triton，哪些是自定义算子

```text
index Q + paged index K
    → block scores / top-k
    → q2k[G,T,K]
    → AscendC k2q CSR
    → KV-gather-Q prefill 自定义算子
    → TND output
```

Indexer 存在 Triton score、score preparation 和 top-k mask 路径；部分路径也调用 AscendC index-score。不能把整个 MSA 描述成单一 Triton 实现。相关入口见 [msa_m3_triton.py](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/vllm_ascend/models/minimax_m3/ops/msa_m3_triton.py)。

`npu_k2q_csr` 由五个 AscendC 阶段组成：

1. Meta：建立 token→请求映射和 KV block→CSR 行映射。
2. Hist：统计各行的稀疏边数。
3. RowPrefix：生成 CSR 行前缀和。
4. TilePrefix：划定并行 query tiles 在各行内的写入范围。
5. Scatter：写入 query 下标及原 top-k slot。

这是上游的 native custom-op pipeline；不是 Triton kernel。[上游 CSR 说明](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/csrc/attention/k2q_csr/README.md)

随后 `npu_sparse_attention_score_prefill` 将 CSR、Q、分页 K/V、block table 传给
`aclnnMinimaxSparseAttentionSplitKv`。Python 路径标记为 `no_grad`；A3 BF16 路径使用 `inner_precise=0`。这个分支依赖兼容的可选 vendor Split-KV 包；不可用时，上游走原来的 Q-gather-KV 实现。[Python 分派与调用](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py#L533)

该 C++ wrapper 明确传入 `input_layout="TND"`，并把 `softmax_lse_flag` 设为 false、使用空 FP32 LSE 占位输出。本次核对到的是 vendor 算子的调用边界，不能据此声称其内部 kernel 已被完整迁移或提供了 backward。[C++ attention binding](https://github.com/vllm-project/vllm-ascend/blob/a99c3a30737fc5e84ddaee8fbfc5274b789bd99b/csrc/torch_binding.cpp)

## 4. 本项目训练适配

以下为 `msa_triton` 的实现选择，不能视为上游已提供的训练功能。

- 公共三阶段接口保持不变，仍返回 `indices[T,G,K]`。k2q 是 sparse attention 内部的离散辅助元数据。
- 训练 K/V 是 packed TND，没有分页 block table。CSR 采用一维 block-major 行号：
  `row = (cu_block_lens[sequence] + local_block) * G + group`。
- 本地行指针为 `[R*G+1]`，边记录按行组织 query/slot；它与上游 `[G,R+1]` 的物理布局不兼容，不作二进制接口复用。
- Forward 保持 query 所有权，沿该 query 的 selected blocks 做 online softmax，输出 FP32 LSE；backward 保存 FP32 O 和 max/denominator，避免从舍入后的 LSE 恢复概率。
- dQ 保持 query 所有权，并输出中心化 softmax backward 所需的统计；dK/dV 使用 k2q，让每个 key token/group 遍历其所属 KV block 行中的 queries 以及该 group 的主 query heads。初版按 32 个 queries 分块并补偿归约；`df31a1d` 候选改为逐 query 的 `[D]` TwoSum 累加以绕开旧树结构，仍独占写出 dK/dV，公共路径无需浮点 atomic。修改依据与验证边界见[调试复盘](debugging_retrospective.md)。
- CSR 只从本次 forward 的 indices 构建；backward 不重新 top-k，不对整数索引求梯度。

保留 query forward 是当前训练适配的取舍。若以后使用 KV-gather-Q forward，一个 query 的不同 KV 分片必须正确合并 softmax 统计和 partial output，不能直接相加各块独立 softmax 后的输出；其工作区和收益需在 A3 上实测。本轮并不把上游推理的可选 vendor 调用替换为训练实现。

Indexer KL 的学生/教师 token logits、梯度边界不因 k2q 改变。k2q 仅可用于改进同一稀疏边集合的计算组织，不替代 KL，也不会令 top-k 可微。

## 5. 验证边界

CSR 需要验证边集合精确往返、group/序列隔离、slot 映射、`-1` 过滤、空序列、空行及零 top-k。attention 需要固定相同 indices，对照 CPU FP64 的 O/LSE/dQ/dK/dV，继续使用原精度要求。

CPU Triton 解释器验证本地实现的索引和数学逻辑；上游 AscendC/vendor 算子不能因此被认定为在 CPU 验证通过。A3 编译、CSR 构建成本、KV 行负载不均、真实 backward 性能和显存占用仍需设备验收。最新执行结果以测试报告为准。
