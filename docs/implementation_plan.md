# MSA 独立算子实施计划

## 范围与实现组织

目标为 Ascend 910C/A3、BF16 packed self-attention。所有新增实现位于
`msa_triton`，不修改 VeOmni，不提供 backend 参数或隐式 eager fallback。
`eager/` 保留 Transformers BSND 数学，TND wrapper 在内部 unpack/pack；
`triton/` 直接处理 TND。公开入口的签名、布局和序列边界相同，后续交给
VeOmni 选择实现。主 projection、norm、RoPE 和通信位于这些算子之外。

使用 `conda activate veomni`。本机无 GPU/NPU，先运行 CPU FP64 和 Triton
解释器验证，再由 A3 实机验证编译、并发归约、实际精度与性能。

## 算子接口

```python
scores = m3_index_score(index_q, index_k, cu_seqlens, max_seqlen,
                       block_size=128, scale=1.0)
indices = m3_topk(scores, cu_seqlens, block_size=128,
                 topk_blocks=16, local_blocks=1)
out = m3_sparse_attention(q, k, v, indices, cu_seqlens, max_seqlen,
                          block_size=128, scale=None, return_lse=False)
```

* index Q/K 为 `[T,G,Di]` / `[T,1,Di]`，主 Q/K/V 为 `[T,Hq,D]` /
  `[T,G,D]` / `[T,G,D]`，要求 `Hq % G == 0`。分组与主 KV heads 一一对应。
* 元数据 `cu_seqlens` 为同设备 int32；允许空序列，起点为 0、终点为 T。
  每条序列从自身起点划块，causal 判断使用序列内 token 偏移。
* score 输出 FP32 `[T,G,ceil(max_seqlen/block_size)]`。每组独立计算，
  点积后仅对 causal block tokens 取 max。未来/无效块为 `-inf`。
  默认 scale=1 保留上游 Indexer 行为；需要论文的缩放分数时显式传入。
* score 提供 dIndexQ/dIndexK；并列最大值按 `torch.amax` 均分梯度。
  Triton 以补偿点积的高位/残差判定胜者，再舍入为公开 FP32 score；
  公开分数舍入相等不自动视为 block 内真正并列。
  无效块不贡献梯度。top-k 不可微，只验证离散行为，不伪造 backward。
* top-k 输出 int32 `[T,G,K]`，local blocks 占用 K 名额、不修改原始 scores。
  有效项不重复、左对齐，空位为 -1；并列候选允许不同的合法选择。
* attention 按 group indices 访问 KV，继续施加 token causal mask。
  输出与 Q 同 dtype。可选 FP32 自然对数 LSE `[T,Hq]` 不可微。
  scale 默认 `D**-0.5`，dropout=0。全无效行输出 0、LSE=-inf、梯度 0。
* attention 提供 dQ/dK/dV，反向复用 indices，不重跑选择。

## 内部实现与 KL 边界

Eager 迁移锁定上游源码，内部允许 BHSD 转置；wrapper 对外保持 TND。
稠密 padding/causal/group mask 是 eager attention 的内部细节，不暴露到
公共接口，也不假设普通 FlashAttention varlen 能接收任意稠密 mask。

Triton score 融合补偿点积、causal 和块最大归约，保留 block-score 工作区。
需要 backward 时另存 max 高位、残差、胜者位置/并列数；no_grad 不分配
这些状态。维度归约次序在前反向一致，补偿表达式禁用浮点融合。
初版 top-k 使用设备上的 `torch.topk`。attention 流式访问稀疏 blocks，
前向输出 FP32 LSE，并保存 FP32 O、max/denominator；反向重算概率。
dQ 使用 query-owned 计算，dK/dV 经 k2q CSR 由 key/group 独占归约，
采用 FP32 补偿求和，公共路径不依赖浮点 atomic。CSR 和反向统计仅在
backward 构建，额外空间为 O(T*G*K + T*Hq + T*G*D)。
Q/K/V 梯度按需计算：Q-only 跳过 CSR，KV-only 仍保留依赖的 query 统计，
V-only 保留概率质量精化。Triton 地址在乘加前提升为 int64，int32 metadata
的容量检查独立保留。数值修复的细节与内存代价见 `correctness_fixes.md`。
首版采用可移植、易验证的 kernel，不承诺在 A3 上已经获得速度提升。

参考 vLLM-Ascend 的反向邻接思想，具体来源和差异见 `vllm_ascend_k2q.md`。
上游使用推理专用 AscendC/vendor 算子；本项目实现独立的训练 backward，
不直接调用其无梯度推理接口。保留私有 atomic reference 只用于算法回归。

Indexer KL 暂不实现独立 kernel。官方 KL 的学生是选中 tokens 的分布，
不能用 block max scores 替代。后续 KL 直接从 index Q/K 重算 token logits，
教师使用 detached 主 Q/K，Indexer 输入 hidden states detach。即使本轮
实现了 score backward，KL 仍无需经过 block max 或 top-k。

## 验证标准

使用同一份 BF16 输入，转 CPU FP64 完成参考前向和反向，再转为公开结果的
dtype：attention 输出/输入梯度为 BF16，scores/LSE 为 FP32。反向使用同一份
上游梯度。严格比较 `atol=1e-4, rtol=1e-4`，不得自动放宽。
无穷位置单独检查，indices 验证合法性，无并列时比较选择集合。

FP64 oracle 独立实现，不能复用 eager 中强制 FP32/BF16 的降精度操作。
原生 BF16 eager 的中间舍入与 FP64 oracle 有差异；两种对照分别报告：
TND eager adapter 应与原生 BSND 路径一致；Triton 的严格数值门禁对齐 FP64。
Eager 与 Triton 的低精度差异必须给出统计，不通过改写原生 eager 来掩盖。

测试包含：unpack/pack 值和梯度往返；score 前反向及 max ties；top-k 的 local、
padding、去重；固定 indices 的 attention 前反向；三阶段组合及小模块参数梯度。
覆盖 1/127/128/129/257 等边界长度、变长 packing、不同 group/GQA/head_dim、
尾块、无有效选择及跨序列隔离。8K/16K/32K 留作 A3 长序列与性能验收。

实施顺序：文档 -> 环境 -> eager/layout/FP64 -> 独立 Triton 前反向 ->
CPU interpreter -> A3 编译、数值和性能。解释器执行不能验证设备并发或性能；
不支持的测试明确跳过并记录，不将 FP32 模拟冒充 BF16 设备测试。
