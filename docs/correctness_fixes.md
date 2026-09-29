# 审查后的必要修复

日期：2026-09-28。本文记录对 [审查复核](code_review_response.md) 中已确认问题
的修复。三个公开接口、TND 布局、eager 的 BSND 迁移数学及精度阈值保持不变。

## 1. score 的胜者与并列最大值

原 FP32 点积在消减时可能改变最大值的胜者集合。两个 BF16 反例中，score
仍满足前向阈值，但 dIndexQ/dIndexK 的最大绝对误差为 0.5。单独保存原先的
argmax 或放宽分数误差都不能解决这个梯度问题。

[index_score.py](../triton/index_score.py) 现在用两个 FP32 分量保存点积的
高位和残差：先补偿乘法舍入，再以固定的相邻维度归约树合并。前向与反向
重算使用相同的维度归约次序，不随 query/key tile 大小改变。涉及补偿的
kernel 设置 `enable_fp_fusion=False`，避免融合改变依赖逐步舍入的表达式。

最大值先比较高位，再比较残差；比较发生在公开 FP32 score 舍入之前。
scale 为负时反转排序，为零时所有有效 token 并列。唯一胜者保存 token
位置；多个胜者保存数量，反向以相同的高位/残差比较恢复集合并均分梯度。
无效块显式写入 `-inf`，不贡献梯度。

因此，两个候选最终都舍入成 FP32 的 `1.0`，不意味着 block max 的 backward
必须把它们当作真正并列。下游 top-k 仍只读取公开 FP32 scores，其并列选择
规则不变。这与 top-k 的离散性、后续 KL 从 index Q/K 重算 token logits
的方案不冲突，也不需要改变接口。

原生 eager 继续保留上游 FP32 打分行为。极近分数或消减反例处，eager 可能
与修复后的 Triton 得到不同梯度；Triton 的正确性基准是同一输入的 CPU FP64
计算，不能为了复刻 eager 的数值缺陷而修改这个基准。

两个 FP32 分量不是精确 FP64 算术。极端动态范围、上溢、下溢和任意维度下，
仍不能保证与 FP64 完全相同的胜者集合。本次修复已复现的问题，未增加输入
幅度限制来排除反例，也不把有限测试外推为全输入域证明。

### 保存状态与内存

只有启用 autograd 且 index Q/K 至少一个需要梯度时，才分配 `max_hi`、
`max_lo` 和 `tie_info`，供 backward 使用；`torch.no_grad()` 路径不分配
这三份状态。BF16/FP16 的 FP32 staging 与之前相同。

例如 T=32768、G=4、max_seqlen=32768、block_size=128：

| 张量 | dtype | 大小 | 何时需要 |
|---|---|---:|---|
| 公开 scores | FP32 | 128 MiB | 始终 |
| 最大值高位 `max_hi` | FP32 | 128 MiB | score backward |
| 最大值残差 `max_lo` | FP32 | 128 MiB | score backward |
| 胜者位置或并列数 `tie_info` | int32 | 128 MiB | score backward |

上述四项同时存活时合计 512 MiB，不含输入 staging、上游梯度或反向工作区。
新增精度状态有明确代价；离散选块常用的 `no_grad` 路径仍只有公开 scores
这一份 block 工作区。

## 2. 地址运算在乘加前提升为 int64

[index_score.py](../triton/index_score.py) 与
[sparse_attention.py](../triton/sparse_attention.py) 中的 program id、token、
head、block、序列边界及 CSR 游标在地址乘加前提升为 int64。尤其包括从
int32 CSR 加载的 query token，以及 `offset + arange` 和最后一轮游标步进。
只把 program id 转成 int64，不能覆盖这些独立的溢出来源。

共享的 [_addressing.py](../triton/_addressing.py) 提供 TND 元素偏移和游标
加法；新增微型 kernel 用实际生产 helper 测试超过 `2^31-1` 的偏移，无需
分配巨大张量。T=262144、Hq=64、D=128 的最后一个有效偏移恰好是
`2^31-1`；T=262145 才出现超出 int32 的合法元素地址。

公开 cu/indices/CSR 存储仍为 int32，因此 token、边数和累计计数的容量
检查仍然保留。64 位地址不能替代 32 位 metadata 的容量约束；本次没有
通过更改排序算法删除这些保护。

## 3. attention 按实际需要计算梯度

backward 按 Q/K/V 独立的 `needs_input_grad` 分配输出，并用 constexpr
控制对应计算。不需要的输入梯度返回 `None`。

| 需要的梯度 | query 阶段 | k2q / KV 阶段 |
|---|---|---|
| 仅 dQ | 中心与概率质量精化、dQ | 跳过；不分配 STATS |
| 仅 dK 或 dK+dV | 中心与概率质量精化，写 STATS | 构建 CSR，计算所需梯度 |
| 仅 dV | 概率质量精化，写 STATS | 构建 CSR，仅计算 dV |
| dQ 加任意 KV 梯度 | 精化、dQ、写 STATS | 构建 CSR，计算所需梯度 |

即使 Q 不需要梯度，也不能跳过 KV backward 依赖的 query 统计。
V-only 保留概率质量修正，省略不需要的导数中心计算。私有 atomic 对照路径
同步支持这些组合；公共路径仍由 key/group 独占 dK/dV，不使用浮点 atomic。

## 4. 本次没有采纳的优化

- 删除中心精化：已有 D=128 反例，dQ 误差约 4.88e-4，超过原阈值。
- 用普通 tile sum 加跨 tile Kahan 替代补偿树：早期有限消融通过，后续新增
  12 项反例均失败，`2^-12` 梯度残差在 tile 内被丢弃；详见第 6 节。
- stable sort 后删除容量检查、按对象身份缓存 metadata：不能解决累计计数
  溢出或张量原地修改造成的缓存失效。
- Cube 分块、合并 program、稀疏 Indexer、8K–32K 性能优化：留待 NPU 测量
  后决定；本次不宣称已经提速。

## 5. 验证与设备边界

新增正式回归共 47 项：score 33 项、attention 11 项、地址边界 3 项。
覆盖 BF16/FP16/FP32、消减与真正并列、公开分数舍入相等、FP32 乘法残差、
负/零 scale、空序列 packing、no_grad、7 种 Q/K/V 梯度组合、输入未被覆盖、
GQA=16 和 int32 边界外地址。原有 eager、top-k、k2q 和组合验证继续保留。
比较仍用同一输入的 FP64 oracle、原公开 dtype 和 `atol=rtol=1e-4`。

完整回归 **138 passed，0 failed，0 skipped**，执行记录见
[validation.md](validation.md)。独立诊断
`python -m msa_triton.tests.probe_score_rounding` 保留两个原始反例；修复后
score、dQ、dK 的最大绝对误差均为 0。

原消融脚本已适配独立梯度开关，并在临时源码副本上复查 D=128 用例：
`tile-sum-kahan` 为 1 passed，`no-center` 为预期的 1 failed，仍在
dQ[1,1,96] 产生 0.00048828125 的误差。后者是验证删除精化的风险，不计入
生产实现的正式测试；生产 kernel 没有应用这两种消融。

另增加 [probe_offline_compile.py](../tests/probe_offline_compile.py)，在本机
上游 Triton 上离线编译 CUDA sm80 的 31 个变体。该检查发现并修复了新增
梯度条件的连续三个 `or` 语法问题；改成明确括号分组后全部通过。这弥补了
CPU interpreter 不检查所有 Triton 语言限制的不足，不构成 NPU 验收。

本机没有 GPU/NPU。CPU interpreter 不能验证 triton-ascend 编译、bitcast /
gather 的目标布局、补偿表达式是否被后端保留、并发或性能。迁移到 NPU 后须
重新执行完整精度门禁，再测长序列内存与耗时；不应根据本机时间推算加速比。

## 6. NPU 结构对照后的 attention 累加候选

`6d47514` 的六项 NPU 对照为 5 failed / 1 passed：独立树编译失败，原
全梯度、QK、QV、拆分 DK/DV launch 均未通过；tile-sum + Kahan 只通过该
D=128 样例。普通 tile sum 会先丢掉 tile 内消减残差，跨 tile Kahan 无法
恢复。新增 12 项正式回归对齐 CPU FP64，覆盖 dK/dV × BF16/FP16/FP32 ×
仅目标梯度/全部梯度；参考残差 `0.000244140625` 不能被算为零，阈值仍为
`atol=rtol=1e-4`。旧树 12 项通过，tile-sum 消融 12 项失败。

正式 KV 候选改为逐 CSR query、逐同组 head 累加单个 FP32 `[D]` 梯度向量，
用 TwoSum 保留高低分量，最后写回。该 kernel 设置 `enable_fp_fusion=False`
以保留补偿表达式；center/mass 精化和独立梯度开关保持。改变了浮点加法
顺序，因此原长序列 BF16 舍入用例仍是必需的回归，而非假定结果逐位不变。

4 项生产 helper 测试改为单 FP32 贡献流，对照独立 `math.fsum`；所选样例
hi+lo 使用 `1e-12` 门禁，并检查非零 low。这不是对任意输入范围、任意
双分量输入的无损累加保证。历史树固定在显式诊断文件中，未从历史证据中
抹除，也不会混进新候选的正式 NPU 测试。

更小的同时存活张量不保证 Ascend 编译成功，也不能证明原故障是容量不足。
串行 query 的吞吐可能下降，本轮目标是可编译与正确性候选，性能待实测。
score 的非法 GM 地址问题仍未解决。当前任务见
[48 项 attention 复测](npu_attention_streaming_retest.md)。
