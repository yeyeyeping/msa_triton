# MSA Triton 实现与 NPU 调试复盘

整理日期：2026-09-29。代码基线：`df31a1d12cdbb737022fe6bcb572d4145649a32d`，
分支 `fix/npu-portability`。

本文将接口约定、CPU 数值修复、NPU 故障定位、失败尝试和新候选的形成过程
串成一条可追溯的记录。实验数字来自仓库验证记录及用户回传报告；本机没有
GPU/NPU，远端 NPU 结果由用户机器上的执行代理采集。

## 1. 本文结论与验证范围

截至本文整理时，明确记录如下：

| 对象 | 已有证据 | 状态 |
|---|---|---|
| 三阶段接口、eager TND 适配、CPU 数值 | 独立 FP64 oracle、原生 BSND 对照、组合测试 | 已完成本机验证 |
| score 数值胜者及梯度、地址宽度、按需反向 | 反例复现后加入正式回归 | 已完成对应 CPU 修复与验证 |
| attention 原 32 行补偿树 | NPU 多次 KV 编译失败，最后定位到 PlanMemory pass | 原路径失败已确认，具体编译器错误 op 未定位 |
| attention 逐 query 补偿实现 `df31a1d` | CPU 全套 157 passed；41 个 CUDA 离线编译变体通过；用户确认 NPU 48 项全部通过 | 本轮 attention 实机门禁通过 |
| score NPU 非法 GM 访问 | 边界钳制候选仍失败，blocking/non-blocking 均复现 | 尚未解决 |
| 全 MSA NPU、长序列性能、VeOmni 接入 | 无完整通过记录 | 未完成 |

不能把旧版本六项消融中的一个 `passed`、离线编译成功或一次结果采集成功
解释为整个算子验收通过。本次整理过程中，用户明确确认了新提交 `df31a1d`
的“48 项 NPU 测试全部通过”，因此 attention 故障在本轮测试范围内完成
修复闭环。该确认未附新的耗时、完整日志目录、退出码和环境哈希，本文不
补写这些数据，也不把它当成本机执行。score、性能和模型接入仍须分别验收。

## 2. 起点：先约定数学语义，再实现算子

最初考虑将 MSA 拆成选块和 sparse attention 两部分。但把打分和 top-k
合在一个不返回 score 的接口中，会隐藏每个 group 的独立分数，也不利于
检查 Indexer 的训练边界。因此最终确定三个公开阶段：

```text
Index Q/K，TND
    │
    ▼
m3_index_score ── 每个 query、group、block 的独立分数
    │
    ▼
m3_topk ──────── 序列内 block indices，离散且不可微
    │
    ▼
m3_sparse_attention ◄── 主 Q/K/V，TND
    │
    ▼
输出及 dQ/dK/dV
```

三项约束一直保留到本轮候选：

1. **接口与后端选择分离。** 不增加 `backend="eager"` 或隐式 fallback。
   `eager/`、`triton/` 各自实现相同的 TND 接口，后续由 VeOmni 选择。
2. **兼容路径与计算布局分离。** eager 迁移 Transformers 原生 BSND 数学，
   内部通过可微 unpack/pack 适配 TND；Triton 直接消费 packed TND。
   eager 的稠密 mask 属于内部实现，不能假设普通 FA varlen 接口可直接接收
   任意 block mask。单算子对齐不等于已完成整模型接入。
3. **可微分数与 KL 语义分离。** score 提供 backward，top-k 不伪造梯度。
   官方 Indexer KL 的学生分布来自选中 tokens 的 logits，不能用 block max
   分数代替；后续 KL 从 index Q/K 重算，教师和 hidden-state 梯度边界按
   训练约定处理。返回 score 不意味着 KL 必须经过 block max/top-k。

完整接口见 [实施计划](implementation_plan.md)，KL 见
[训练流程](msa_training_flow.md)，迁移源版本及舍入行为见
[eager 来源记录](../eager/PROVENANCE.md)。

## 3. 建立两种不同用途的参考实现

原生 eager 保留低精度中间舍入，不能同时作为严格 FP64 oracle。验证因此
分为两条：

| 对照 | 解决的问题 |
|---|---|
| TND eager adapter 对原生 BSND | pack/unpack、mask、group 与序列边界是否迁移正确 |
| Triton 对独立 CPU FP64 oracle | 同一输入与上游梯度下，O/LSE/score 和各输入梯度是否准确 |

严格测试先生成并量化同一份输入，oracle 在 CPU FP64 完成中间计算和
autograd，最后转为公开输出 dtype。沿用 `atol=1e-4, rtol=1e-4`；这表示
绝对加相对容差，不能改写成所有元素纯绝对误差均小于 `1e-4`。`-inf` 位置
单独比较。top-k、CSR 检查离散规则和边集合，不做虚假的梯度比较。

原生 BF16 eager 与 FP64 的差异单独统计，没有通过改写 eager、跳过失败项
或放宽 Triton 门槛来消除差异。CPU 使用 `TRITON_INTERPRET=1` 执行实际
kernel 逻辑，但不能验证 Ascend 编译、设备并发、实际舍入或吞吐。

## 4. CPU 阶段：从“已有测试通过”到反例驱动修复

初版 CPU 测试为 91 passed。审查时没有直接照搬所有优化建议，而是区分
正确性缺陷、可验证的优化假设和未经测量的性能判断。

### 4.1 score 前向误差小，不代表 max backward 正确

一个 T=2、G=1、Di=3 的 BF16 反例中：

```text
q[0] = q[1] = [1, 2^-12, 1]
k[0] = [1, 2^-12, -1]
k[1] = [0, 0, 0]
FP64 logits = [2^-24, 0]
原 FP32 logits = [0, 0]
```

前向 score 差值只有约 `5.96e-8`，通过容差；但唯一胜者被变成并列胜者，
dIndexQ/dIndexK 最大绝对误差达到 `0.5`。另一个反例则把真实并列变成
唯一胜者。只保存原 argmax/tie 信息不能补救前向已经选错的集合。

修复使用 FP32 高位与残差保存补偿点积，前反向保持相同维度归约次序，
在公开 FP32 score 舍入前比较胜者。处理负 scale、零 scale、无效块及
真实并列的均分梯度；需要 backward 时才保存相应状态。两个原始反例
修复后的 score、dQ、dK 最大绝对误差均为零。

代价是额外状态和计算；双 FP32 分量也不等价于任意范围的精确 FP64。
数值修复没有同时证明该 kernel 可在 NPU 正确运行。

### 4.2 地址宽度必须覆盖整条计算链

地址乘加前提升为 int64，覆盖 program id、从 int32 CSR 读出的 token、
`offset + arange` 和循环递增。中间值溢出以后再 cast 无法修复地址。
微型 kernel 用实际生产 helper 验证超过 `2^31-1` 的偏移，无需分配巨型
张量。int32 metadata 的容量检查仍保留，不能用 int64 地址替代它。

### 4.3 按需反向不能删除统计依赖

按 Q/K/V 的实际梯度需求分配和计算：Q-only 跳过 CSR；KV-only 仍生成
KV backward 所需的 query 统计；V-only 保留概率质量修正。不需要的
梯度返回 `None`。新增用例覆盖全部七种非空梯度组合。

删除 center/mass 精化的消融在短 D=128 用例就失败：dQ 的 BF16 值从
参考 `0.07373046875` 变成 `0.07421875`，误差 `0.00048828125`。
真值接近 BF16 舍入中点，极小 FP32 偏差也可能变成超过门槛的末端误差。
因此后续编译排查始终保留这部分数学。

这些修复及新增回归使 CPU 全套从 91 项增至 138 项。上游 CUDA sm80
离线编译也发现解释器未暴露的条件表达式语法限制，修正后 31 个变体通过。
详见 [审查复核](code_review_response.md) 和 [必要修复记录](correctness_fixes.md)。

## 5. k2q：改变稀疏边的访问方向，不交换注意力数学角色

参考 vLLM Ascend 的 k2q 思路，将“query 选择哪些 blocks”反转为“block
被哪些 queries 选择”。softmax 仍沿每个 query 可见的 keys 归一化，
反向仍固定 forward 的 indices，不能交换 Q/K 的数学角色。

本项目 forward 和 dQ 由 query 负责；dK/dV 由每个 key token/group 遍历
CSR 中的关联 queries，独占写出梯度，避免公共路径的浮点 atomic 累加。
上游参考是推理接口，包含 AscendC/vendor 算子；没有把其无梯度接口直接
当作训练 backward。布局与适配差异见 [k2q 参考记录](vllm_ascend_k2q.md)。

早期 KV 实现每次处理 32 个 queries，对 `[32,D]` 梯度做五层相邻行
TwoSum 补偿树，再合入跨 tile 高低分量。它在 CPU 的数学回归中通过，
但成为后续 NPU 编译结构排查的重点。

## 6. 第一轮 NPU：先把两个故障分开

用户回传的首轮结果为 **138 项，90 passed / 48 failed，419.39 秒**。
原报告没有设备端 Git SHA，不能事后给它补写一个确定提交号。工作在
Ascend910_9382、物理卡 15／逻辑卡 0 上执行，使用 sudo、`veomni_m3`、
Python 3.12、torch/torch_npu 2.10 和 triton-ascend 3.2.2。

| 故障 | 最小关键配置与阶段 | 观测 |
|---|---|---|
| attention | lengths=[5,3]，Hq=4，Hkv=2，D=128，block=4，topk=2；backward KV 编译 | forward 通过；bishengir SIGABRT，returncode=-6 |
| score | lengths=[0,3,0,2,0]，G=2，D=7，block=4；forward 后同步 | 507035、非法 scalar GM 地址／地址超过 48 位 |

attention 文件为 31 passed / 1 failed；score 文件为 2 passed / 47 failed。
score 首个设备故障之后，普通 tensor 拷贝也失败，说明有上下文污染和
级联错误。**47 个 score 失败不是 47 个已定位的独立缺陷**；总计 90 个
通过还包含固定在 CPU 执行的 eager 测试。

score 最初呈非确定性：相同 shape 可能成功也可能失败；单独 vector add
和 gather 正常，dtype/是否求导未形成稳定关联。这些结果有助于缩小范围，
但不能证明组合 kernel 的 gather、参数传递或地址降低正确。

## 7. 第一轮兼容性候选：改动有依据，结果仍失败

`ebf85cd` 做了两组保守修改，`8649b73` 增加完整远端执行协议：

| 对象 | 候选改动 | 保留的语义 |
|---|---|---|
| score | 直接归约活动序列 begin/end，去掉间接 CU 加载；构造指针前钳制无效 lane | mask、补偿点积、scale、ties、int64、公开 dtype |
| attention | 相邻行 gather 改为 reshape → permute → split | 五层配对、TwoSum、center/mass、梯度开关 |

本机 **145 passed，219.27 秒**，41 个 CUDA 离线编译变体通过。然而远端
明确在 `8649b7377a00e2a512cd1b333b2385c7e274a2ed`、核心哈希匹配时复测：

| 阶段 | 独立进程重复结果 |
|---|---|
| attention | 3/3 失败；KV 编译新增明确 fatal：`LLVM ERROR: PlanMemory Traverse IR Failed!` |
| score blocking | 10/10 失败；非法 GM 地址 |
| score non-blocking | 10/10 失败；非法 GM 地址 |

B/C/D 大范围阶段按协议停止。这里是 23 次重复失败，不是 145 项全量新结果。
两个 score 组各覆盖两个节点的重复，不能把整组十次都算到首个节点。

这一轮否定了“上述边界简化或 gather 写法替换已经解决设备故障”。scalar
GM 字样不唯一指向 CU 标量 load；被 mask 掉的地址表达式越界也不能单独
证明违反 Triton load 契约。blocking 是定位手段，不是修复。

## 8. 从大包日志转向可核验的短证据

受远端文本传输限制，流程从“交付全部日志与 IR”收敛为“远端保存原始
文件，本机接收源码身份、阶段、退出码、首个错误与必要摘要”：

1. 记录 Git SHA、dirty 状态、核心文件哈希、实际导入路径、设备及版本。
   从仓库父目录启动，避免本地 `triton/` 遮蔽第三方包。
2. 区分 Python 同步位置与真正首次失败阶段；每个设备故障复现使用新进程。
   新进程隔离上下文，不保证恢复卡级故障。skip、timeout、crash 不计通过。
3. 将失败 specialization、异常链、TTIR、ttadapter、cache 元数据和二进制
   对应起来。缓存中出现 `.npubin` 不等于失败的 KV kernel 已成功编译。
4. 保存完整日志于新的结果目录，短回传保留 `not_run`，不补猜未执行结果。

后续报告补齐 CANN 9.1.0、驱动 26.0.rc1、固件 9.0.0.0.205；Triton
runtime 为 3.2.0、distribution 为 3.5.0。记录这个差异，但没有仅凭版本
字符串不一致就要求重装。编译器二进制 SHA256 固定为：

```text
89655a56941efe9a184e4d5dfccb783ad88458707827ff6fdd146e7bd2d4af5c
```

### 8.1 同 IR、同编译器，只改变 multi-buffer

离线重放先要求 baseline 重现原失败，再只把
`--enable-auto-multi-buffer=True` 改为 `False`，不运行生成的设备代码。
用户回传两者均为 returncode=-6、同一个 PlanMemory fatal。

结论是**单独关闭该选项不能绕过故障**，不是“所有内存相关问题已排除”，
也不是“该选项肯定未生效”。重放脚本见
[probe_attention_compile_replay.py](../tests/probe_attention_compile_replay.py)。

### 8.2 before-pass IR 将范围缩到 PlanMemory

用户确认 `IR_MATCH=confirmed`；添加 before-pass dump 后重现原失败，
共 235 个 pass 标题，最后五个为：

```text
canonicalize-ext → memref-dse → hivm-inline-load-copy
→ hivm-mark-multi-buffer → hivm-plan-memory
```

最后保存的是 2110 行 module，唯一函数 `_backward_kv_kernel` 有 18 个
参数，AIV core；除 fatal/backtrace 外，没有 op 级 error/note/warning。
这确认失败 pass，仍不能定位具体操作。

当时特别纠正了三种容易误导后续修改的推断：

- 1953 次 `address_space<ub>` 出现不等于 1953 个 alloc，也不等于峰值
  存活字节数。类型还会出现在 view、load/store 和运算参数中。
- 最后看到的 `%alloc_111`、128 个 FP32、vadd/store 只说明某个缓冲为
  512 字节，不能把日志末尾当作崩溃操作或证明整个 kernel UB 超限。
- `multi_buffer=2` 来自恢复原选项的 pass-dump baseline，不能据此判断
  前一轮关闭开关的实验无效；`autoblockify.subloop` 也不是根因证据。

历史源码核对提示 PlanMemory 的 fatal 可能来自遍历／本地操作处理失败，
并非特定的容量不足诊断；公开源码未确认与实际二进制同版本，故仅作线索。
完整判断依据保存在 [编译取证记录](npu_attention_compile_probe.md)。

## 9. 六项结构消融：找到可继续验证的方向

在 `6d4751489a0fe0e1140f42dbd5571b07537dabbe` 上，用户回传：

| 用例 | 唯一关注的变化 | 结果／最后阶段 |
|---|---|---|
| tree-d128 | 独立执行 32×128 补偿树 | compile_error／tree launch |
| qkv-d128 | 原完整梯度路径 | compile_plan_memory／KV launch |
| qk-d128 | 保留 dQ，仅请求 dK | compile_plan_memory／KV launch |
| qv-d128 | 保留 dQ，仅请求 dV | compile_plan_memory／KV launch |
| split-kv-d128 | 共用原 stats，DK/DV 分两次 launch | compile_plan_memory／第一次 DK launch |
| tile-sum-kahan-d128 | 保留精化，用 tile sum 加跨 tile Kahan | passed／complete |

总计 **5 failed / 1 passed，not_run=0**。这说明只减少输出或拆开 launch
不足以解决，而绕开原行补偿树值得继续验证。独立树摘要只写了
`compile_error`，不能补写成已经确认相同 PlanMemory 根因。

此报告后来被再次粘贴。我们用 SHA 和相同结果目录识别它仍是旧实验，
没有将其当成新候选 `df31a1d` 的复测。**源码版本是证据的一部分。**

## 10. 为什么没有直接采用唯一通过的 tile-sum + Kahan

该消融在已有 CPU 形状和 257-token BF16 舍入回归中也曾通过，但覆盖仍
不足。新增针对 tile 内消减的测试揭示：普通 FP32 tile sum 先丢掉小项，
后面的跨 tile Kahan 无法恢复已经丢失的信息。

| 梯度 | 某个 key 的数学贡献示例 | 正确残差 |
|---|---|---:|
| dV | `[4096, 2^-12, -4096]` | `2^-12` |
| dK | `[0, 6144, 2^-12, -6144]` | `2^-12` |

残差 `2^-12 = 0.000244140625` 被该消融算成零时，超过现有精度门禁。
构造覆盖 dK/dV × BF16/FP16/FP32 × 仅目标梯度/全部梯度，共 12 项：
**历史补偿树 12 passed，tile-sum + Kahan 12 failed**。

这一步把“编译结构可能更简单”与“训练数值仍然正确”重新放到同一标准上。
没有将通过的单例升级为生产实现，也没有用较宽容差接受残差丢失。反例已
加入 [正式 attention 测试](../tests/test_triton_attention.py)，历史消融可由
[ablation 工具](../tests/probe_attention_ablation.py) 的 `--case cancellation`
复现；这些预期失败不计入生产测试的通过数。

## 11. 最终采用的修复：保留补偿数学，改变计算结构

`df31a1d` 把 KV backward 的 32 行树改为逐 CSR query、逐同组 head 的
`[D]` 向量累加：

```text
每个 key token/group 独占自己的 dK/dV
    读取对应 CSR 行
    对每个有效 query：
        对该 group 的各 query head：
            重算概率，保留 center/mass 精化
            计算一个 FP32 dK/dV 贡献向量
            将贡献用 TwoSum 合入 (high, low)
    写回 high + low
```

关键选择与代价：

- 每次贡献进入累加器就保留舍入残差，不先做会丢失小项的普通 tile sum。
- KV kernel 设置 `enable_fp_fusion=False`，保留 TwoSum 所依赖的逐步舍入。
- 保留 causal、packing、group、独立梯度、int64 地址及公共 TND 接口。
- 累加顺序发生变化，必须重跑 BF16 舍入边界及完整回归，不能假定逐位相同。
- 同时存活的张量更小，但减少了 query 并行度；可能牺牲吞吐。它是兼容性
  与正确性修复，不是已完成的 NPU 性能优化，也不证明原故障就是 UB 容量。

4 个 helper 用例改为实际生产的单 FP32 贡献流：输入 `b_lo=0.0`，独立
`math.fsum` 作为基准，所选样例 hi+lo 检查 `1e-12`，并验证输出 low 非零。
这里也经历一次测试契约修正：最初向 helper 注入任意非零低分量并要求
`1e-12`，暴露了约 `1.49e-8` 的二次舍入；生产调用并不使用这种输入。
最终按真实输入契约测试，没有降低公开 attention 的 `1e-4` 门槛，也不
宣称双分量累加器在任意输入范围都无损。

旧树及三个关键函数固定在
[_attention_tree_reference.py](../tests/_attention_tree_reference.py)，核对与
`6d47514` 的函数 AST 一致，仅供显式诊断。生产路径移除了树 helper；
旧失败证据得到保留，没有混入新候选的正式验收。

## 12. 验证里程碑与下一步门禁

| 阶段 | 本机 CPU 全套 | 其他验证 | 设备结论 |
|---|---|---|---|
| 初版数学实现 | 91 passed | 原生差异独立统计 | 无设备结论 |
| 审查后的数值修复，归入 `0dc8eee` | 138 passed | 31 个 CUDA 离线编译变体 | 随后首轮 NPU 90/48，设备 SHA 未记录 |
| 第一轮 portability 候选 `ebf85cd`／协议 `8649b73` | 145 passed | 41 个离线变体 | 远端 A 阶段全部重复失败，后续未运行 |
| 结构取证至 `6d47514` | 不作为新的全量验收 | 同 IR 重放、pass trace、六项消融 | 六项 5 failed / 1 passed |
| 逐 query 实现 `df31a1d` | 157 passed，0 failed/skip，439.58 秒 | 41 个离线变体；最终 helper 另复跑 4 passed；CPU 七项诊断通过 | 用户确认正式 NPU 48 项全部通过 |

91、138 等早期数字来自验证过程记录，不代表每一阶段都有单独 Git 提交。
CUDA 离线编译使用上游 Triton 的 sm80 target，不执行 GPU kernel，不能
替代 triton-ascend 编译。157 项仍有 795 条上游解释器弃用 warning，未屏蔽。
详细命令和统计见 [验证记录](validation.md)。

本轮已通过的远端门禁是 **4 个生产 helper + 44 个 attention 用例，共 48 项**，
协议要求固定候选文件哈希，清理解释器／kernel override／pytest 附加参数
影响，从父目录执行，使用新 cache/IR 目录并在首个失败处停止。回传实际
SHA、统计、首个失败 node、阶段和错误首行，完整日志保留远端。命令见
[NPU 复测协议与结果](npu_attention_streaming_retest.md)。

用户于 2026-09-29 明确确认 48 项全部通过，验证范围包括原 D=128 编译
失败节点、不同长度/GQA/独立梯度、BF16 长序列舍入及新增消减回归。由此
可以确认：新的计算结构已在这组实机测试中同时满足编译、执行和数值要求。
它绕开了旧路径的故障，但没有定位或修复 bishengir 内部的具体失败 op。

后续仍须分别完成 score 首发错误定位、同进程及正常异步重复、全量 NPU
前反向验收、8K/16K/32K 性能与内存测试，再推进 VeOmni／整模型接入。
不能用这一轮 attention 通过替代其余工作。旧失败报告作为版本历史保留，
新通过结论只归属于此次明确确认的实现与测试范围。

## 13. 可复用的调试方法

1. **先固定契约和 oracle。** 区分 native 兼容性与高精度正确性；离散分支
   的选择误差可能远大于前向数值误差，必须测试梯度。
2. **先找首次错误，再看失败总数。** 编译崩溃、运行时非法访问和数值断言
   分开处理；上下文污染后的失败不应驱动新的源码猜测。
3. **每个实验只回答一个问题。** 同 IR 单开关、保留精化的归约消融、共用
   stats 的拆分 launch，使结果能够约束下一步，而非同时改变所有因素。
4. **把观察与解释分开记录。** pass 名、UB 类型次数、日志末尾、标量访问
   字样都是线索；没有证据时不升级为具体 op、容量不足或已知后端 bug。
5. **简单实现仍要过对抗数值测试。** 一个更易编译的归约可能丢失残差；
   跨 tile 补偿不能自动补救 tile 内精度损失。
6. **CPU、离线编译、设备执行、性能分层验收。** 任何一层的通过都不能
   替代下一层。把真实反例加入回归，保留旧实现作显式对照。
7. **用短摘要降低协作成本，用原文件保留可追溯性。** SHA、哈希、环境、
   阶段与退出码比重复粘贴完整 backtrace 更有效；重复旧报告先核对版本。

## 14. 提交与材料索引

| 提交 | 内容 |
|---|---|
| `0dc8eee` | 初始入库版本，包含三阶段实现及审查后的数值修复 |
| `ebf85cd` | score 寻址简化、attention 树 lowering 候选 |
| `8649b73` | NPU 分阶段复测与证据采集协议 |
| `3df5201` | 记录候选复测失败与已有产物交付要求 |
| `934d238` | 同 IR、同编译器的受限离线重放 |
| `1de29e1` | before-pass IR 采集与隔离结构对照 |
| `2273060` | 解释 IR 尾部，启用短结果结构对照 |
| `6d47514` | 记录 PlanMemory pass，纠正 UB 分配数量推断 |
| `df31a1d` | 逐 query 补偿候选、消减回归、历史树对照和 48 项复测协议 |

首轮原始报告摘要见 [npu_validation_20260928.md](npu_validation_20260928.md)，
候选失败及环境细节见 [npu_retest_20260929.md](npu_retest_20260929.md)。
诊断入口另包括 [隔离 runner](../tests/probe_npu_isolated.py)、
[score 顺序复现](../tests/probe_score_sequence.py)、
[attention 结构对照](../tests/probe_attention_scope.py) 和
[离线 CUDA 编译](../tests/probe_offline_compile.py)。历史协议保留用于解释
历史结果；执行新测试应以对应候选的专门复测文档为准。
