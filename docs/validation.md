# 验证层次、精度和设备限制

> 最新状态：2026-09-29 用户明确确认 `df31a1d` 的 48 项 attention NPU
> 测试全部通过。score 非法 GM 地址仍未解决，全 MSA 与性能尚未验收。
> 下文旧候选失败记录按版本保留。完整过程见[调试复盘](debugging_retrospective.md)。

## 数值门禁

输入先生成并量化为 BF16，两条路径共享同一份量化输入。oracle 转为 CPU
FP64，全部中间计算及 autograd 均保持 FP64，最后将结果转为公开 dtype。
使用 `torch.testing.assert_close(atol=1e-4, rtol=1e-4)`。主输出及输入梯度
转 BF16；block scores 和 LSE 转 FP32。`-inf` 位置必须一致。

最大值在并列处不可微，采用 `torch.amax` 的均分梯度约定；有限差分只能在
没有 argmax 边界的输入上使用。top-k 只返回 int32，测试集合而不伪造梯度。
attention backward 固定同一 indices，单独对齐 dQ/dK/dV。

native eager 保留上游低精度中间舍入；它不等于 FP64 oracle。测试分别验证：
1. TND adapter 对原生 BSND 的输出/梯度一致性；
2. Triton 对 FP64 oracle 的严格前向/反向精度；
3. 多形状下 native BF16 eager 与 Triton 的误差统计。
统计项不会被误标成严格门禁通过，也不会用于放宽第 2 项阈值。

## 本机与 A3 的边界

本机无 GPU/NPU。CPU 上游 Triton 解释器只能验证所执行程序的数学和索引。
BF16/FP16 指针被精确扩展为 FP32 后传入 kernel，适用于同一便携 kernel 路径。
这一线性 staging 内存和复制开销是首版实现的已知代价。

NPU 尚须验证：
* triton-ascend 编译和 CANN 版本兼容；
* exp/log、归约及 BF16 舍入后的严格数值；
* k2q CSR 的 `repeat_interleave`、int64 `argsort`、int32
  `scatter_add/cumsum` 在 NPU 上的支持，以及其构建成本；
* dK/dV 的 key-owned 补偿归约，编译器对 FP32 roundoff 表达式的保留；
* score 补偿点积的 bitcast/gather、固定维度归约布局及资源限制；检查
  `enable_fp_fusion=False` 与补偿表达式在目标后端生效，前反向 ties 一致；
* 8K/16K/32K、G=4、Hq=64、D=128 的耗时与峰值内存。

首版 kernel 使用 FP32 向量归约而非经过 A3 Cube 调优的矩阵分块。Indexer
仍遍历全部 causal keys，dK 用所有相关 queries 重算，不能宣称已实现稀疏
Indexer 或已获得加速。主 attention 仅访问 indices 指定的 blocks。
Python 校验会读取少量 metadata 到 CPU，可能在 NPU 上同步；图捕获和调度
优化属于后续实机工作，不在本次正确性验收中冒充完成。
例如 32K、G=4、block_size=128 时，FP32 scores、max_hi、max_lo 与
int32 tie_info 各占 128 MiB；后三项仅在 score 需要 backward 时分配。
四项同时存活为 512 MiB，另有 FP32 staging 和反向工作区；避免 token²
logits 不等于没有长序列内存成本。详见 [修复记录](correctness_fixes.md)。

公共 attention backward 经 k2q CSR 反转稀疏边，key/group 独占输出 dK/dV，
不进行浮点 atomic 累加。旧的 query-owned atomic 路径作为私有回归参考保留，
不暴露后端参数或自动 fallback。CSR 反转不是交换 softmax 的归一化方向。

## 测试命令

```bash
conda activate veomni
python -m pytest msa_triton/tests/test_eager.py msa_triton/tests/test_topk.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 python -m pytest msa_triton/tests -q -s
# A3 上不设置 TRITON_INTERPRET：
MSA_TEST_DEVICE=npu python -m pytest msa_triton/tests -q -s
```

Eager 的 BSND/FP64 基准测试固定在 CPU；score、top-k、attention、k2q 和
组合测试通过 `MSA_TEST_DEVICE` 选择被测设备。目标设备缺少依赖时 pytest
会报告 skip，不能据此判定该设备验收通过。

## 2026-09-28 初版执行记录（历史）

环境为 conda `veomni`：Python 3.12.0、PyTorch 2.7.1+cpu、上游 Triton
3.3.1、NumPy 2.2.6、pytest 8.3.5。无 GPU/NPU；Triton 测试使用
`TRITON_INTERPRET=1`，并设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。

初版完整执行结果：**91 passed，0 failed，0 skipped，288.94 秒**。
测试进程退出码为 0，命令如下：

```bash
source /home/yeep/env/miniconda/etc/profile.d/conda.sh
conda activate veomni
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 \
python -m pytest -q msa_triton/tests
```

| 测试文件 | 通过数 | 主要覆盖 |
|---|---:|---|
| `test_eager.py` | 27 | pack/unpack、BSND 迁移精确对齐、空选择/空序列、数值诊断 |
| `test_topk.py` | 12 | 独立排序规则、local 名额、group 隔离、无重复与 padding |
| `test_k2q.py` | 11 | CSR 边往返、slot、序列边界、空行、容量保护 |
| `test_triton_score.py` | 16 | BF16 输入的 FP64 前反向门禁、ties、causal 与 block 边界 |
| `test_triton_attention.py` | 21 | O/LSE/dQ/dK/dV、GQA、非连续输入、高扇入与 block 顺序 |
| `test_pipeline.py` | 4 | 三阶段 BF16 组合、小模块 FP32 参数梯度、native 误差诊断 |

出现 513 条 `DeprecationWarning`，来自上游
`triton/runtime/interpreter.py:790` 的 NumPy ndarray→scalar 转换；未屏蔽，
不属于精度断言失败。`python -m compileall -q msa_triton` 亦已通过。

精度比较使用同一份 BF16 量化输入及上游梯度。Triton 的 score、attention
前反向分别对照独立 FP64 oracle；top-k 和 k2q 是离散索引，用独立 Python
规则验证。小模块测试以 FP32 比较两种实现的输出及 projection 参数梯度，
不代表完整模型、分布式训练或 KL 的端到端验证。

原生 BF16 eager 的诊断样例为 lengths=[3,17]、G=2、Hq/G=2、D=16。
以下差异对照末端舍入的 FP64 oracle，保留在报告中，不算严格门禁通过：

| 张量 | 最大绝对差 | 超过 `atol=rtol=1e-4` 的元素数 |
|---|---:|---:|
| output | 0.0078125 | 508 / 1280 |
| LSE | 0.0057024956 | 62 / 80 |
| dQ | 0.0078125 | 576 / 1280 |
| dK | 0.01171875 | 333 / 640 |
| dV | 0.015625 | 321 / 640 |

组合路径中 native BF16 eager 与 Triton 的输出差异也单独记录：

| lengths | G | Hq/G | D | 最大绝对差 | 平均绝对差 |
|---|---:|---:|---:|---:|---:|
| [1,3] | 1 | 1 | 8 | 0.00390625 | 0.00025177002 |
| [7,5] | 2 | 2 | 16 | 0.015625 | 0.00082971901 |
| [17,9] | 4 | 1 | 8 | 0.01171875 | 0.00094251451 |

Benchmark 入口已执行以下 CPU smoke，score、top-k、k2q、attention、组合
前向及 score/attention 前向加反向均能运行。计时仅用于检查脚本功能：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 \
python -m msa_triton.benchmark --device cpu --lengths 4 \
  --q-heads 2 --kv-heads 1 --dim 4 --index-dim 4 \
  --block-size 2 --topk 2 --warmup 0 --repeats 1 --backward
```

NPU 编译、BF16 实机精度、并发执行、8K–32K 内存和性能均未执行；目前没有
设备加速比结论。

## 后续审查与修复

同日复核补充了两个 T=2、输入绝对值不超过 1 的 BF16 score 样例：FP32 舍入
改变最大值的胜者集合，score 误差仍通过上述门槛，但 dQ/dK 最大误差为 0.5。
这两个样例不在历史 91 项测试中，不能将历史通过记录理解为全输入域的梯度保证。
现已修复 score 的胜者判定，并将反例纳入正式回归；两个原始诊断的 score、
dQ/dK 最大绝对误差均降为 0。另修复了 int64 地址运算与按需 attention
backward，精度阈值保持不变。实现、内存代价和新增 47 项测试说明见
[correctness_fixes.md](correctness_fixes.md)。历史审查和消融结果保留在
[code_review_response.md](code_review_response.md)。

## 2026-09-28 修复后执行记录

同一 conda `veomni` 环境与完整测试命令执行结果为：
**138 passed，0 failed，0 skipped，287.68 秒**，退出码 0。

| 测试文件 | 通过数 | 本次变化 |
|---|---:|---|
| `test_eager.py` | 27 | 原生 BSND / TND 适配回归 |
| `test_topk.py` | 12 | 离散选块回归 |
| `test_k2q.py` | 11 | CSR 与容量保护回归 |
| `test_index_addressing.py` | 3 | 新增生产 helper 的 int32 边界外地址计算 |
| `test_triton_score.py` | 49 | 新增 33 项，覆盖消减、ties、乘法残差、三种 dtype、no_grad |
| `test_triton_attention.py` | 32 | 新增 11 项，覆盖梯度组合、输入不被修改、FP16/FP32、GQA=16 |
| `test_pipeline.py` | 4 | 三阶段组合及小模块参数梯度回归 |

795 条 warning 均来自上述上游 Triton interpreter 的 NumPy 标量转换弃用
提醒，未屏蔽；没有精度失败。两个原始 score 诊断的前向与 dQ/dK 最大
绝对误差均为 0。原生 eager 误差诊断与历史报告一致；阈值没有放宽。
Benchmark 的 CPU smoke 与 Python compileall 也通过，不报告设备加速比。

补充的 CUDA sm80 离线编译检查发现新增梯度条件中的三个连续 `or` 不被
Triton 编译器接受，现已加括号明确分组，逻辑不变。最终 31 个变体通过：
15 个 score 变体覆盖保存/不保存反向状态、正/负/零 scale 和不同维度；
16 个 attention 变体覆盖前向、所有非空梯度组合、KV 和私有 atomic 分支。
离线检查命令如下，仅适用于本机上游 Triton 环境：

```bash
unset TRITON_INTERPRET
TRITON_CACHE_DIR=/tmp/msa_fixes_compile_cache \
python -m msa_triton.tests.probe_offline_compile
```

该步骤没有执行 GPU kernel，也没有编译 NPU 目标。编译修正后的 11 项
独立梯度、dtype/GQA 和私有 atomic 用例另行复跑：**11 passed，21 deselected，
18.92 秒**。NPU 的完整数值、长序列内存和性能验证仍须迁移后执行。

## 首轮 NPU 报告后的候选验证

用户报告的设备结果为 **90 passed / 48 failed**，包括一个确定性的 KV
backward 编译崩溃，以及 score 首发非法访问后的级联失败。报告摘要、
候选改动与实机复测命令见 [npu_validation_20260928.md](npu_validation_20260928.md)。
此轮仍没有本机 NPU；下列结果仅验证 CPU 语义和上游编译接口。

完整 CPU interpreter 命令保持不变，结果为 **145 passed，0 failed，
0 skipped，219.27 秒**，退出码 0。795 条上游 NumPy 标量转换弃用 warning
未屏蔽。原 138 项全部保留，新增 4 项 attention 补偿归约树高低分量精确
比较及 3 项连续空序列边界测试。精度阈值未修改。

扩展的离线编译检查 **41 个 CUDA sm80 变体通过**，增加报告中的 D=7 /
NSEQ=5 score 特化，以及 D=128 / G=2 / Hq=4 / block=4 / NSEQ=2 attention
特化。它们没有执行 GPU 算术，也不验证 Ascend 编译与调度。

两个诊断工具的本机 smoke：同进程七种 shape 的 BF16 前后向和 FP32
forward-only 各 7/7 通过；隔离 runner 能逐项启动新进程，拒绝把 skip、
xfail、xpass、超时或崩溃计为通过。Python compileall 与文档本地链接检查
也通过。诊断工具不增加 pytest 收集数量。
最终版本的隔离 runner 又分别执行了报告中的首个 score 与 D=128 attention
节点，CPU interpreter 下 2/2 通过，确认复测命令能选择并运行这些节点。

**候选已于 2026-09-29 回传 NPU 复测失败结果，NPU 验收状态保持失败。**
具体结果见 [npu_retest_20260929.md](npu_retest_20260929.md)。本机耗时变化
不用于推断 NPU 加速比。

## 六项结构对照后的逐 query 累加候选

用户在 `6d47514` 上运行的 NPU 六项对照为 **5 failed / 1 passed**，并非
attention 或 MSA 验收通过。进一步 CPU 消融显示，tile-sum + Kahan 丢失
`2^-12` 的 dK/dV 残差，新增 12 项精度回归全部失败；历史补偿树则通过这
12 项。因此正式候选保留补偿，改为逐 CSR query 的 `[D]` TwoSum 累加，
KV kernel 禁用浮点融合。改动与数值反例见 [修复记录](correctness_fixes.md)。

conda `veomni`、CPU Triton interpreter 完整回归：**157 passed，0 failed，
0 skipped，439.58 秒**，退出码 0。命令为：

```bash
source /home/yeep/env/miniconda/etc/profile.d/conda.sh
conda activate veomni
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MSA_TEST_DEVICE=cpu TRITON_INTERPRET=1 \
python -m pytest msa_triton/tests -q -o addopts=
```

相对历史 145 项，attention 新增 12 个消减用例；4 个归约 helper 用例由
树配对检查改为真实生产单 FP32 贡献流，对照独立 `math.fsum`。其最终版本
直接传入常量 `b_lo=0.0`，另行复跑 **4 passed，1.20 秒**。旧树精确 hi/lo
测试仍保留在显式诊断文件中。正式公开 API 门禁仍为 `atol=rtol=1e-4`，
未放宽；本轮 795 条上游 NumPy 标量转换弃用 warning 未屏蔽。

新 KV 的联合/独立梯度组合及原报告 D=128 特化，通过 **41 个 CUDA sm80
离线编译变体**；编译选项与生产 KV 一致，关闭浮点融合。7 项隔离进程
legacy/current 诊断在 CPU 全部通过，仅验证 runner 与相应数学路径；不能
覆盖 NPU 上已经观察到的 legacy 编译失败。Python compileall、48 项远端
attention 测试收集、文档本地链接与候选哈希检查均通过。

本机仍没有 NPU，以上记录为本机 CPU 与离线编译验证；串行 query 的设备
吞吐尚未测量。后续实机确认见下节。

## 2026-09-29 attention NPU 复测通过（用户确认）

针对明确指向 `df31a1d` 的确认问题，用户答复：“48 项 NPU 测试全部通过”。
测试范围为 4 个生产累加 helper 和 44 个 attention 用例，包含原 D=128
失败节点、不同长度/GQA/独立梯度、长序列 BF16 舍入及 12 个消减回归。
公开精度门禁仍为 `atol=rtol=1e-4`。

据此记录 **48 项通过，本轮 attention 编译、执行与数值门禁通过**。
这是用户实机确认，本机没有执行 NPU 测试；该确认未附本轮耗时、退出码、
日志目录或环境重核结果，本文不补造这些字段。执行协议保留在
[npu_attention_streaming_retest.md](npu_attention_streaming_retest.md)。

此结果证明新路径通过该测试范围，不等于 bishengir 具体失败 op 已定位，
也不覆盖 score、长序列性能、并发、VeOmni 或完整模型验收。
**score 非法 GM 地址仍未解决，全 MSA NPU 验收尚未通过。**
