# code_review.md 复核、反例与修订建议

> 状态更新：下文保留修复前的审查过程与观测结果，其中“当前实现”和失败
> 诊断均指审查时版本。已确认的 score 梯度、地址宽度及按需反向问题现已
> 修复，详见 [correctness_fixes.md](correctness_fixes.md)；当前验证结果见
> [validation.md](validation.md)。原始 `code_review.md` 未改动。

复核日期：2026-09-28。对应 [原审查](code_review.md)。本次保留原文和生产算子，
核对代码、已有测试、本地 Triton 3.3.1/PyTorch 2.7.1 源码及官方文档，并运行
CPU interpreter 小实验和独立副本上的数值消融。没有执行 A3 编译或性能测试。

总体判断：**优化方向大体合理，但不能直接按 D 的清单实施。** 当前实现确实有
细粒度 program、重复读取、FP32 工作区和 metadata 同步的成本；但原文混合了
已确认问题、设备风险和未经测量的收益推断。部分“几行修改”会破坏依赖或容量约束。
另发现原文没有覆盖、可在小型 CPU 用例上复现的 score 梯度问题。

## 1. 应先处理的新增数值问题

位置：[index_score.py](../../triton/index_score.py) 的 FP32 点积及最大值判定，尤其是
forward 的 `logits/maximum/winners`。它与 A3 的“前反向重算使用不同归约布局”
是两个问题：**即使前反向完全一致，FP32 与 FP64 的胜者集合也可能不同。**

用 BF16、T=2、G=1、Di=3、block_size=2、scale=1，所有输入绝对值均不超过 1：

```text
q[0] = q[1] = [1, 2^-12, 1]
k[0] = [1, 2^-12, -1]
k[1] = [0, 0, 0]
cu_seqlens = [0, 2]
只给 score[1,0,0] 上游梯度 1
```

FP64 下，token 1 的两个 logits 为 `[2^-24, 0]`，只有 k[0] 获胜；当前 FP32
计算得到 `[0,0]`，按并列分配梯度。因此：

| 结果 | Triton | FP64 oracle |
|---|---|---|
| score[1] | 0 | 5.960464477539063e-8 |
| dQ[1] | `[0.5, 2^-13, -0.5]` | `[1, 2^-12, -1]` |
| dK[0] | `[0.5, 2^-13, 0.5]` | `[1, 2^-12, 1]` |
| dK[1] | `[0.5, 2^-13, 0.5]` | `[0,0,0]` |

score 通过原门槛，但 dQ/dK 最大绝对误差为 **0.5**。进一步把 k[1] 改为
`[0,2^-12,0]`：FP64 中两者真正并列，当前 Triton 只选中 k[1]；token 1 的
score 此时完全一致，梯度仍相差 0.5。

复现脚本已保存为 [probe_score_rounding.py](../../tests/probe_score_rounding.py)：

```bash
source /home/yeep/env/miniconda/etc/profile.d/conda.sh
conda activate veomni
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 \
python -m msa_triton.tests.probe_score_rounding
```

它报告 `score.within_tolerance=true`、`dq/dk.within_tolerance=false`，不是一个
被计为通过的 pytest 测试。原 91 项通过的历史记录仍然成立，但不能外推为所有
有限 BF16 输入均满足 FP64 梯度门槛。本次没有放宽阈值或更改既定 tie 语义。

建议将这个反例作为优先补充的失败回归，评估点积精度及最大值判定。保存 tie 位图
只能固定“当前前向认定的胜者”，不能修复 FP32 前向相对 FP64 已经选错胜者的问题。
不能以“连续输入 tie 概率为零”忽略 BF16、重复 key、scale=0 或消减造成的情况。

## 2. A 类逐项复核

### A1：风险成立，示例阈值与验证方式需要纠正

T=262144、Hq=64、D=128 时，最大有效元素偏移是
`(T-1)*64*128 + 63*128 + 127 = 2^31-1`，尚未溢出。
T=262145 才开始出现合法元素偏移溢出。`T*Hq*D < 2^31` 可以作为保守限制，
但不要把等号边界描述为已经发生越界；还必须覆盖 scores、indices、stats 等地址。

只把 `program_id` 提升为 int64 不完整：KV backward 的 `query_tokens` 是从
int32 CSR 中加载的，也必须在乘法前提升。CSR 的 `offset+qs`、`offset+=32`
接近 int32 上限时也会回绕，mask 不会自动修复已溢出的比较。中间结果溢出后再
`.to(tl.int64)` 已经太晚。最终应审计各条地址计算链，而不是机械替换一个变量。

这并非 interpreter 结构上无法测试的问题。本次用两个合成 token ID 和小数组执行
同样的偏移表达式，得到 `2147483647` 与 `-2147475457`，无需分配巨大 Q 张量。
NPU 上的地址提升代价、编译和实际边界仍须验证。

### A2：属于性能浪费，但不能独立跳过 query kernel

[sparse_attention.py](../../triton/sparse_attention.py) 的 query backward 不仅计算
dQ，还生成 `center/centered_delta/probability_mass`。KV backward 使用这些统计量，
所以 `Q.requires_grad=False` 并不意味着可以跳过整个 query kernel。

正确拆分应区分“统计预处理”和“dQ 计算/写回”，并按 dK、dV 的需求控制 KV 分支。
只有 K/V 都不需要梯度时才可整体省 CSR/KV kernel。当前 dV 也使用重算的 probability
mass；设计 V-only 快路径时须保持这部分数值策略一致。补测 Q/K/V 的 7 种非空梯度
组合，尤其 Q 无梯度而 K 有梯度、以及 V-only。

原文 KL 教师的例子不准确：KL 损失分支不沿 detached teacher 回传；LM 主损失
仍可正常触发主 attention backward。这里更直接的应用是部分冻结或独立算子的
部分梯度需求。

### A3：合理的设备一致性风险，不能以改文档替代既定语义

不同归约布局导致 tie 重算不一致的推理成立，但尚未在 A3 复现，不能写成已确认
设备缺陷。原文三个方案中：

* 放弃均分违反已约定接口；不能直接采纳。
* dQ 自行重数、dK 仍用旧 count 会产生不一致的梯度规则，不能作为完整修复。
* tie 位图可保证前反向采用同一集合，但需要计算代价。32K/G=4/block=128 时，
  每个 score 存 4 个 int32 位图本身就需 512 MiB；若保留旧 tie-info，还要另计。

应同时验证“前反向集合一致”和第 1 节的“FP32/FP64 集合差异”；两者不可混为一谈。

### A4：同步次数成立，建议的缓存键不安全

正常训练链路 score、top-k、attention、backward CSR 各读取一次 host metadata，
合计 4 次；index score backward 自身没有额外 D2H。冻结 indexer 或重算策略会改变
实际每步调用次数。240 次是 60 层且每层各调用一次的场景计数，不是耗时测量。

`(data_ptr, _version, numel)` 不能直接作为通用缓存键。本次已复现：

* `base[:3]` 与 `base.as_strided((3,),(2,))` 可具有相同三元组，但内容不同。
* inference-mode 张量读取 `_version` 会报错。
* `.data` 写入不更新版本；此外还存在设备、指针复用、生命周期及 T/block_size
  校验上下文缺失的问题。

更合理的是由 batch/集成层拥有、明确不可变的 prepared metadata，验证一次后复用
host lengths、device block prefix/token 映射。它仍需定义失效、生命周期、设备和
图捕获规则；本次不擅自修改三个公开接口。缓存一次 Python 校验也不会自动消除
逐层 repeat_interleave、序列查找、device 临时张量和所有同步来源。

### A5：维护建议成立

统一 Q_TILE 和静态归约树、加静态断言合理。“改宽度一定静默丢边”过于绝对，也可能
编译报错或产生非法 gather。当前硬编码一致，未发现这部分现有功能错误。

## 3. B 类逐项复核

| 条目 | 复核结论 |
|---|---|
| B1 gather | 当前官方文档已经列出 A2/A3 的 FP32 gather 支持，并给出 int32 索引、axis=0 的例子。应改为“锁定目标版本并验证本实现形状/资源/精度”，不再写完全未知。 |
| B2 constexpr | 降低 NSEQ/NB 特化数量有价值，但并非每个新 max_seqlen 都编译：NB 实际为 ceil(max_seqlen/block_size)，同桶且其余参数相同时可复用。SCALE 通常由固定维度/配置决定，不随 packing 必然变化。 |
| B3 延迟导入 | 本地非 interpreter 的 JITFunction 创建及 constexpr 识别通过。字符串注解是本地 Triton JIT 明确支持的机制，不是已经确认的失效。模块拆分可改善维护性，优先级较低。 |
| B4 CSR 原生算子 | 目标 NPU 支持与性能确实需验证。改 int32 stable sort 仍引入 stable=True 的目标后端要求，不能据此宣称移植问题消失。 |

B1 来源：[Triton-Ascend gather 文档](https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.gather.html)
（页面更新于 2026-09-24；本次访问于 2026-09-28）。文档支持不等于本项目已经在 A3 编译通过。

B2 还应考虑：移除 constexpr 后，上游 JIT 的普通整数仍可能按值 1、对齐类别特化；
pow2 tile 跨桶也仍产生变体。必要时配合 `do_not_specialize`，并同时检查调用的 JIT
helper 参数注解。先统计真实 packing 的 NB/NSEQ 组合与首次编译耗时，再决定桶化
和运行期参数；不能承诺仅删注解即可消除所有编译抖动。
[JIT 参数说明](https://triton-lang.org/main/python-api/generated/triton.jit.html)

## 4. C 类逐项复核

### C1/C2/C3/C6：矩阵分块方向正确，倍率与所有权结论过强

当前向量归约实现不适合作为已经调优的 A3 训练版本。矩阵化、减少 program 数量、
复用 KV/Q/dO 是主要优化方向。但“快 1–2 个数量级”“访存直接降 16×/128×”尚无
实测：逻辑 load 次数不等于 HBM 流量，缓存、UB/寄存器、并行度和稀疏行负载都会影响
结果。没有显式 autotune/num_warps/num_stages 也不是独立正确性缺陷，存在默认配置。

| 条目 | 建议保留的方向 | 必须补的条件 |
|---|---|---|
| C2 | 同 query/group 的多个 heads 共享 KV tile；比较 head 子块与完整 group | 输出/softmax 累加器资源随 head 数增长；补到 32/64 会增加无效计算，不是必然更快 |
| C3.1 | 无因果有效 key 的 program 直接写 `-inf/0`，跳过点积 | 必须写满输出，不能留下 empty 工作区；约一半只适用于相应形状的工作量估计，不是端到端加速比 |
| C3.2 | 按 query tile 做 QK，再在 block 内取 max | tile 必须处理 group 与 packed 序列边界，不能让同 tile 跨序列后复用错误的 KV |
| C6 | 由 key 子块复用 CSR queries/Q/dO，比较 block ownership | 整个 block 合并会把 128 个 program 合成 1 个，可能降低并行度、恶化热点尾部；可试 key 子块或 query 分片加确定性二次归约 |

C3 的顶层动态 early return 已用小 kernel 在本地 interpreter 和离线 CUDA sm80
编译验证；这不代表 Ascend 支持已验证。采用 if/else 并显式写出无效结果同样可行。

C6 中 `local_blocks=1` 的本地块通常只被本块的 queries 强制选择；热门 sink/global
块才可能被大量后续 queries 选择。SoA 的统计量共有 **5 个**，不是 4 个；它可以
简化布局，但 CSR query 本身是不连续索引，不能保证改 SoA 就变成连续合并访存。

新增的精度约束：改用 `tl.dot` 不只是换写法。特别是把 FP32 probability 或 dS 转
BF16 做 PV/dQ/dK 会增加中间舍入，不能假设仍满足当前 FP64 门禁。契约约束结果
精度，并非形式上禁止所有中间低精度运算。需要明确每次 dot 的输入和累加 dtype，
所有候选 tile 都要验证；不能借用原生 eager
的 BF16 中间舍入来解释优化 kernel 的门禁失败。
[dot 精度参数](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

### C4：理由错误，但简化方案有初步实验支持

BF16 在数值约为 1 时，一个 ULP 约为 0.0078125，而门槛约为 0.0002。
很小的 FP32 误差若跨过 BF16 舍入中点，末端误差就会超过门槛；“BF16 精度低，
所以 1e-4 余量巨大”的判断恰好相反。

此外，热门 key 最坏可接收 `T * (Hq/G)` 项；32K、Hq/G=16 时是 524288 项，
不是“至多几千项”。当前双分量归约也不代表点积/exp 全部具有 FP64 精度。

不过，不应因为理由错误就否决实验方案。本次在独立副本中将 tile 内 TwoSum 树
改为朴素 `tl.sum`，跨 tile 改用标准 Kahan：**现有 8 个参数化 FP64 前反向用例和
257-token 舍入回归均通过**。因此它是值得推进的简化候选。尚未验证 NPU、不同
随机种子、目标 GQA=16、8K–32K 热点和编译器的补偿表达式保留，不能立即判为普遍
冗余；也不应把 tile 内归约与跨 tile 补偿一起删除。

原实现跨 tile 也是 `_merge_pairs` 的 TwoSum，并非原文所说的现成 Kahan。
消融必须同步调整补偿符号及最终写回：Kahan 版本写 high，原双分量版本写 high+low，
不能机械删除 gather 后混用两种补偿状态。

### C5：可替换排序键，不能解除容量上限或承诺速度

当前复合 key 排序等价于“按 row 稳定排序”，但当前 `safe_rows` 将无效边设为 0。
直接套用原文代码会把无效边排入有效 CSR 前缀。应使用独立的排序键：

```python
sort_rows = rows.masked_fill(~valid, num_rows).to(torch.int32)
order = torch.argsort(sort_rows, stable=True)
```

计数用的 safe_rows 仍须是合法行号，不能拿 num_rows 哨兵去 scatter。valid row
排序内部保持原 flat-edge 顺序，才与旧 `(row,edge)` 复合 key 等价。

`row_ptr/counts/query_indices/slot_indices` 的 int32 表示约束仍然存在，不能因为
去掉复合 key 就删除容量保护。若要支持更大容量，必须重新设计指针/边数组及 kernel
索引位宽，或者证明更细的 nnz 上界。`argsort` 输出仍是 int64 索引，也没有消除
所有 int64 临时数组。当前 `rows/safe_rows/selected` 等本身仍为 int64。

“通常快 1.5–2×”没有本设备依据；稳定排序可能付出额外成本，应测整个 CSR 构建，
而不只测 key 张量大小。[PyTorch 2.7 argsort 文档](https://docs.pytorch.org/docs/2.7/generated/torch.argsort.html)

### C7：应优先评估内存，但转换必须完整

32K/Hq=64/D=128 下 q32、output32 各 1 GiB 的计算正确；backward 的 FP32
DOUT 和 dQ 又各占 1 GiB。还要加 K/V、BF16 原件、scores、CSR、STATS 及峰值
同时存活的张量，不能只预算 forward。

设备直接加载 BF16 后立即转 FP32 是合理候选。不能只删除 wrapper 的 `.float()`：
`empty_like(q32)`、dQ/dK/dV 工作区必须继续显式使用 FP32，否则会把累计结果提前
降精度。CPU 模拟和设备应保持等价算术，并明确测试设备 BF16 指针分支；不引入公开
backend 参数。tie-info 压缩成 int16 必须有 block_size 范围保护，当前公开接口并未
限定 block_size<=128。

### C8：消融已失败，不能直接删除中心精化

令重算概率为 r，m=Σr，c=O32·dO，代码保存的 δ=Σr(dp-c)/m。
当前 dS 为 `(r/m)*((dp-c)-δ)`；δ 已经除过 m，不能在使用时再除一次。
精确算术下 m=1，公式退化为通常的 softmax backward。

删除前扫确实少一次 QK/dP 遍历，但剩余 dQ 外积、CSR、KV 重算仍存在，不代表
query backward 或整个 backward 的总算力减半。应独立测“保留归约/关闭精化”和
“简化归约/保留精化”，最后再测组合，不能同时删掉后仅凭一个样例通过就下结论。

本次关闭中心精化、保留原归约，现有 8 个参数化测试结果为 **1 failed, 7 passed**。
失败不是长序列：lengths=[5,3]、G=2、Hq/G=2、D=128、block=4、topk=2，
`dQ[1,1,96]` 的结果如下：

```text
关闭精化后的 BF16：0.07421875
FP64 oracle 转 BF16：0.07373046875
oracle 未舍入值：0.07397456824450169
末端绝对误差：0.00048828125
```

真值距两个 BF16 值的舍入中点仅约 4.11e-8。它直接说明“小 FP32 误差被 BF16
量化跳变放大”的风险。257-token 专门回归单独通过，所以只跑该回归也不足以证明
中心精化可删除。

两种消融保存在 [probe_attention_ablation.py](../../tests/probe_attention_ablation.py)，
运行时创建临时源码副本，不修改生产文件。示例：

```bash
# 已知失败的 C8 小形状，pytest 应返回失败，不能计入通过数。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 \
python -m msa_triton.tests.probe_attention_ablation --mode no-center --case dim128

# C4 的 8 个现有参数化用例；另用 --case rounding 运行 257-token 回归。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 \
python -m msa_triton.tests.probe_attention_ablation --mode tile-sum-kahan --case matrix
```

### C9：优化方向成立，仍须保留准确的三种状态

unique winner 时 dQ 直接加载胜者一行 K 合理；dK 中该 tile 没有 tie 时也可省去
logits 重算。应区分 `info>0` 唯一、`info<0` 并列、`info==0` 无有效项。
tile 是否存在 tie 的判断使用 `<0`；将 0 一起计入会做无意义的慢路径。
这不解决第 1 节的前向胜者集合误差。

### C10：逐项处理，避免把接口条件当成通用事实

* 对新分配的 ranked 做第二次 `masked_fill_` 合理，不会修改调用者 scores。
* `sorted=False` 必须另行保证有效左对齐；原文的条件提醒正确。
* score dK 在 program 内循环 G 可省工作区/launch，但减少并行度并改变归约顺序。
* 遇到 `-1` 即停止依赖已经约定的左对齐条件；不能推广到带洞的内部测试/辅助索引。
  forward 与相关 backward 的消费规则应一致。
* 未请求 LSE 时可省其 allocation/log/store；无 backward 时还有更多可省状态，见下节。
* “未注册 custom_op 就必然断图”不成立。PyTorch 已支持直接编译用户 Triton kernel；
  custom_op 是 opaque 边界，triton_op 则可被编译器观察。注册 fake 也不会自动消除
  D2H、Python metadata 或设备图捕获问题。未来接入还须处理 autograd 注册和具体
  torch_npu 编译栈，不能只加两个装饰器。
  [PyTorch 用户 Triton kernel 教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)

## 5. 原文遗漏的工程与验证事项

1. **无 backward 状态分配。** 常见的冻结 indexer/离散选择路径仍计算并分配整份
   tie-info。可按实际是否构建反向图专门化；attention 无 backward 且无 LSE 时，
   normalizers 等也可以省。不能仅依据输入 requires_grad 推断外部 no_grad 状态。
2. **CSR 无用 slot 输出。** 当前 KV backward 只消费 query_indices，不消费
   slot_indices；可提供私有的轻量构建路径，同时保留完整 CSR 的测试/调试能力。
3. **重复序列查找。** 每个 score program 都扫描序列边界；metadata 预处理和 query
   tiling 应一起设计，避免只缓存 host 验证却保留海量重复的 token→sequence 搜索。
4. **目标形状覆盖。** 现有直接 attention 用例最大 GQA ratio=4，D=128 只测试了
   短序列。需补 ratio=16、多种 dtype、近 tie/精确 tie、消减、偏斜长度、热点、多种
   seed；FP16/FP32 既然公开支持，也需要直接单算子前反向测试。
5. **Benchmark 的 lengths 不是 packing。** 当前脚本逐个 length 运行 `cu=[0,length]`。
   需加入固定 T/不同 NSEQ、长度倾斜、不同有效 top-k 数与 CSR 热点，才可评估 A4/B2/C6。
6. **Benchmark 峰值和耗时拆分。** 只 reset 一次 peak 会把 saved_scores、score 梯度
   和后续 attention 的存活状态混在一起。需要分阶段 baseline/增量峰值、CSR、统计/dQ、
   dK/dV、组合 backward，以及冷编译/预热后耗时；报告 index_dim/local_blocks、设备
   和完整运行配置。CSR 独立构建时间虽已输出，仍缺反向子阶段数据。
7. **所有优化配置都要过精度门禁。** 改 tile、head/key 分块、group 归约顺序、dot
   dtype 或编译选项都可能改变 BF16 舍入结果；CPU interpreter 不模拟 NPU 并发和
   编译器重排。补偿表达式是否保留必须检查目标编译/数值，不能只看源码。

## 6. 修订后的实施顺序

1. **先补反例与边界保护。** 加 score 近 tie/精确 tie 失败回归，审计地址和 CSR
   游标位宽；保留原容差、tie 语义和容量检查。记录待解决的问题而不是归为测试通过。
2. **做局部且易验证的改动。** ranked 原地 boost、按需 LSE、unique-winner gather、
   因果 program 分支、无 backward 状态省略。每项分别测试；A2 先拆清统计依赖。
3. **做有证据的数值简化实验。** C4/C8 分开消融、补目标形状，再做 A3 编译与数值验证。
   不以 BF16 低精度为删补偿的充分理由，也不因原理由错误而拒绝可能有效的简化。
4. **解决 metadata 与编译变体。** 设计受控生命周期的 metadata，评估 runtime 参数
   和桶化；stable sort 作为候选，保留合法 sentinel 与容量上限。
5. **实机矩阵化与调度。** 合并 head 子块、key 子块、tl.dot、设备 BF16 load，联合
   测资源占用、数值、热点尾部和完整 backward 峰值，再决定 autotune 配置。

A3 tie 语义不是可以随意放弃的“独立决策项”；它属于已约定的正确性约束。
预期 16×/128× 复用、1–2 个数量级性能或“一天几行”不作为实施承诺。
