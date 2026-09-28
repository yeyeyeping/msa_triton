# MSA 流程与 Indexer 训练行为

本文约定 `msa_triton` 的算子职责，以及训练和不训练 Indexer 时的计算、梯度差异。当前阶段只进行独立算子实现、测试和验证，暂不接入 VeOmni。

优化目标为 Ascend 910C/A3，首版关注 BF16、8K–32K packed self-attention，允许混用 Triton 与 NPU 原生算子。本文描述接口与训练语义；已实现范围及实际执行的验证见 `implementation_plan.md`、`validation.md`。独立 KL kernel 和 VeOmni 接入仍属后续工作。

## 1. 基本约定

- 采用每个 GQA group 独立选块的设计，本文记 `G = Hkv`，每个 group 包含 `Hq / G` 个主 query heads。
- 主 Q/K/V 和 index Q/K 使用 TND 布局；`T` 是 packed tokens 总数，序列边界由 `cu_seqlens` 指定。
- 每条 packed 序列独立划块，block 编号从各自序列起点计算；不同序列之间不可相互访问。
- local blocks 指当前块及紧邻的历史块，占用 top-k 名额，不在 top-k 之外追加。
- 有效 block indices 无重复、左对齐，不足部分填 `-1`；下游不依赖有效项的排列顺序。
- 首版主 attention 的 dropout 为 0；KV cache decode、Context Parallel 和分布式接入不属于当前独立验证范围。

这里的 group 独立选择遵循 [MSA 技术报告 §3.1](https://arxiv.org/html/2606.13392v1#S3.SS1)。锁定的 Transformers 上游源码也已经保留 group 维度；本地 VeOmni 中较旧的 HF 派生实现仍有跨 index heads 取 max 的行为。本项目从上游迁移 eager，来源见 `../eager/PROVENANCE.md`，不以旧派生实现作为等价基准。

## 2. MSA 的三个阶段

```text
hidden states
    │
    ├── Indexer projection / norm / RoPE
    │       │
    │       └── index Q/K
    │               │
    │               ▼
    │         m3_index_score
    │               │
    │       block scores [T,G,Bmax]
    │               │
    │               ▼
    │            m3_topk
    │               │
    │         indices [T,G,K]
    │               │
    └── 主 Q/K/V ───┴──► m3_sparse_attention ──► O ──► output projection
```

### 2.1 打分：`m3_index_score`

输入为 `index_q[T,G,Di]`、`index_k[T,1,Di]`、序列边界及 block size。每个 group 独立计算缩放点积，先施加 token 级 causal mask，再对 block 内 tokens 取最大值：

\[
S_{t,g,j}=\mathrm{scale}\,(IQ_{t,g}\cdot IK_j),\qquad
M_{t,g,b}=\max_{j\in b,\ j\leq t} S_{t,g,j}.
\]

这里的 `t`、`j` 在 causal 判断中使用各自序列内位置。scale 可显式传入，默认值为 `1.0`，保留 Transformers 原生选块路径的缩放约定。论文 token KL 使用 `Di**-0.5`；数学上正比例缩放不改变排序，但公开 FP32 舍入可能引入并列，分数和概率也会改变，不能混淆两者。

输出为 FP32 `scores[T,G,Bmax]`，其中 `Bmax = ceil(max_seqlen / block_size)`。未来块、超出本序列的块填 `-inf`。该阶段不跨 group 归约，不做 softmax、local boost 或 top-k。

独立算子提供 index Q/K 的 backward，最大值并列时按 `torch.amax` 的规则均分梯度；无效块梯度为零。MSA 的离散选块链路仍可在 `no_grad` 下调用它。这个 backward 不替代官方 token 级 KL。

Triton 在公开 FP32 分数舍入前，以补偿点积的高位与残差判定胜者和真正并列。
候选分数舍入相等不等于 block 内的数学最大值并列；top-k 仍按公开分数选块。
只有需要 score backward 时才保存最大值高位、残差和胜者信息，`no_grad`
不分配这些状态。原生 eager 保留上游打分数学，在极近分数或消减处可能与
Triton 存在梯度差异，详见 [必要修复](correctness_fixes.md)。这不改变 KL
需要从 index Q/K 重算 token logits 的边界。

### 2.2 选块：`m3_topk`

输入为 block scores、序列边界、block size、top-k 和 local blocks 配置。每个 query、每个 group 独立选择，先保证 local blocks 被保留，其余名额按原始 block scores 选择。

输出为 int32 `indices[T,G,K]`，`K = topk_blocks`。要求 `0 <= local_blocks <= K`，且不原地修改输入 scores。并列分数允许选择不同的等价候选，不要求参考实现和优化实现返回完全相同的索引。

该阶段输出离散索引，不可微。

### 2.3 注意力：`m3_sparse_attention`

输入为 `q[T,Hq,D]`、`k/v[T,G,D]`、indices 和序列边界，要求 `Hq % G == 0`。group `g` 对应连续的 `Hq/G` 个主 query heads，共用 `indices[:,g,:]`。

该阶段只在选中 blocks 的 causal tokens 上执行 attention，忽略 `-1`。即使当前 block 被选中，也必须屏蔽其中的未来 tokens。

输出为 `out[T,Hq,D]`；可选返回 FP32 `lse[T,Hq]`，使用自然对数定义。scale 默认 `D**-0.5`。主 attention 支持 Q/K/V backward，反向复用本次前向的 indices，不重新选块。

## 3. 不训练 Indexer

这里指主模型仍然训练，但 Indexer 参数冻结，不等同于推理模式。

Indexer 继续执行打分和选块，主 attention 正常计算输出及 Q/K/V 梯度。选块通过离散 indices 连接主 attention，LM loss 不会沿这条路径训练 Indexer。

实现约定：

- 冻结 Indexer 参数，清除可能遗留的梯度，并排除其 optimizer 更新。
- Indexer projection、打分和 top-k 均可在 `no_grad` 下运行。
- 不计算 KL，也不保存 Indexer 的反向计算图。
- 主 attention 仍须保存自身 backward 所需的信息；不对外返回 LSE，不代表内部无需保存。
- Indexer 参数固定不意味着 indices 固定：主模型的 hidden states 随训练变化，选块结果也可能变化。

## 4. 使用 KL 训练 Indexer

三个选块与 attention 阶段保持不变，额外增加 KL 分支：

```text
index Q/K ───────────────────┐
                            │
主 Q/K 的 detached 视图 ─────┼──► 选中 tokens 上的 KL ──► Indexer 梯度
                            │
本次前向的 indices ──────────┘
```

### 4.1 KL 的监督对象

官方目标是在选中 blocks 内的 causal tokens 上匹配学生和教师分布：学生对 index token logits 做 softmax；教师先对主 attention 各 head 的 logits 分别做 softmax，再在同一 group 内平均概率。损失方向是 `KL(stopgrad(teacher) || student)`，并对 query 和 group 取平均。详见 [MSA 技术报告 §3.2，公式 9–11](https://arxiv.org/html/2606.13392v1#S3.SS2)。

因此必须区分两种分数：

- block scores：块内 token logits 的最大值，用于选块。
- token logits：用于构造 KL 的学生分布，不能由 block scores 还原。

不得对 block scores 做 softmax 后当作官方 KL，也不得将 local 选择过程中使用的 `+inf` boost 带入 KL。

### 4.2 梯度边界

Indexer 输入使用 `hidden_states.detach()`，但其 projection 输出 index Q/K 保留计算图。KL 更新 Indexer 参数，不通过输入流入 backbone。

本项目据此约定：

- `m3_index_score → m3_topk` 仍可仅执行 forward。
- KL 从可导的 index Q/K 重算选中 tokens 的 logits，梯度直接回到 Indexer，不经过 block max 或 top-k。
- 教师使用主 Q/K 的 detached 视图；主 attention 自身的 LM backward 不受影响。
- 若 Indexer 包含可训练的 norm 参数，这些参数也属于 KL 更新的分支参数。

总损失为：

\[
L=L_{\mathrm{LM}}+\lambda\sum_{\mathrm{layers}} L_{\mathrm{KL}}.
\]

`λ` 是上层训练配置，不属于三个核心算子的参数。独立 packed 验证时，KL 对真实 query tokens 和 groups 取平均，不计填充槽位；后续分布式归约规则在接入阶段单独约定。

### 4.3 Warmup 与稀疏训练

官方 warmup 使用完整 causal context，随后切换到选中 tokens 上的稀疏 KL。两个阶段都要求学生与教师具有相同的有效 token 集合。[MSA 技术报告 §3.2](https://arxiv.org/html/2606.13392v1#S3.SS2)

本文的三个核心算子描述稀疏阶段。Warmup 是另外的训练流程，不能简单地认为“开启 KL”就自动进入 full attention；其实现不在当前三阶段接口内隐式处理。

## 5. 两种模式的行为差异

| 项目 | 不训练 Indexer | KL 训练 Indexer |
|---|---|---|
| Indexer 前向 | 执行 | 执行 |
| block score / top-k | 仅前向 | 仍可仅前向 |
| Indexer projection 计算图 | 不保留 | 保留，输入 hidden states detach |
| 主 attention backward | 正常执行 | 正常执行 |
| KL 计算 | 跳过 | 稀疏阶段在本次选中的 tokens 上执行 |
| Indexer 参数更新 | 无 | 来自 KL |
| KL 对 backbone 的梯度 | 无 | 无 |
| KL 对主 Q/K/V 参数的梯度 | 无 | 无 |
| 额外开销 | 无 KL 开销 | token logits 重算、KL 和 Indexer backward |

训练策略由上层的 `train_indexer` 开关控制。它与 `model.train()/eval()` 是不同维度，也不属于三个核心算子的参数。

## 6. 实现组织与接口边界

三个核心算子直接调用，不提供 `backend="eager"` 一类参数，也不在 `msa_triton` 中实现后端选择或自动 fallback：

```python
scores = m3_index_score(index_q, index_k, ...)
indices = m3_topk(scores, ...)
out = m3_sparse_attention(q, k, v, indices, ...)
```

- Triton 实现：直接处理 TND，提供 score 和 attention 的实际 kernel 前反向；top-k 首版使用设备上的 PyTorch 原生算子。
- Eager reference：从上游迁移 BSND 计算，对外采用与 Triton 相同的 TND 接口，内部通过可微 unpack/pack 适配；任意 block mask 只在 eager attention 内部构造，不通过参数切换。
- VeOmni：后续由其算子注册与配置机制负责实现选择，当前不进行接入。
- KL：作为独立辅助计算消费 index Q/K、主 Q/K 和 indices，不要求改变三个核心算子的职责。

主 attention 的 LSE 可作为 KL 的可选缓存；复用时必须匹配 token 集合、scale 和 head 对应关系，并按教师统计量 detach。LSE 不能代替计算 token logits，完整 token-score 张量也不要求落地保存。

以下代码仅说明梯度组织，不是完整可执行接口：

```python
# 主分支正常保留梯度。
q, k, v = main_projection_norm_rope(hidden_states)

# 参数冻结和 optimizer 配置由上层负责。
with torch.set_grad_enabled(train_indexer):
    index_q, index_k = index_projection_norm_rope(
        hidden_states.detach()
    )

# 选择路径不构建反向图，不影响原始 index Q/K 供 KL 使用。
with torch.no_grad():
    scores = m3_index_score(index_q, index_k, ...)
    indices = m3_topk(scores, ...)

out = m3_sparse_attention(q, k, v, indices, ...)

if train_indexer:
    # 概念接口，具体 KL 算子签名另行约定。
    kl_loss = index_kl(
        index_q, index_k,
        q.detach(), k.detach(),
        indices,
        ...
    )
```

## 7. 验证要求

详见 `implementation_plan.md` 和 `validation.md`。使用同一份 BF16 输入，转 CPU FP64 完成参考前反向，结果再转换为公开 dtype，以 `atol=rtol=1e-4` 比较。FP64 oracle 独立实现，不复用原生 eager 的 FP32/BF16 中间舍入。原生 BF16 eager 与 oracle 的精度差异另行报告，不能通过放宽 Triton 的 FP64 门禁来掩盖。

1. 对齐各 group 的 block scores；构造不同 groups 选择不同 blocks 的样例，防止误做跨 group 归约。
2. 验证 top-k 的 local 保留、名额占用、去重、`-1` 填充和输入 scores 不被修改。并列分数按合法选择集合验收。
3. 固定相同 indices，对照参考实现的 attention 输出及 dQ/dK/dV。
4. 关闭 KL 后，确认 Indexer 参数不更新，主分支仍有正常梯度。
5. 仅对 KL backward 时，确认 Indexer 有梯度，backbone 和主 Q/K/V 参数无梯度。
6. 固定输入、参数和选择结果，确认 KL 分支不会修改 scores、indices 或当前 attention 输出。
7. 覆盖变长 packing、短序列、尾块、因果边界及跨序列隔离；KL 的教师和学生必须使用同一个有效 token 集合。

第 5、6 项及第 7 项中的 KL 部分是后续 KL 实现的验收要求，本轮尚未执行。
当前组合测试使用小型 projection→MSA→projection 模块验证主分支参数梯度
和冻结 Indexer 的边界；尚未执行完整 MiniMax 模型或 VeOmni 端到端测试。
