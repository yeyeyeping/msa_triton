# MiniMax M3 TND sparse attention

独立实现 MiniMax M3 的三阶段接口，目标设备为 Ascend 910C/A3。当前版本是
独立算子实现，正在进行 NPU 适配。**attention 已获用户确认通过 48 项 NPU
测试，score 的非法地址故障仍未解决，全 MSA 尚未通过验收。** 性能尚未验收。

* `eager/`：迁移 Transformers BSND 数学，以可微 unpack/pack 提供 TND 接口。
* `triton/`：score、attention 的 TND Triton 前反向；top-k 使用设备原生 `torch.topk`。
  attention backward 使用 k2q CSR 组织 dK/dV，不通过公共路径的浮点 atomic 累加。
* `tests/reference_fp64.py`：独立 CPU FP64 前反向基准。
* [文档索引](docs/README.md)：[知识与设计说明](docs/knowledge/)、
  [调试、审查与报告](docs/debugging/)。

2026-09-28：修复后的 conda `veomni` CPU Triton interpreter 完整测试
**138 项通过，无失败或跳过**；另有 31 个 CUDA sm80 离线编译变体通过。
这些结果不代表 NPU 验收；详细精度对照、原生 eager 差异及设备限制见验证报告。

用户提供的首轮 NPU 结果为 **90 passed / 48 failed**，包含 attention KV
backward 编译崩溃和 score 非确定性非法地址访问后的级联失败。
2026-09-29 回传的 `8649b73` 候选复测仍失败：attention 3/3 次编译崩溃，
score blocking 10/10、non-blocking 10/10 次设备异常。B/C/D 未执行。
不能将候选或本机通过记录视为 NPU 缺陷已经修复。
该旧候选的本机回归为 **145 passed，0 failed，0 skipped**，41 个 CUDA sm80
离线编译变体通过，精度阈值仍为 `atol=rtol=1e-4`。

后续六项 NPU 结构对照为 **5 failed / 1 passed**。唯一通过的 tile-sum +
Kahan 在新增消减反例中丢失梯度残差，未采用。新 attention 候选改为逐
CSR query 的向量 TwoSum 累加，保留概率精化、接口和精度门禁。用户于
2026-09-29 明确确认 `df31a1d` 的 **48 项 attention NPU 测试全部通过**。
详细过程见[完整调试复盘](docs/debugging/debugging_retrospective.md)。score 非法地址
故障仍未解决，串行 query 的性能代价亦待实测。
新候选的本机 CPU interpreter 回归为 **157 passed，0 failed，0 skipped**，
41 个 CUDA sm80 离线编译变体通过；均不包含 NPU 编译或执行。

## 使用

从项目父目录（含 `msa_triton/` 的目录）运行，避免子目录名 `triton` 遮蔽第三方包。

```bash
git clone https://github.com/yeyeyeping/msa_triton.git msa_triton
# 保持在当前父目录运行下面的 Python 和测试命令。
```

```python
from msa_triton.triton import m3_index_score, m3_topk, m3_sparse_attention
# 参考实现使用相同签名：from msa_triton.eager import ...

scores = m3_index_score(index_q, index_k, cu_seqlens, max_seqlen,
                       block_size=128, scale=1.0)
indices = m3_topk(scores, cu_seqlens, block_size=128,
                 topk_blocks=16, local_blocks=1)
out = m3_sparse_attention(q, k, v, indices, cu_seqlens, max_seqlen,
                          block_size=128)
out.backward(grad_out)
```

Q/K/V 必须已经完成 norm/RoPE。`cu_seqlens` 是同设备 int32；输入、输出和
梯度为 TND。Indexer group 数对应主 KV heads；同组 query heads 连续排列。
block indices 为序列内编号。公开 API 不选择后端，也不自动 fallback。

## CPU 测试

```bash
# 在已初始化 Conda 的 shell 中，从 msa_triton 的父目录执行：
conda activate veomni
python -m pip install -r msa_triton/requirements-cpu.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pytest msa_triton/tests/test_eager.py msa_triton/tests/test_topk.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_INTERPRET=1 python -m pytest msa_triton/tests -q -s
```

CPU 解释器执行实际 Triton kernel，BF16 输入在调用前精确扩展为 FP32。
此路径没有使用 PyTorch attention 冒充 kernel。BF16 输出和梯度在末端还原；
不验证设备编译、Ascend 算术细节或性能；私有 atomic reference 的并发行为
也不能由解释器证明。

## NPU 验收

在兼容的 CANN、torch_npu、triton-ascend 环境中运行；不要安装上面的上游
`triton` CPU 依赖覆盖 `triton-ascend`。记录完整软件版本后执行：

```bash
unset TRITON_INTERPRET
MSA_TEST_DEVICE=npu python -m pytest msa_triton/tests -q -s -x
python -m msa_triton.benchmark --device npu --lengths 8192 16384 32768 --backward
```

Benchmark 分别记录 score、top-k、k2q CSR 构建、attention 和组合前向耗时；
`--backward` 额外记录 score/attention 的前向加反向耗时，后者包含 CSR 构建。
attention 的 [48 项复测](docs/debugging/npu_attention_streaming_retest.md)已由用户确认
通过；仍须解决 score、完成全量数值验收，再进行长序列性能测试。

BF16 native eager 会在 QK 和 softmax 概率处舍入，与一次 FP64 计算最后才舍入
并不等价。因此测试分别验证 native BSND 适配一致性、Triton 对 FP64 的严格
精度，并报告 native BF16 与 Triton 的差异，不隐藏基准本身的误差。
