# NPU 首轮失败报告与复测候选

## 状态与证据来源

**NPU 验收未通过；本轮改动是待实机复测的候选，不能标记为已修复。**

用户于 2026-09-28 提供了 NPU 测试报告及错误堆栈。下表来自用户报告，
不是本机执行结果；报告中的 `/tmp` 日志、最小复现脚本和失败 IR 不在本机。
候选基于仓库提交 `0dc8eee`，位于 `fix/npu-portability` 分支。原报告未附
设备端 Git SHA，后续复测应一并记录源码版本。

| 项目 | 用户报告 |
|---|---|
| 设备 | Ascend910_9382，`ASCEND_RT_VISIBLE_DEVICES=15`，chipId:7 / dieId:1 |
| 环境 | `veomni_m3`，Python 3.12 |
| 软件 | torch 2.10.0+cpu、torch_npu 2.10.0、triton 3.5.0 / triton-ascend 3.2.2 |
| CANN | 已 source `set_env.sh`，具体版本待补充 |
| 权限 | 用户报告需要 sudo 访问 NPU；复测沿用已获授权的设备环境 |
| 总结果 | 138 项：90 passed / 48 failed，419.39 秒 |

| 文件 | 通过 | 失败 |
|---|---:|---:|
| `test_eager.py` | 27 | 0 |
| `test_index_addressing.py` | 3 | 0 |
| `test_k2q.py` | 11 | 0 |
| `test_pipeline.py` | 4 | 0 |
| `test_topk.py` | 12 | 0 |
| `test_triton_attention.py` | 31 | 1 |
| `test_triton_score.py` | 2 | 47 |

`test_eager.py` 的 BSND/FP64 测试固定在 CPU，不能把全部 90 个 passed
都解释为 NPU kernel 验证通过。score 单文件结果为 47 failed / 2 passed，
15.66 秒；首次设备错误后的普通 tensor 操作也失败，存在明显级联。

## 两个问题与当前判断

### A. attention KV backward 编译崩溃

稳定触发节点：

```text
test_triton_attention.py::test_forward_backward_against_fp64[lengths6-2-2-128-4-2-None]
lengths=[5,3], G=2, Hq/G=2, D=128, block_size=4, topk=2
```

前向及其断言通过，`actual.backward()` 中的 `_backward_kv_kernel` 编译
失败。`bishengir-compile --target=Ascend910_9382` 以 SIGABRT / returncode=-6
退出，Triton 报 `MLIRCompilationError: [ConvertLinalgRToBinary]`。
无符号堆栈还不能定位具体 LLVM pass 或源代码表达式。

候选把 KV 归约的相邻行 `tl.gather` 改为固定结构的
`reshape → permute → split`，去掉每层生成的 gather 索引。保持五层配对
顺序、TwoSum 的高低分量运算、center/mass 精化、Q/K/V 梯度开关和 k2q
所有权不变。它没有采用之前失败的“删除中心精化”消融，也没有改成普通
FP32 sum。

[Ascend v3.2.2 split 文档](https://triton-ascend.readthedocs.io/zh-cn/v3.2.2/python-api/generated/triton.language.split.html)
支持末维为 2 的 FP32 tensor 拆分；这说明候选使用了目标版本文档中的接口，
并不证明该编译崩溃已经消失。新增四个测试精确比较整棵归约树的高低分量。

### B. score forward 非确定性非法 GM 访问

首个报告节点是 `[0,3,0,2,0] / G=2 / D=7 / block=4 / scale=1`，
执行 score forward 后同步失败，尚未进入 backward。设备报错包括：

```text
ACL stream synchronize failed: 507035
errcode (0, 0x4000, 0): GM address accessed by scalar exceeds 48 bits
errcode (0, 0x8000, 0): scalar instruction accesses an invalid GM address
retCode=0x31: vector core exception
```

报告中同 shape 多次执行有成功也有失败；同进程连续切换 shape 可在后续
case 才崩溃。dtype、是否求导、单独的 gather 测试都未形成稳定关联。
因此不能把 47 个失败视为 47 个独立 score 缺陷，更不能用上下文污染后
`q.to(device)` 或 metadata 测试失败来定位新的 kernel 问题。

静态检查没有发现原 `count(starts <= token)-1` 在合法 CU、空序列情况下
计算错误。[官方 load 契约](https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.load.html)
规定 false mask 不读取相应地址；原 masked lane 的地址表达式越过分配范围，
本身不足以证明违反 Triton 语义。此外，Ascend 会将部分不规则张量访存
降低为标量访问，见[编译架构说明](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)。
所以设备日志中的 scalar GM 也不能单独把故障归因到 CU 的标量加载。

本轮候选做两项保守简化：

1. 连续加载 CU 的 starts/ends，直接归约活动序列的 begin/end，移除
   “归约产生 seq_id，再通过它间接加载 CU”的路径。连续空序列仍合法。
2. 在构造指针前钳制无效 seq/key/query/dim/group lane，保留原 mask 和
   `other=0` 等数值语义。维持 int64 地址乘加，不退回 int32。

没有同时更改 grid、补偿点积、scale、ties、公开 dtype 或阈值，也没有
新增同步、自动 fallback 或 NPU 专用 backend 参数。若复测仍失败，需要
失败 IR 与进一步拆分实验，不能继续靠猜测扩大改动。

## 复测步骤

从包的父目录运行；若通过 zip 下载，继续保证导入路径名为 `msa_triton`。
使用 `fix/npu-portability` 分支，记录 SHA。沿用原有 CANN、Conda 和设备
访问权限，不用本机的上游 Triton 依赖覆盖 triton-ascend。

```bash
export ASCEND_RT_VISIBLE_DEVICES=15 MSA_TEST_DEVICE=npu
unset TRITON_INTERPRET
git -C msa_triton rev-parse HEAD
```

先在独立进程中重复两个首发问题：

```bash
python -m msa_triton.tests.probe_npu_isolated \
  'msa_triton/tests/test_triton_attention.py::test_forward_backward_against_fp64[lengths6-2-2-128-4-2-None]' \
  --repeat 3 --output-dir /tmp/msa-attention-candidate

python -m msa_triton.tests.probe_npu_isolated \
  'msa_triton/tests/test_triton_score.py::test_score_forward_backward_fp64[lengths2-2-7-4-1.0]' \
  'msa_triton/tests/test_triton_score.py::test_score_forward_backward_fp64[lengths3-2-7-4-1.0]' \
  --repeat 5 --output-dir /tmp/msa-score-first-candidate
```

`probe_npu_isolated` 为每次执行启动新 Python 进程，默认设置
`ASCEND_LAUNCH_BLOCKING=1`，保留可见设备配置。每个 case 记录完整命令、
退出码、耗时、日志和 pytest JSON，增量写入 `summary.json`。输出目录必须
是新目录；重复运行时换名。skip、xfail、timeout、无测试或崩溃均不计通过。
新进程隔离 runtime 状态，不能保证恢复卡级硬件故障。

随后分别检查同进程累计运行与正常异步执行：

```bash
ASCEND_LAUNCH_BLOCKING=1 python -m msa_triton.tests.probe_score_sequence \
  --device npu --repeats 20 > /tmp/msa-score-sequence-blocking.log 2>&1

unset ASCEND_LAUNCH_BLOCKING
python -m msa_triton.tests.probe_score_sequence \
  --device npu --repeats 20 > /tmp/msa-score-sequence.log 2>&1

python -m msa_triton.tests.probe_score_sequence \
  --device npu --repeats 20 --forward-only > /tmp/msa-score-forward.log 2>&1
```

该工具按原测试顺序循环七种 shape，首轮使用原 seed=731。每次 transfer、
forward、backward 后显式同步并与同一输入的 CPU FP64 对齐，在动作开始前
刷新阶段日志；第一次失败立即退出，不再向污染后的上下文提交下一 case。
“正常异步”指 kernel launch 未设置全局阻塞；阶段间仍有工具的显式同步，
它不代替多 stream/无同步并发测试。可另加 `--dtype FP32` 或 `--dtype FP16`。

首发用例通过后再执行 score 的隔离全量与全部算子回归：

```bash
python -m msa_triton.tests.probe_npu_isolated \
  msa_triton/tests/test_triton_score.py --output-dir /tmp/msa-score-all-candidate

python -m pytest msa_triton/tests -q -s -x
```

全量使用 `-x`，首次失败即停，避免再次生成大量级联失败。没有跳过且全部
严格断言通过才可更新验收状态；单次成功不足以排除报告中的非确定性问题。

## 若仍失败，保留编译现场

只对失败的单个节点启用 IR 转储，并使用独立的新缓存目录：

```bash
TRITON_CACHE_DIR=/tmp/msa-att-cache-candidate \
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/msa-att-ir-candidate \
MLIR_ENABLE_DUMP=1 \
python -m msa_triton.tests.probe_npu_isolated \
  'msa_triton/tests/test_triton_attention.py::test_forward_backward_against_fp64[lengths6-2-2-128-4-2-None]' \
  --output-dir /tmp/msa-att-compile-candidate
```

这些调试变量来自 [v3.2.2 环境与编译选项文档](https://triton-ascend.readthedocs.io/zh-cn/v3.2.2/environment_variable_and_compiler_options_reference.html)。
score 故障时把目标节点换成其首发节点。回传独立日志、`summary.json`、失败
之前生成的 IR、具体 CANN 版本和 Git SHA；没有生成的后续 IR 不应伪造。
本轮没有向上游发送 issue，也没有声称命中已知编译器缺陷。

## 本机验证

本机依然没有 NPU。CPU 与离线编译检查仅用于验证候选没有破坏已有语义，
不能替代上述实机复测。详细执行结果见 [validation.md](validation.md)。

- 完整 CPU interpreter：145 passed，0 failed，0 skipped，219.27 秒。
- 41 个 CUDA sm80 离线编译变体通过，包含报告中的具体特化参数。
- 同进程诊断：BF16 前后向、FP32 forward-only 各完成一轮七种 shape。
- 保持 `atol=rtol=1e-4`；新增边界测试和补偿树高低分量精确测试。
