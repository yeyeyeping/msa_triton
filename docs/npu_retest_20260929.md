# NPU 候选复测失败与下一步取证

## 当前状态

**候选未解决两类 NPU 故障，NPU 验收仍然失败。** 本文根据用户于
2026-09-29 回传的报告整理；随后已收到两条原始异常链，IR 和缓存仍未取得。
以下结果来自用户实机执行，不是本机复现。

用户无法传输大量文本；当前优先执行
[只回传短结果的 attention 离线编译对照](npu_attention_compile_probe.md)。
下方完整取证清单保留为参考，不要求此时全部回传。

后续离线对照回传：原参数和仅关闭 auto multi-buffer 均为 returncode=-6，
同一 `PlanMemory Traverse IR Failed`。该选项不能单独绕过错误，下一步
采集崩溃前 pass 的局部 IR，生产内核尚未产生新的修复候选。

本次测试针对 `fix/npu-portability` 的
`8649b7377a00e2a512cd1b333b2385c7e274a2ed`，内核改动提交为 `ebf85cd`。
四个核心文件 SHA256 全部匹配；报告称已跟踪源码未改动，只有测试结果
`runs/` 未跟踪。该目录尚未出现在本机工作树。

| 环境项 | 用户报告 |
|---|---|
| 设备 | Ascend910_9382；物理卡 15，逻辑卡 0；sudo |
| Python | 3.12.0，conda `veomni_m3` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| Triton | runtime 3.2.0 / distribution 3.5.0；triton-ascend 3.2.2 |
| CANN | 9.1.0，V100R001C11SPC001B243 |
| 驱动 / 固件 | 26.0.rc1 / 9.0.0.0.205 |
| 源码 | `/mnt/share/y00977881/project/msa_triton/__init__.py` |
| 第三方 Triton | `/mnt/share/y00977881/env/veomni_m3/lib/python3.12/site-packages/triton/__init__.py` |
| bishengir SHA256 | `89655a56941efe9a184e4d5dfccb783ad88458707827ff6fdd146e7bd2d4af5c` |

runtime 与 distribution 的版本差异需要保留，不能仅凭这项差异断定环境
损坏或要求重装。实际导入路径已排除本地 `triton/` 遮蔽。

| 阶段 | 通过 | 失败 | 退出码 | 秒 |
|---|---:|---:|---:|---:|
| A-attention，独立进程重复 | 0 | 3 | 1 | 110 |
| A-score，blocking | 0 | 10 | 1 | 221 |
| A-score-normal，non-blocking | 0 | 10 | 1 | 228 |

B/C/D 未执行，符合遇到算子失败后停止大范围测试的要求。上述是 **23 次
重复执行失败**，不是 23 个不同缺陷，也不是完整 145 项的新结果。
score 每组 10 次按执行文档覆盖两个节点各 5 次；逐节点明细仍以尚未取得的
`summary.json` 为准，不将整组 10 次全部归到报告列出的首个节点。

## 两类故障的判断

### Attention：编译器内存规划阶段失败

节点为
`test_triton_attention.py::test_forward_backward_against_fp64[lengths6-2-2-128-4-2-None]`。
在 `_backward_kv_kernel` 编译时得到 SIGABRT / returncode=-6，并新增了关键诊断：

```text
LLVM ERROR: PlanMemory Traverse IR Failed!
```

原先将相邻行 `gather` 改为 `reshape → permute → split` 没有解决该失败。
目前不能继续把故障定位为某个 gather 表达式。

公开的 [AscendNPU-IR PlanMemory.cpp](https://github.com/Ascend/AscendNPU-IR/blob/master/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp)
在 IR 遍历被中断时发出上述消息，相关路径包括本地分配检查失败、无法处理的
操作访问本地 buffer。因此这条消息本身**不能证明 UB 容量不足**；优先需要
完整 stderr 中它之前的诊断和失败 pass 的 IR。后续收到的异常链确认
stderr 没有前置诊断，也未包含失败 op，下一步改用同 IR 离线编译对照。
该公开源码并非已确认与报告
中的编译器二进制一致，只作为查找线索。

### Score：地址候选改动未解决设备异常

首个报告节点为
`test_triton_score.py::test_score_forward_backward_fp64[lengths2-2-7-4-1.0]`，
`lengths=[0,3,0,2,0], G=2, D=7, block_size=4`。
forward 后同步失败，错误 507035，包含 scalar GM 地址超过 48 位及非法 GM
访问信息。blocking 和 non-blocking 两组本次重复全部失败。

直接归约序列边界及无效 lane 地址钳制均未消除异常。静态检查暂未发现这两个
指定 shape 的 Q/K/CU 源码越界；这不等于证明设备编译产物正确。日志尚不能
区分边界加载、Q/K 加载、输出写入、归约降低或运行时参数传递中的问题。
不能仅由 scalar GM 字样断定是 CU 标量加载，也不能把同步模式当作修复。

## 交给 GLM 5.2 的下一步任务：交付已有证据

**当前不重跑 A/B/C/D，不修改 kernel、精度阈值、编译选项或依赖版本。**
先完成原结果包的交付；此前已经采集的文件不需要重新生成。

1. 定位本次 `8649b73` 的结果目录，回报它的实际绝对路径，以及已有压缩包的
   路径、字节数和 SHA256。仅给出远端路径不代表本机已经可以读取；由用户转交
   文件或提供可访问位置。不要擅自上传至新的第三方服务。
2. 优先保留 `exception-chain.jsonl`、`failure-ir-attention.log`、
   `failure-ir-score.log`、两个 `failure-ir-*` 目录及其对应缓存目录，保留
   原始文件名、相对目录和缓存 JSON。附上三组 A 阶段 summary、逐 case 日志、
   `environment.json`、阶段命令和 `stages.tsv`。
3. 新建一份 `ARTIFACT_INDEX.md`，为 `_backward_kv_kernel` 与
   `_index_score_forward` 分别列出下表。前向和 query backward 的产物分别列出，
   不和失败的 KV kernel 混在一起。

| kernel / specialization | 对应 node / PID / 时间 | cache key / metadata 路径 | TTIR 路径 | ttadapter 路径 | 失败 pass IR 路径 | 二进制路径 | 文件 SHA256 |
|---|---|---|---|---|---|---|---|
| `_backward_kv_kernel`, D=128 | 依据日志 | 有则填 | 有则填 | 有则填 | 有则填 | 有则填 | 对每个文件记录 |
| `_index_score_forward`, D=7 | 依据日志 | 有则填 | 有则填 | 有则填 | 有则填 | 有则填 | 对每个文件记录 |

不存在的产物填写“未生成/未采集”，无法匹配填写“归属未确认”。
**目录中存在 `.npubin` 不等于崩溃的 KV specialization 已成功编译**；它可能
属于 forward、query backward、其他参数或旧缓存。不要重命名这些文件来推定
归属。`ttadapter` 是后续编译的输入，不一定是 `PlanMemory` 报错时的 IR。

4. 从完整异常链提取 attention 的 `CalledProcessError.cmd`、returncode、
   stdout/stderr。单独保存未经截断的 stderr，包括 fatal 行之前的内容。
   已有 `--mlir-print-ir-after-failure` 输出也要保留。若没有更具体诊断，明确
   记录“原始 stderr 无前置诊断”，不要补猜错误 op。
5. 对 score 保留第一个设备错误对应的 PID/时间/core/PC，以及同一调用的
   kernel/cache 元数据。不要在失败进程中读取设备 tensor 来补充诊断。
6. 如果原包未包含安装后端源码，额外复制**实际导入的** Triton 目录下
   `backends/ascend/compiler.py`、`utils.py`、`driver.py`，保留来源路径与哈希；
   另附相关 distribution 的 `METADATA`，存在 `direct_url.json` 则一并保存。
   缺失文件如实记录。这是用于核对本地编译选项和启动参数的只读采集，不要求
   导入模型、初始化设备、重装环境或复制整个 site-packages。

新增索引/后端文件应放入新的补充目录，保留原结果包。结果包较大时先回传
异常链、两个完整失败日志、IR/cache JSON 和索引；二进制可随后提供，但记录
路径、大小和哈希。若原压缩包已含所有必要内容，直接交付它即可。

## 收到原始证据后的排查顺序

先核对失败调用与产物映射，再确定最小实验；以下是判断分支，不是当前的
设备执行命令：

- attention：查出内存规划无法处理的具体 op/分配和所在 pass；必要时用同份
  ttadapter 与相同编译器做不运行设备的最小编译复现，再设计单变量实验。
  不以移除 center/mass 精化或普通 FP32 sum 作为默认修复。
- score：逐项核对输入/输出指针签名、int64 地址降低、mask/control flow、
  chained gather 与补偿点积的 IR。若仍无法定位，再隔离“边界与地址计算”、
  “实际加载/写入”、“补偿归约”三个部分；单个 gather 曾成功不能证明组合
  kernel 正确，也不能证明 gather 就是原因。
- 只有有证据支持的新改动才进入下一轮 NPU 小范围复测，随后恢复原数值门禁。

本次仓库更新只记录失败状态和取证要求，没有新的内核修复或新的设备通过
记录。此前 CPU interpreter 145 项、CUDA 离线编译 41 变体仍为历史回归证据。
