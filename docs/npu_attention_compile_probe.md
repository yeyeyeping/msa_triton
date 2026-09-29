# Attention 编译崩溃：仅回传短结果的离线对照

> 六项结构对照已完成：5 failed / 1 passed。正式实现现已产生逐 query
> TwoSum 累加候选，**当前执行 [48 项 attention 候选复测](npu_attention_streaming_retest.md)**。
> 本文保留取证历史，不再重复 A/B、pass dump 或旧六项对照。

## 六项对照结果与新候选

用户回传源码 `6d4751489a0fe0e1140f42dbd5571b07537dabbe`：

| 历史用例 | NPU 结果 |
|---|---|
| tree-d128 | compile_error；独立树不能编译，摘要未给出具体错误类型 |
| qkv-d128 / qk-d128 / qv-d128 | KV launch 的 compile_plan_memory |
| split-kv-d128 | 第一次 DK launch 的 compile_plan_memory |
| tile-sum-kahan-d128 | 完成该样例的数值门禁 |

仅拆分 DK/DV 无效。移除行树的对照通过，为绕开该 IR 结构提供依据，但不能
把树的独立 compile_error 等同于已证明的同一 PlanMemory 根因。进一步 CPU
反例发现 tile-sum + Kahan 会丢失 `2^-12` 的 dK/dV 残差，不能作为正式修复。
新候选逐 query 累加 `[D]` 向量并保留 TwoSum；NPU 仍待验证。

## 已确认的 pass 信息与尚未成立的内存判断

用户回传 `IR_MATCH=confirmed`、`pass_trace_captured_original_failure`，
编译返回码 -6。共打印 235 个 before-pass 标题，最后五个依次为：

```text
canonicalize-ext
memref-dse
hivm-inline-load-copy
hivm-mark-multi-buffer
hivm-plan-memory
```

据回传摘要，最后的 IR 为 2110 行 module，唯一函数是具有 18 个参数的
`_backward_kv_kernel`，AIV core；无 op 级 error/note/warning。这确认了
崩溃 pass，但本机仍没有完整 IR，不能据尾部几行定位出错操作。

报告中的“1953 个 address_space<ub> 操作（UB 分配）”需要区分统计口径：
该地址空间类型也会出现在同一缓冲的 view、运算参数、load/store 中，
出现次数不等于 `memref.alloc` 定义数，更不等于同时存活的 UB 字节数。
`multi_buffer=2` 标记也不能单独证明容量不足。

当前上游 [PlanMemory.cpp](https://github.com/Ascend/AscendNPU-IR/blob/master/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp)
在 IR 遍历被中断时产生这个 fatal；相关检查包括分配的地址空间及无法识别的
本地 buffer 操作，并非直接报告容量耗尽。公开源码不能替代实际二进制的
定位，但“内存规划 pass 失败”不应直接写成“UB 内存不足”。先前关闭
multi-buffer 仍失败，也不能支持“仅双缓冲容量导致”的解释。

无需为修正这个判断额外回传计数或重跑 dump。六项结果见文首；生产候选
依据结构对照与数值反例产生，并不假定已经证明 UB 容量不足。

## 对最新 IR 片段的判断

`memref<128xf32, #hivm.address_space<ub>>` 的单个数据缓冲为 512 字节，
该大小本身不能说明整个内核的 UB 峰值。`vadd` 后 `store` 与生产 KV 内核
最后的 `high + low` 写回形式一致；缺少 `%34` 定义和目标 GM 指针来源，
目前不能确认它具体对应 dK 还是 dV，也不能认定这就是失败 op。

`annotation.mark {hivm.multi_buffer = 2}` 不表示关闭开关的对照失效：
`--dump-pass-ir` 按协议使用 multi-buffer=True 的 baseline。
`autoblockify.subloop` 只是片段中看到的循环属性，不足以认定 auto-blockify
是根因。当前上游 [PlanMemory.cpp](https://github.com/Ascend/AscendNPU-IR/blob/master/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp)
明确处理本地 alloc 与 annotation mark，实际安装版本仍须区别对待。
仅截取日志末尾不能替代失败位置诊断。

## 已有取证步骤：一次离线编译，提取最后一个 pass 的 IR

本节保留命令背景，当前不用再次执行。

沿用刚才已成功运行的 `--exception-chain`、`--ir`、`--ir-sha256`、`--cwd`
参数，更新分支脚本后，**只增加 `--dump-pass-ir`**。若之前手动指定了
`--output-dir`，这次换一个新目录或省略该参数，不能覆盖旧结果。

```bash
/mnt/share/y00977881/env/veomni_m3/bin/python \
  msa_triton/tests/probe_attention_compile_replay.py \
  --exception-chain /实际结果目录/exception-chain.jsonl \
  --ir /刚才重现失败的同一份ttadapter.mlir \
  --ir-sha256 刚才同一个IR的SHA256 \
  --cwd /mnt/share/y00977881/project \
  --dump-pass-ir
```

该模式只编译一次，恢复原 baseline 的 multi-buffer=True，唯一新增参数为
`--mlir-print-ir-before-all`，不加载或运行生成的 NPU 二进制。这个调试参数
也记录在[Ascend PyTorch 的编译配置说明](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/ascend_npu_ir/config.py)
中，但具体安装版本是否支持仍以执行结果为准。

完整输出保留在远端；脚本从日志流式提取最后一段 before-pass IR 到
`pass_trace/last-before.mlir`，并仅打印最后几个 pass 标题与错误摘要。

交给 GLM 的回传要求：

1. 先回传脚本的短摘要。退出码 0 在此模式仅表示成功采集到同一崩溃前的 IR，
   不表示编译通过。
2. 如果存在 `last-before.mlir`，在远端读取它与完整 stderr。若有指向某个
   op 的具体诊断，仅附该 op 前后约 8 行；若没有，明确写“无 op 级诊断”。
   不要把日志最后一个 op 当作出错 op。
3. 若最后一份 IR 是整个 module，列出其中的函数名及 `_backward_kv_kernel`
   对应位置；不要直接回传整个 module。若 pass 标题不含 PlanMemory，保留
   实际标题，不猜测它就是报错 pass。
4. 不支持参数、没有 dump 或遇到不同错误时，回传短状态和错误首行即可。
   暂不自动升级环境、添加其他开关或继续全量测试。

## 历史任务：六个结构对照（已完成）

当时 IR 片段仍无法定位，因此运行以下工具。每个用例在独立进程中执行，原始
输入与精度门禁保留，源码中的实验修改仅在对应进程生效。

| 用例 | 验证内容 |
|---|---|
| tree-d128 | 32×128 补偿树独立执行，NumPy FP32 hi/lo 逐位对照 |
| qkv-d128 | 原 D=128 全梯度失败节点 |
| qk-d128 / qv-d128 | 保留 dQ，分别请求 dK 或 dV，保持 query 统计数学路径 |
| split-kv-d128 | 共用原 stats，将同一 KV kernel 拆成 dK、dV 两次 launch |
| tile-sum-kahan-d128 | 保留 center/mass，仅在临时源码里改用 tile sum + Kahan |

```bash
source /mnt/share/y00977881/cann/Ascend/cann/set_env.sh
cd /mnt/share/y00977881/project
ASCEND_RT_VISIBLE_DEVICES=15 \
  /mnt/share/y00977881/env/veomni_m3/bin/python \
  -m msa_triton.tests.probe_attention_scope
```

仍从项目父目录和原 CANN 环境运行，只回传控制台摘要。runner 将完整日志、
IR/cache、临时消融源码和 JSON 保存在新的 `/tmp/msa-att-scope-*` 中。
出现设备运行异常时停止后续用例；编译失败可以继续下一个独立进程。
该历史版本常规 pytest 收集数量为 145，六项需显式选中，不作为新增 NPU 验收结果。
沿用此前能访问卡 15 的权限环境。回传六条 `用例: 状态 reason=... STAGE=...`
及最后的 `Status:`；若提前停止，保留 `not_run`，不补猜未执行结果。
已有 pass 标题可以额外附一行，但不用因此重新跑编译或回传大量 IR。

split 对照通过只能支持进一步验证拆分 launch；不能证明是 UB 容量不足。
孤立 tree 通过也不能排除它与循环/周边 IR 组合的问题。Kahan 会改变归约
次序，即使一个 shape 通过，也不能直接替换生产实现。

新版本的诊断 runner 已分为七项：`legacy-tree-d128`、`legacy-qkv-d128`、
`current-qkv-d128`、`current-qk-d128`、`current-qv-d128`、
`current-split-kv-d128`、`tile-sum-kahan-d128`。历史 KV 三个函数固定保存
在 `tests/_attention_tree_reference.py`，不依赖已删除的生产树 helper。
该 runner 供明确需要时比较，不是本轮要求，也不能把 legacy 预期失败计入
正式候选验收。

## 已收到的证据

用户提供的异常链包含两条记录。attention 的 `CalledProcessError.stderr`
直接以 `LLVM ERROR: PlanMemory Traverse IR Failed!` 开头，之后只有未符号化
backtrace；**没有前置诊断或失败 op**。继续抽取同一 stderr 不会得到新信息。
编译器退出码为 -6，原命令明确包含 `--enable-auto-multi-buffer=True`。

score 日志中的故障 PC 相对 `pc start` 为 `0x1fb0` / `0x1fb4`。
这些偏移只适用于对应编译产物，目前没有二进制/反汇编，不能定位源码指令。
本任务只缩小 attention 的编译问题，不运行 score。

## 历史任务：multi-buffer A/B（已完成，不再执行）

目标：在原机器上，使用**同一份已保存 IR 和同一个编译器**做两次离线编译。
第一遍保留原参数；只有重现相同 SIGABRT/fatal，才进行第二遍，将唯一的
`--enable-auto-multi-buffer=True` 改为 `False`。不执行生成的二进制、不使用
设备 tensor、不运行 pytest，不修改安装环境或生产内核。

1. 获取 `fix/npu-portability` 最新分支中的
   `tests/probe_attention_compile_replay.py`。保留远端已有改动和结果目录。
2. 找到上一轮的 `exception-chain.jsonl` 和失败 `_backward_kv_kernel` 对应的
   `ttadapter` 文件。核对它属于以下 specialization：
   `lengths=[5,3], Hq=4, Hkv=2, D=128, block_size=4, topk=2`，dK/dV 同时启用。
   应以原失败日志/缓存的对应关系核对，**文件包含 KV 名称并不足以区分参数**。
   不要使用 forward、query backward 的 IR，也不要将 TTIR 当作 ttadapter。
   若找不到对应文件，只回报 `IR_NOT_IDENTIFIED` 和原因，停止，不重跑整套测试。
3. 记录该 IR 的 SHA256，填入下面命令。脚本核对哈希和 kernel 符号，但不能
   自动证明它就是历史失败的 specialization。原编译器路径从异常链读取，
   必须匹配此前报告的 SHA256；不匹配时停止，不能覆盖预期哈希来继续。
4. 沿用原 CANN 环境。编译本身在 host 执行；仍需原 ARM 机器上的编译器和
   依赖，不是在本机 CPU Triton interpreter 中执行。

```bash
source /mnt/share/y00977881/cann/Ascend/cann/set_env.sh
cd /mnt/share/y00977881/project
/mnt/share/y00977881/env/veomni_m3/bin/python \
  msa_triton/tests/probe_attention_compile_replay.py \
  --exception-chain /实际结果目录/exception-chain.jsonl \
  --ir /实际缓存目录/失败KV对应的ttadapter.mlir \
  --ir-sha256 填入该文件SHA256 \
  --cwd /mnt/share/y00977881/project
```

脚本默认每次编译超时 180 秒，结果写入新的 `/tmp/msa-att-replay-*`
目录；原日志和 IR 保留。若异常链包含多条匹配记录，按脚本提示使用
`--record-index` 明确选择，不拼接不同运行的异常与 IR。

两个实验的环境、输入字节和其他编译选项保持一致，输出分别保存。完整命令、
stdout/stderr、产物和哈希留在远端，**只回传脚本打印的短结果**，并在开头写
`IR_MATCH=confirmed`。不要复制 backtrace 或大段 IR。
如果前置核对失败，只回传脚本错误和原因，不需要传整个结果包。

## 如何解释短结果

| baseline | multi-buffer=False | 结论与下一步 |
|---|---|---|
| 同一 SIGABRT/fatal | 编译成功且存在非空产物 | 该选项影响此次编译；随后才考虑保持数学不变的候选及 NPU 数值复测 |
| 同一 SIGABRT/fatal | 同一失败 | 单独关闭该选项不能绕过故障；下一步采集失败前 pass 的局部 IR |
| 同一 SIGABRT/fatal | 其他错误/超时 | 保留新的错误类型，不能当作编译成功或原问题解决 |
| 成功、其他错误或超时 | 不执行 | baseline 未重现，先核对 IR、依赖和编译环境 |

退出码 0 只表示第二次离线编译成功，**不代表 NPU 执行、数值、反向或 score
通过**。本轮不把 `multibuffer=False` 写入生产内核。

该对照选择来自原始命令中已启用的选项，属于单变量诊断；尚无证据证明
multi-buffer 是根因。背景见[候选复测记录](npu_retest_20260929.md)。

脚本本机验证：conda `veomni` 下使用模拟编译器，10 项检查通过，覆盖原失败
重现与单参数差异、两次均失败、baseline 成功/其他错误时停止、缺少二进制、
仅有 IR 产物、拒绝覆盖已有结果、IR/编译器哈希不匹配、超时清理子进程。
这些是诊断工具测试，不是真实 bishengir 编译或 NPU 数值验证。

后续扩展验证：pass IR 模式另有 8 项模拟检查通过，包括 stderr/stdout 提取、
仅保留最后一个 pass、超过 2 MB 日志不回显、参数不支持、缺少/空 dump、
编译行为变化和其他错误。六个后备结构对照在 CPU interpreter 下全部通过，
其独立进程 runner 的 CPU 六项及故障分类、超时清理检查也通过。
