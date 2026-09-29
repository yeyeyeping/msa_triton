# Attention 编译崩溃：仅回传短结果的离线对照

## 已收到的证据

用户提供的异常链包含两条记录。attention 的 `CalledProcessError.stderr`
直接以 `LLVM ERROR: PlanMemory Traverse IR Failed!` 开头，之后只有未符号化
backtrace；**没有前置诊断或失败 op**。继续抽取同一 stderr 不会得到新信息。
编译器退出码为 -6，原命令明确包含 `--enable-auto-multi-buffer=True`。

score 日志中的故障 PC 相对 `pc start` 为 `0x1fb0` / `0x1fb4`。
这些偏移只适用于对应编译产物，目前没有二进制/反汇编，不能定位源码指令。
本任务只缩小 attention 的编译问题，不运行 score。

## 交给 GLM 5.2 的任务

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
