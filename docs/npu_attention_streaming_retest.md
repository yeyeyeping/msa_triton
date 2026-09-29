# 当前任务：attention 逐 query 补偿累加候选复测

交给 NPU 机器上的 GLM 5.2 执行。本轮只测试正式 attention 路径；score 的
507035 非法地址故障仍未解决。**全部通过也只代表下述 attention 用例通过，
不能判定整个 MSA 可用。** 不需要再次提交大段 IR 或重跑历史六项消融。

## 改动及依据

`6d47514` 的六项 NPU 对照为 5 failed / 1 passed：独立树编译失败，原全梯度、
仅 QK/QV、拆分 DK/DV launch 均未通过，tile-sum + Kahan 的 D=128 样例通过。
这支持绕开原 32 行补偿树继续验证，但没有定位具体失败 op，也没有证明 UB
容量不足。

不能直接采用 tile-sum + Kahan：新增 CPU FP64 反例中，正确梯度残差
`2^-12 = 0.000244140625` 被普通 tile sum 消去，跨 tile 补偿无法恢复。
该消融在新增的 12 项测试中全部失败，超过原 `atol=rtol=1e-4` 门禁。

正式候选改为逐 CSR query、逐同组 head 累加 `[D]` 梯度向量，使用 FP32
TwoSum 高低两部分并关闭该 KV kernel 的浮点融合。保留 center/mass 精化、
独立 dQ/dK/dV、int64 寻址以及公共 TND 接口。减少同时存活的张量并非保证
Ascend 编译成功；串行 query 循环可能降低性能，须在正确性通过后实测。

## 执行

沿用此前可以访问物理卡 15 的 sudo 权限、Python/CANN 环境。不重装依赖。
在父目录更新已有 `fix/npu-portability` 分支（保留远端已有修改与结果；
有源码冲突时停止并回报）。先确认当前分支为 `fix/npu-portability`，所有
已跟踪源码无本地修改，再运行下方命令；不要把更新合并到其他分支：

```bash
cd /mnt/share/y00977881/project
git -C msa_triton pull --ff-only origin fix/npu-portability
source /mnt/share/y00977881/cann/Ascend/cann/set_env.sh
unset TRITON_INTERPRET TRITON_KERNEL_OVERRIDE PYTEST_ADDOPTS
export ASCEND_RT_VISIBLE_DEVICES=15 MSA_TEST_DEVICE=npu ASCEND_LAUNCH_BLOCKING=1
MSA_ATT_RUN=$(mktemp -d /tmp/msa-att-stream.XXXXXX)
export TRITON_CACHE_DIR="$MSA_ATT_RUN/cache"
export TRITON_DUMP_DIR="$MSA_ATT_RUN/ir"
export TRITON_KERNEL_DUMP=1
git -C msa_triton rev-parse HEAD > "$MSA_ATT_RUN/source.txt"
git -C msa_triton branch --show-current >> "$MSA_ATT_RUN/source.txt"
git -C msa_triton status --short >> "$MSA_ATT_RUN/source.txt"
sha256sum msa_triton/triton/sparse_attention.py \
  msa_triton/tests/test_attention_reduction.py \
  msa_triton/tests/test_triton_attention.py >> "$MSA_ATT_RUN/source.txt"
```

核对三个候选文件的 SHA256；不匹配时停止并回报实际值：

| 文件（相对 msa_triton） | SHA256 |
|---|---|
| `triton/sparse_attention.py` | `2d168f9e1f0dc5ff80de499c9a39697d3649736e80f29fa5589192720f57adaa` |
| `tests/test_attention_reduction.py` | `a118f037e7895361733fe22ba1f6f5b18057d66af6788fd711672d8a3e73d628` |
| `tests/test_triton_attention.py` | `aad431d4ea82fe48d0af3819783258b8269e80f1f2c9cfda8e2b8ea10b513eda` |

保存实际导入来源和版本；本步骤失败先回报，不继续正式测试：

```bash
/mnt/share/y00977881/env/veomni_m3/bin/python - <<'PY' > "$MSA_ATT_RUN/environment.json"
import json, sys
from pathlib import Path
from importlib.metadata import version
import torch, torch_npu, triton, msa_triton
package = Path(msa_triton.__file__).resolve()
assert package.parent == Path('msa_triton').resolve(), package
assert 'site-packages' in triton.__file__, triton.__file__
assert torch.npu.is_available(), 'NPU unavailable'
print(json.dumps(dict(python=sys.version, executable=sys.executable,
    package=str(package), triton_path=triton.__file__, torch=torch.__version__,
    torch_npu=torch_npu.__version__, triton_runtime=triton.__version__,
    triton_distribution=version('triton'), triton_ascend=version('triton-ascend'),
    device=torch.npu.get_device_name(0)), indent=2))
PY
```

用下面命令收集，预期 **48 tests collected**，收集失败/数量不符则先回报：

```bash
/mnt/share/y00977881/env/veomni_m3/bin/python -m pytest \
  msa_triton/tests/test_attention_reduction.py \
  msa_triton/tests/test_triton_attention.py --collect-only -q -o addopts= \
  > "$MSA_ATT_RUN/collection.log" 2>&1
```

正式运行保留输出在远端。`-x` 在首个失败后停止；发生设备异常时不在该进程
继续测试。用执行代理的超时设为 15 分钟，超时要单列，不能当作通过：

```bash
/mnt/share/y00977881/env/veomni_m3/bin/python -m pytest \
  msa_triton/tests/test_attention_reduction.py \
  msa_triton/tests/test_triton_attention.py -q -x -o addopts= \
  --junitxml="$MSA_ATT_RUN/results.xml" > "$MSA_ATT_RUN/pytest.log" 2>&1
MSA_ATT_EXIT=$?
printf 'OUTPUT=%s\nEXIT=%s\n' "$MSA_ATT_RUN" "$MSA_ATT_EXIT"
tail -n 8 "$MSA_ATT_RUN/pytest.log"
```

48 项包括 4 个实际生产累加 helper 用例和 44 个 attention 用例，包含原 D=128
失败节点、不同长度/GQA/独立梯度、长序列 BF16 舍入及新增 12 个消减回归。
helper 使用独立 `math.fsum`，所选样例的 hi+lo 误差门禁为 `1e-12`；公开
attention 前反向仍对齐 CPU FP64 oracle，`atol=rtol=1e-4`，没有放宽阈值。
旧故障树只保留在显式诊断文件中，不在这 48 项正式验收用例里。

## 只需回传这些内容

```text
Git SHA: ...；分支: ...；被测源码是否修改: ...；三个哈希是否匹配: ...
来源/环境: 与上一轮一致，或列出实际变化
结果: passed=... failed=... skipped=... timeout=...，退出码=...
首个失败 node: ...（全部通过填无）
阶段和错误首行: ...（如 PlanMemory / 507035 / 精度断言）
结果目录: /tmp/msa-att-stream....
```

有 skip 或只完成部分用例，不算 48 项通过。若精度失败，仅补充张量名、
最大绝对/相对误差和失败坐标的 actual/expected；不要改阈值或重跑全 MSA。
若编译/执行失败，保留完整日志与缓存，先回传上述短摘要即可。

可选七项 legacy/current 结构诊断仍由 `probe_attention_scope` 提供，历史
`legacy-*` 可能故意复现编译失败，tile-sum 对照也不满足新增精度回归。
本轮不要以该诊断的总 `Status` 代替正式 48 项结果。
