# 给 NPU 机器上 GLM 5.2 的执行任务

> 2026-09-29：本文针对 `8649b73` 的 A 阶段已执行，两个故障均复现。
> 后续 `df31a1d` 的 [48 项 attention 复测](npu_attention_streaming_retest.md)
> 已由用户确认全部通过；score 仍未解决。不重复下方历史测试。
> 本文保留原协议及旧哈希，用于解释已采集结果；这些哈希不适用于新的
> 逐 query 累加实现。完整过程见[调试复盘](debugging_retrospective.md)。

## 1. 任务与本次改动

请验证 `fix/npu-portability` 分支的 NPU 兼容性候选，采集可复现的证据。
本轮执行范围是测试与诊断；不要自行修改实现、测试、精度阈值或安装版本。
首次失败后的处理按第 6 节执行，最后交付第 7 节的报告与文件包。

内核修改提交：`ebf85cdd7b3ccb5290ec3c3183b44839038775d9`。
后续提交可能只更新本文；必须记录实际 HEAD，并检查下面的核心文件哈希。
背景报告见 [npu_validation_20260928.md](npu_validation_20260928.md)。

| 修改 | 具体行为 | 要验证的问题 |
|---|---|---|
| score 序列边界 | 连续加载 CU 起止值并直接归约 begin/end，去掉用归约得到的 seq_id 再间接读 CU | 空序列 packing 下的边界正确性与设备访存稳定性 |
| score 地址 | 在构造指针前钳制无效 seq/key/query/dim/group lane，前后向均保留原 mask | 非确定性非法 GM 访问是否仍出现 |
| attention KV 归约 | `gather` 相邻行配对改为 `reshape → permute → split` | D=128 的 bishengir 编译崩溃是否消失 |
| 诊断与回归 | 新增逐用例独立进程 runner、同进程七种 shape 循环、3 项边界测试和 4 项补偿树测试 | 分开记录首发故障、进程间重复性及连续调用问题 |

公开接口、TND、int64 地址乘加、score 的补偿点积与 tie 梯度、attention
的五层 TwoSum 和 center/mass 精化均保留。没有删除中心精化、改成普通
FP32 sum、加入 eager fallback，或放宽 `atol=rtol=1e-4`。

**这些是候选改动，不是已获 NPU 验证的修复。** 本机 CPU interpreter
145 项通过、41 个 CUDA 离线编译变体通过，只能作为回归证据。
原报告的 47 个 score 失败包含设备上下文污染后的级联，不能当作 47 个
独立问题；scalar GM 错误也尚未定位到具体源代码加载指令。

## 2. 执行环境和结果目录

沿用原报告可访问卡 15 的 sudo/CANN/Conda 环境。**下面所有检查与测试必须
在实际访问设备的同一权限环境执行**，包括 sudo 后的 Python 路径检查。
不要用另一套 root Python，也不要重装 torch、torch_npu 或 triton。

源码必须是物理目录 `<父目录>/msa_triton`，或指向同一父目录内候选目录的
符号链接。从父目录运行；不要 `cd msa_triton` 后执行 Python，以免本地
`triton/` 遮蔽第三方包。不要覆盖已有工作；必要时在新父目录克隆分支：

```bash
git clone --branch fix/npu-portability https://github.com/yeyeyeping/msa_triton.git msa_triton
```

无 Git 凭据时可使用该分支的 zip，解压并保持可导入名称为 `msa_triton`。
不要误用 GitHub 默认 `main` 分支的压缩包。记录 zip 文件名和 SHA256；
Git SHA 不可得时明确写“zip，无 Git 元数据”，仍须通过核心文件哈希检查。

按实际环境调整前三个路径，然后在 Bash 中逐阶段执行。不要把本文所有
代码块无条件拼接运行：每个阶段结束后检查状态，按分支要求继续或停止。

```bash
export MSA_PROJECT_ROOT=/mnt/share/y00977881/project
export MSA_PYTHON=/mnt/share/y00977881/env/veomni_m3/bin/python
export MSA_CANN_ROOT=/mnt/share/y00977881/cann/Ascend/cann
source "$MSA_CANN_ROOT/set_env.sh"
cd "$MSA_PROJECT_ROOT"
test -x "$MSA_PYTHON"
test -f msa_triton/triton/index_score.py
export ASCEND_RT_VISIBLE_DEVICES=15 MSA_TEST_DEVICE=npu
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset TRITON_INTERPRET ASCEND_LAUNCH_BLOCKING PYTEST_ADDOPTS
export MSA_RUN_DIR
MSA_RUN_DIR=$(mktemp -d /tmp/msa-npu-glm-XXXXXX)
printf 'Results: %s\n' "$MSA_RUN_DIR"
set -o pipefail

# 不通过 tee 的退出码判断测试成功；保存原命令、完整输出和真实退出码。
msa_run() {
    local msa_stage="$1"
    shift
    local msa_start=$SECONDS msa_rc
    {
        printf 'cwd=%s\ncommand=' "$PWD"
        printf '%q ' "$@"
        printf '\n'
    } > "$MSA_RUN_DIR/$msa_stage.command.txt"
    if timeout --signal=TERM --kill-after=30s 1800 "$@" \
        > "$MSA_RUN_DIR/$msa_stage.log" 2>&1; then
        msa_rc=0
    else
        msa_rc=$?
    fi
    printf '%s\t%s\t%s\n' "$msa_stage" "$msa_rc" "$((SECONDS-msa_start))" \
        >> "$MSA_RUN_DIR/stages.tsv"
    printf '%s: exit=%s, log=%s/%s.log\n' "$msa_stage" "$msa_rc" "$MSA_RUN_DIR" "$msa_stage"
    return "$msa_rc"
}
```

可见设备过滤后使用 `npu` / 逻辑卡 0，**不要把测试改成 `npu:15`**。
`msa_run` 的总阶段超时为 1800 秒，隔离 runner 的单子进程超时另设为 600 秒。
超时必须报告为未通过，不可忽略退出码。结果目录放在仓库之外。

## 3. 前置检查与源码记录

先记录 Git/设备信息。zip 模式下 Git 命令失败是元数据缺失，不是算子失败；
保留输出并继续哈希检查。`npu-smi` 不可用或无权限时记录原因，不安装替代工具。

```bash
msa_run git-head git -C msa_triton rev-parse HEAD
msa_run git-status git -C msa_triton status --porcelain
msa_run git-diff git -C msa_triton diff --stat
msa_run npu-smi npu-smi info
```

在结果目录创建并执行以下检查。四个哈希不匹配、导入路径错误、NPU 不可用
或基础设备运算失败时，停止算子测试并报告环境/源码阻塞。

```bash
cat > "$MSA_RUN_DIR/preflight.py" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['MSA_PROJECT_ROOT']).resolve()
sys.path.insert(0, str(root))
package = (root / 'msa_triton').resolve()
result = {'python': sys.executable, 'python_version': sys.version,
          'cwd': str(Path.cwd()), 'package_directory': str(package)}
expected = {
    'triton/index_score.py': 'ebb61070dc9a608308bcc99788f953c12f5b3942e1eb6a4be526d7f0fd11c0c1',
    'triton/sparse_attention.py': '12069b373c33ee10ffa9f4471e3bb67be87a927a8c89551de68b4d8f2c82c1af',
    'tests/probe_npu_isolated.py': 'bdd4546f1307f1d324acac0833e34debf4f5b9b5d8c65e0733edf11c00232002',
    'tests/probe_score_sequence.py': '8807f15f518451c3784ef9de8f37579f1a1ddac248af7caf405d0b08c57efea0',
}
result['source_sha256'] = {
    str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(package.rglob('*.py')) if '.git' not in p.parts
}
result['environment'] = {key: os.environ.get(key) for key in (
    'ASCEND_RT_VISIBLE_DEVICES', 'MSA_TEST_DEVICE', 'TRITON_INTERPRET',
    'ASCEND_LAUNCH_BLOCKING', 'ASCEND_HOME_PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH',
    'TRITON_CACHE_DIR', 'TRITON_KERNEL_OVERRIDE', 'TRITON_OVERRIDE_DIR',
    'TRITON_ALWAYS_COMPILE', 'TRITON_DEFAULT_FP_FUSION',
    'TRITON_ASCEND_COMPILE_SPEED_OPT', 'TRITON_ALL_BLOCKS_PARALLEL')}
result['distributions'] = {}
for name in ('torch', 'torch-npu', 'triton', 'triton-ascend', 'pytest', 'numpy'):
    try:
        result['distributions'][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result['distributions'][name] = None
print(json.dumps(result, indent=2), flush=True)
Path(os.environ['MSA_RUN_DIR'], 'environment.json').write_text(json.dumps(result, indent=2))
assert all(result['source_sha256'].get(p) == sha for p, sha in expected.items()), 'wrong source hashes'
assert os.environ.get('TRITON_INTERPRET') is None
assert os.environ.get('TRITON_KERNEL_OVERRIDE', '0') in ('', '0'), 'kernel override is active'

import torch
import torch_npu
import triton
import msa_triton
paths = {m.__name__: str(Path(m.__file__).resolve()) for m in (torch, torch_npu, triton, msa_triton)}
result['import_paths'] = paths
result['runtime_versions'] = {m.__name__: m.__version__ for m in (torch, torch_npu, triton)}
print(json.dumps({'import_paths': paths, 'runtime_versions': result['runtime_versions']}, indent=2), flush=True)
Path(os.environ['MSA_RUN_DIR'], 'environment.json').write_text(json.dumps(result, indent=2))
assert Path(msa_triton.__file__).resolve().parent == package
assert not Path(triton.__file__).resolve().is_relative_to(package), 'local triton shadows runtime'
assert torch.npu.is_available(), 'NPU unavailable'
torch.npu.set_device(0)
device_info = {'npu_count': torch.npu.device_count(),
               'logical_device': torch.npu.current_device(),
               'device_name': torch.npu.get_device_name(0)}
print(json.dumps(device_info), flush=True)
result['device'] = device_info
Path(os.environ['MSA_RUN_DIR'], 'environment.json').write_text(json.dumps(result, indent=2))
assert device_info['npu_count'] == 1, 'check visible-device mapping before testing'
print('stage=basic_npu_operation', flush=True)
x = torch.arange(8, dtype=torch.float32, device='npu')
y = x + 1
torch.npu.synchronize()
torch.testing.assert_close(y.cpu(), torch.arange(8, dtype=torch.float32) + 1, atol=0, rtol=0)
print('preflight=passed', flush=True)
PY
msa_run preflight "$MSA_PYTHON" "$MSA_RUN_DIR/preflight.py"
```

补充保存 **CANN 的具体版本文件内容与文件路径**，以及 `npu-smi` 显示的
驱动版本。可在 `$MSA_CANN_ROOT` 内用 `rg --files --hidden --follow` 定位
`version.info`、`version.cfg`、`*install.info`；只复制实际存在的版本文件。
不能仅写“已 source set_env.sh”，也不要从 torch_npu 版本猜测 CANN 版本。

执行一次正式环境收集，预期 **145 项**。数量不同或出现收集错误/skip，
先核对源码、导入路径、依赖，不要通过过滤用例凑数。

```bash
msa_run collect "$MSA_PYTHON" -m pytest msa_triton/tests --collect-only -q -o addopts=
```

## 4. 逐阶段测试

默认顺序 A → B → C → D。某阶段出现算子失败时，不运行后续大范围测试，
转第 6 节采集；可以完成该阶段已经启动的独立进程重复测试。
隔离 runner 会继续后续新进程，不会在首个失败时自行停止整个批次。
如果新的进程连基础 NPU 运算也失败，终止剩余测试，记录设备状态；不要
自动重置设备、重启机器或终止其他用户进程。

### A. 两个原始首发问题

```bash
MSA_ATT_NODE='msa_triton/tests/test_triton_attention.py::test_forward_backward_against_fp64[lengths6-2-2-128-4-2-None]'
MSA_SCORE_NODE_1='msa_triton/tests/test_triton_score.py::test_score_forward_backward_fp64[lengths2-2-7-4-1.0]'
MSA_SCORE_NODE_2='msa_triton/tests/test_triton_score.py::test_score_forward_backward_fp64[lengths3-2-7-4-1.0]'

msa_run A-attention env TRITON_CACHE_DIR="$MSA_RUN_DIR/cache-attention" \
  "$MSA_PYTHON" -m msa_triton.tests.probe_npu_isolated "$MSA_ATT_NODE" \
  --repeat 3 --timeout 600 --output-dir "$MSA_RUN_DIR/A-attention"

msa_run A-score env TRITON_CACHE_DIR="$MSA_RUN_DIR/cache-score" \
  "$MSA_PYTHON" -m msa_triton.tests.probe_npu_isolated "$MSA_SCORE_NODE_1" "$MSA_SCORE_NODE_2" \
  --repeat 5 --timeout 600 --output-dir "$MSA_RUN_DIR/A-score"

msa_run A-score-normal env TRITON_CACHE_DIR="$MSA_RUN_DIR/cache-score-normal" \
  "$MSA_PYTHON" -m msa_triton.tests.probe_npu_isolated "$MSA_SCORE_NODE_1" "$MSA_SCORE_NODE_2" \
  --repeat 5 --no-blocking --timeout 600 --output-dir "$MSA_RUN_DIR/A-score-normal"
```

预期依次为 **3/3、10/10、10/10 passed**。默认 runner 强制 blocking=1；
第三项必须使用 `--no-blocking`，仅 unset 父 shell 变量不够。
每个阶段用新缓存，阶段内允许复用缓存；不要声称所有重复都重新编译了。

### B. 新增边界与归约回归、score 隔离全量

```bash
msa_run B-helpers "$MSA_PYTHON" -m msa_triton.tests.probe_npu_isolated \
  msa_triton/tests/test_index_addressing.py msa_triton/tests/test_attention_reduction.py \
  --output-dir "$MSA_RUN_DIR/B-helpers"

msa_run B-score-all "$MSA_PYTHON" -m msa_triton.tests.probe_npu_isolated \
  msa_triton/tests/test_triton_score.py --output-dir "$MSA_RUN_DIR/B-score-all"
```

预期 **10/10、49/49 passed**。10 项 helper 中有 6 项地址/序列边界和
4 项补偿树高低分量精确比较；不可把“精确比较”改为近似比较。

### C. 同进程连续调用

```bash
msa_run C-bf16-blocking env ASCEND_LAUNCH_BLOCKING=1 \
  "$MSA_PYTHON" -m msa_triton.tests.probe_score_sequence --device npu --repeats 20

msa_run C-bf16-normal env -u ASCEND_LAUNCH_BLOCKING \
  "$MSA_PYTHON" -m msa_triton.tests.probe_score_sequence --device npu --repeats 20

msa_run C-bf16-forward env -u ASCEND_LAUNCH_BLOCKING \
  "$MSA_PYTHON" -m msa_triton.tests.probe_score_sequence --device npu --repeats 20 --forward-only

msa_run C-fp32 env -u ASCEND_LAUNCH_BLOCKING \
  "$MSA_PYTHON" -m msa_triton.tests.probe_score_sequence --device npu --repeats 20 --dtype FP32

msa_run C-fp16 env -u ASCEND_LAUNCH_BLOCKING \
  "$MSA_PYTHON" -m msa_triton.tests.probe_score_sequence --device npu --repeats 20 --dtype FP16
```

每项预期 20×7=**140 个 case passed**，退出码 0，并有末尾 JSON
`stage="complete", passed_cases=140`。必须同时检查退出码与末尾记录。
中途某些 `passed` 不能代表该阶段通过。记录首次失败的 iteration、case_index、
seed、shape、dtype、forward-only 和 failed_stage。

工具在 transfer/forward/backward 后均同步，因此 normal 表示未全局阻塞
launch，并不是多 stream 并发验收。每次 shell 命令是新进程；同一命令内部
切换七种 shape。出现异常即退出，不应 try/except 后继续提交设备操作。

### D. 完整回归，只运行一次

```bash
msa_run D-full env -u ASCEND_LAUNCH_BLOCKING \
  "$MSA_PYTHON" -m pytest msa_triton/tests -q -s -x -o addopts= \
  --junitxml="$MSA_RUN_DIR/full.junit.xml"
```

预期 **145 passed / 0 failed / 0 skipped**，不能只看 pytest 退出码。
分文件数量：eager 27、地址/边界 6、k2q 11、pipeline 4、top-k 12、
attention 32、score 49、补偿树 4。eager 27 项固定在 CPU，不能写成
145 个 NPU kernel 测试。历史 native BF16 eager 误差诊断是有意保留的统计，
要区分其文本输出与严格断言失败。

本次不跑 8K–32K benchmark、调优或模型训练；先确认这轮数值与稳定性验收。

## 5. 每个阶段的判断与输出

保留完整 stdout/stderr，不只截取末尾汇总。按 `stages.tsv` 核对每条命令
的退出码；隔离测试还须读取各自 `summary.json` 和子进程 JSON。
`dry_run`、skip、xfail、xpass、timeout、collection_failed 都不是通过。
若因资源/时间停止，记录已完成的轮次和未执行阶段，结论为不完整。

原日志中的并发核心错误、首次同步点和完整错误码必须保留；不要把同一
设备异常产生的多个 traceback 去重到只剩最后一个。报告中区分“独立进程
失败次数”和“已确认的独立缺陷数量”，后者不应由前者直接推导。

## 6. 失败时的定向采集

只对首次失败的精确 pytest node 再启动一个诊断进程，不在已经报非法访问
的进程里读取额外 NPU tensor。把实际 node 字符串保存到 `MSA_FAILED_NODE`。
若失败来自 C 的序列压力，先保存完整原日志；可用同一命令采集 IR，但不能
把单个 pytest node 成功当作压力失败已经消失。

### 6.1 编译崩溃或非法地址：IR、缓存和原始异常链

在结果目录创建 pytest 插件，仅记录异常，不改变测试行为。它用于保留
`CalledProcessError` 的 stdout/stderr/returncode/cmd；普通 pytest 短堆栈
可能没有展示这些字段。不要使用 `--showlocals` 打印设备 tensor。

```bash
cat > "$MSA_RUN_DIR/msa_capture_errors.py" <<'PY'
import json
import os
from pathlib import Path
import pytest

def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    if call.excinfo is None:
        return
    error, chain, seen = call.excinfo.value, [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({'type': type(error).__name__, 'message': str(error),
                      'returncode': getattr(error, 'returncode', None),
                      'cmd': getattr(error, 'cmd', None),
                      'stdout': _text(getattr(error, 'stdout', None)),
                      'stderr': _text(getattr(error, 'stderr', None))})
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    with Path(os.environ['MSA_RUN_DIR'], 'exception-chain.jsonl').open('a') as output:
        output.write(json.dumps({'node': item.nodeid, 'phase': call.when, 'chain': chain},
                                ensure_ascii=False, default=str) + '\n')
PY

# MSA_FAILED_NODE 必须是日志中的原始节点，不要修改测试参数。
msa_run failure-ir env PYTHONPATH="$MSA_PROJECT_ROOT:$MSA_RUN_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  ASCEND_LAUNCH_BLOCKING=1 TRITON_CACHE_DIR="$MSA_RUN_DIR/failure-cache" \
  TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$MSA_RUN_DIR/failure-ir" MLIR_ENABLE_DUMP=1 \
  "$MSA_PYTHON" -m pytest "$MSA_FAILED_NODE" -vv -s -x --tb=long -o addopts= -p msa_capture_errors
```

这个诊断进程强制 blocking=1，只用于定位，不能替代原先 normal 条件的
失败结果。若只在特定压力顺序下失败，额外复制第 C 节那条命令，保留其
原 dtype/轮数/forward-only/blocking 条件，加上新的 cache/dump 路径；
不要修改种子或将其换成 pytest 包装。

保留实际产生的 `.ttir`、`.ttadapter.mlir`、其他 MLIR/LLVM IR、缓存 JSON、
编译命令和 `.so`/二进制；某阶段未生成的文件标为缺失。日志若提到仍存在
的临时失败 IR，复制到结果目录。记录实际使用的 `bishengir-compile` 路径
和 SHA256，版本输出可获得则附上。不要清除原缓存，也不在此轮尝试更换
编译器或开关组合来掩盖失败。

调试变量依据 [triton-ascend v3.2.2 文档](https://triton-ascend.readthedocs.io/zh-cn/v3.2.2/environment_variable_and_compiler_options_reference.html)。
NPU 日志目录按本机配置定位，保存相同 PID/时间窗口对应的首个错误 dump；
日志文件不存在或不可读时写明原因，不假装已经采集。

### 6.2 数值断言失败

与编译/执行异常分开报告：张量名（score/O/LSE/dQ/dK/dV）、shape、dtype、
seed、最大绝对/相对误差、超阈值元素数与前 10 个出错索引。对这些索引
附实际值、FP64 原值及转成公开 dtype 后的参考值；`-inf` mask 单独比较，
不要把 `inf-inf` 的 NaN 算入有限值误差。

**只有同步已成功、故障仅为数值断言时**，才可在结果目录写临时诊断脚本
复现该节点，保存同一份输入/upstream/indices/cu 与结果的 CPU tensor。
调用现有实现和 `reference_fp64.py`，不得改 kernel、oracle、输入量化或
阈值。原 pytest 失败仍保留；临时脚本保存源码和命令一并回传。
若无法取得某项统计，明确写“未采集”，不要推测数值或为继续测试而放宽阈值。

## 7. 交付物与报告模板

所有产物放入 `$MSA_RUN_DIR`：

- `REPORT.md`：按下列模板填写，不用“看起来通过”等模糊结论。
- `environment.json`、preflight.log、Git/zip 来源记录、CANN/驱动版本材料。
- 每阶段 `.command.txt`、完整 `.log`、`stages.tsv`。
- 隔离测试的 summary、collection、逐 case 日志和 JSON；完整测试 JUnit。
- 若失败：原始异常链、IR/缓存/编译器信息、相关设备日志、数值诊断数据。
- 所有临时诊断脚本；没有执行的阶段和没有采集的材料逐项说明。

`REPORT.md` 模板：

```markdown
# MSA NPU 候选复测报告
- 实际 Git SHA/zip 来源、核心哈希是否匹配、是否 dirty：
- Python/torch/torch_npu/triton/triton-ascend/CANN/驱动：
- 物理卡可见性、逻辑 device、实际权限环境：
- 源码与第三方 triton 的实际导入路径：

| 阶段 | 预期 | 实际 pass/fail/skip/timeout | 退出码 | 日志/summary 路径 |
|---|---|---|---|---|
| A-attention | 3 | | | |
| A-score | 10 | | | |
| A-score-normal | 10 | | | |
| B-helpers | 10 | | | |
| B-score-all | 49 | | | |
| C-bf16-blocking / normal / forward | 各140 | | | |
| C-fp32 / fp16 | 各140 | | | |
| D-full | 145，无skip | | | |

## 首次失败（如有）
- node / iteration / seed / shape / dtype / 梯度模式：
- 阶段：导入、传输、编译、forward、backward、同步或数值断言：
- 完整首个错误、进程PID、时间、独立进程重复结果：
- IR/异常链/设备dump/数值明细位置，以及缺失证据：

## 结论
- 测试全部通过 / 算子失败 / 环境阻塞 / 测试不完整：
- 原 attention 编译崩溃是否再次出现：
- 原 score 非确定性异常是否再次出现：
- 未执行阶段及原因：
```

若全部阶段完成且无失败/skip/超时，可写“本轮指定测试通过”；不要扩大为
所有动态范围、长序列、多 stream 或完整模型训练已经验收。

最后打包，不覆盖旧结果、不自动发送至第三方：

```bash
tar -czf "$MSA_RUN_DIR.tar.gz" -C "$MSA_RUN_DIR" .
sha256sum "$MSA_RUN_DIR.tar.gz" > "$MSA_RUN_DIR.tar.gz.sha256"
printf 'Return: %s.tar.gz and its .sha256\n' "$MSA_RUN_DIR"
```

将 `REPORT.md` 的正文及包路径/大小/SHA256 回报给用户，由用户转交。
