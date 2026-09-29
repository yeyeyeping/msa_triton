"""Replay the reported attention compiler failure without running device code.

Only the known bishengir command is accepted. A baseline SIGABRT with the
reported PlanMemory fatal gates a second compile with auto-multi-buffer=False.
This diagnostic imports no torch/Triton and is not an NPU acceptance test.
The caller must establish the IR's D=128 specialization from the saved logs;
a matching function symbol and file hash alone cannot establish its provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


EXPECTED_COMPILER_SHA256 = (
    "89655a56941efe9a184e4d5dfccb783ad88458707827ff6fdd146e7bd2d4af5c"
)
NODE_SUFFIX = (
    "tests/test_triton_attention.py::test_forward_backward_against_fp64"
    "[lengths6-2-2-128-4-2-None]"
)
FATAL = "LLVM ERROR: PlanMemory Traverse IR Failed!"
FLAGS = [
    "--target=Ascend910_9382",
    "--enable-auto-multi-buffer=True",
    "--enable-auto-bind-sub-block=True",
    "--enable-auto-blockify-loop",
    "--enable-hfusion-compile=true",
    "--enable-hivm-compile=true",
    "--enable-triton-kernel-compile=true",
    "--mlir-print-ir-after-failure",
    "--mlir-print-stacktrace-on-diagnostic",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _digest_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("expected a 64-character SHA256 hex digest")
    return value.lower()


def _select_command(path: Path, record_index: int | None) -> tuple[dict, dict, int]:
    # Index refers to zero-based JSONL records, including unrelated records.
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    choices = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not str(record.get("node", "")).endswith(NODE_SUFFIX):
            continue
        chain = record.get("chain", [])
        if not isinstance(chain, list) or not all(isinstance(item, dict) for item in chain):
            raise ValueError("matching record.chain must be a list of exception objects")
        for exception in chain:
            if (exception.get("type") == "CalledProcessError"
                    and exception.get("returncode") == -6
                    and FATAL in (exception.get("stderr") or "")):
                choices.append((record, exception, index))
    if record_index is not None:
        choices = [choice for choice in choices if choice[2] == record_index]
    if not choices:
        raise ValueError("no matching attention CalledProcessError (-6 / PlanMemory fatal)")
    distinct = {json.dumps(choice[1].get("cmd"), sort_keys=True) for choice in choices}
    if len(distinct) != 1:
        raise ValueError("multiple attention compiler commands; select --record-index")
    record, exception, index = choices[0]
    command = exception.get("cmd")
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError("CalledProcessError.cmd must be an argv list of strings")
    if len(command) != len(FLAGS) + 4 or command[2:-2] != FLAGS or command[-2] != "-o":
        raise ValueError("command differs from the exact reported compiler flags/order")
    if (not Path(command[0]).is_absolute()
            or Path(command[0]).name != "bishengir-compile"
            or not Path(command[1]).is_absolute()
            or not command[1].endswith(".ttadapter.mlir")
            or not Path(command[-1]).is_absolute()):
        raise ValueError("expected absolute bishengir/input/output paths from original command")
    return record, exception, index


def _stop_group(process: subprocess.Popen) -> None:
    """Clean compiler descendants even if the leader exits after SIGTERM."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _stderr_info(path: Path) -> tuple[bool, list[str]]:
    fatal_seen = False
    excerpt = []
    with path.open(errors="replace") as stream:
        for line in stream:
            fatal_seen |= FATAL in line
            line = line.strip()
            if (line and len(excerpt) < 8 and not line.startswith(
                    ("PLEASE submit", "Stack dump", "Stack trace"))
                    and not re.match(r"^\d+\s", line)):
                excerpt.append(line[:500])
    return fatal_seen, excerpt


def _disable_core_dump() -> None:
    # Called only in the single-threaded POSIX child immediately before exec.
    # An expected SIGABRT should not create a large core in the original cwd.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _is_compiler_binary(path: Path) -> bool:
    # A successful process that only emitted IR/JSON is not a complete compile.
    if path.name in {"kernel", "kernel.o", "kernel.so", "kernel.npubin"}:
        return True
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def _compile(name: str, command: list[str], cwd: Path, root: Path, timeout: float) -> dict:
    directory = root / name
    directory.mkdir()
    command = command.copy()
    command[1] = str(root / "input.ttadapter.mlir")
    command[-1] = str(directory / "kernel")
    if name == "multi_buffer_off":
        command[3] = "--enable-auto-multi-buffer=False"
    _write_json(directory / "command.json", {"argv": command, "cwd": str(cwd)})
    started = time.monotonic()
    result = {"name": name, "returncode": None, "timed_out": False,
              "start_error": None, "no_device_execution": True}
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True,
                preexec_fn=_disable_core_dump,
            )
        except OSError as error:
            result["start_error"] = repr(error)
        else:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                _stop_group(process)
            except BaseException:
                _stop_group(process)
                raise
            result["returncode"] = process.returncode
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["plan_memory_fatal"], result["stderr_excerpt"] = _stderr_info(directory / "stderr.log")
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"command.json", "stdout.log", "stderr.log"}:
            artifacts.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                              "sha256": _sha256(path),
                              "compiler_binary": _is_compiler_binary(path)})
    result["output_files"] = artifacts
    result["compile_complete"] = (
        result["returncode"] == 0 and not result["timed_out"]
        and any(item["bytes"] > 0 and item["compiler_binary"]
                for item in artifacts)
    )
    _write_json(directory / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exception-chain", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--ir-sha256", type=_digest_argument, required=True)
    parser.add_argument("--cwd", type=Path, required=True,
                        help="original compilation invocation parent directory")
    parser.add_argument("--output-dir", type=Path, help="new directory; defaults to /tmp/msa-att-replay-*")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--record-index", type=int,
                        help="zero-based nonempty JSONL record index (including score records)")
    parser.add_argument("--expected-compiler-sha256", type=_digest_argument,
                        default=EXPECTED_COMPILER_SHA256)
    args = parser.parse_args(argv)
    root = None
    summary = {"no_device_execution": True, "npu_acceptance": False,
               "status": "provenance_error", "results": []}
    try:
        if os.name != "posix":
            raise ValueError("this replay requires POSIX process-group control")
        if not 0 < args.timeout < float("inf"):
            raise ValueError("--timeout must be positive and finite")
        if args.record_index is not None and args.record_index < 0:
            raise ValueError("--record-index must be nonnegative")
        cwd = args.cwd.resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError("--cwd must be an existing directory")
        source = args.ir.resolve(strict=True)
        source_hash = _sha256(source)
        if source_hash != args.ir_sha256:
            raise ValueError("IR SHA256 mismatch; no compiler executed")
        source_bytes = source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source_hash:
            raise ValueError("IR changed during validation; no compiler executed")
        symbols = re.findall(
            r"(?:func\.func|tt\.func)\s+(?:public\s+|private\s+)?@([\w.$-]+)",
            source_bytes.decode("utf-8"),
        )
        kernel_symbols = [symbol for symbol in symbols if symbol.startswith("_backward_kv_kernel")]
        if len(kernel_symbols) != 1:
            raise ValueError("IR must define exactly one _backward_kv_kernel function")
        chain = args.exception_chain.resolve(strict=True)
        record, exception, index = _select_command(chain, args.record_index)
        command = exception["cmd"]
        compiler = Path(command[0]).resolve(strict=True)
        compiler_hash = _sha256(compiler)
        if compiler_hash != args.expected_compiler_sha256:
            raise ValueError("compiler SHA256 mismatch; no compiler executed")
        if args.output_dir is None:
            root = Path(tempfile.mkdtemp(prefix="msa-att-replay-"))
        else:
            destination = args.output_dir.absolute()
            destination.mkdir(exist_ok=False)
            root = destination
        (root / "input.ttadapter.mlir").write_bytes(source_bytes)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()), "script_sha256": _sha256(Path(__file__)),
            "python": sys.executable, "cwd": str(cwd),
            "exception_chain": str(chain), "exception_chain_sha256": _sha256(chain),
            "record_index": index, "original_node": record["node"], "original_argv": command,
            "original_stderr": exception["stderr"], "original_returncode": exception["returncode"],
            "compiler": str(compiler), "compiler_sha256": compiler_hash,
            "source_ir": str(source), "source_ir_sha256": source_hash,
            "copied_ir": str(root / "input.ttadapter.mlir"), "symbols": symbols,
            "specialization_provenance": "caller must match saved D=128 IR to failed invocation",
            "environment": {key: os.environ.get(key) for key in (
                "PATH", "LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
                "ASCEND_TOOLKIT_HOME", "ASCEND_RT_VISIBLE_DEVICES", "TRITON_CACHE_DIR",
            )},
            "no_device_execution": True,
        }
        _write_json(root / "manifest.json", manifest)
        summary.update({"status": "running_baseline", "compiler_sha256": compiler_hash,
                        "ir_sha256": source_hash, "symbol": kernel_symbols[0]})
        _write_json(root / "summary.json", summary)
        baseline = _compile("baseline", command, cwd, root, args.timeout)
        summary["results"].append(baseline)
        reproduced = (baseline["returncode"] == -6 and baseline["plan_memory_fatal"]
                      and not baseline["timed_out"])
        if not reproduced:
            summary["status"] = "baseline_not_reproduced_variant_not_run"
            code = 1
        else:
            if (_sha256(compiler) != compiler_hash
                    or _sha256(root / "input.ttadapter.mlir") != source_hash):
                raise ValueError("compiler or copied IR changed after baseline; variant not run")
            summary["status"] = "running_multi_buffer_off"
            _write_json(root / "summary.json", summary)
            variant = _compile("multi_buffer_off", command, cwd, root, args.timeout)
            summary["results"].append(variant)
            if variant["compile_complete"]:
                summary["status"] = "variant_compiled_device_unverified"
                code = 0
            else:
                summary["status"] = "variant_not_compiled"
                code = 1
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        code = 130
    except (OSError, ValueError, TypeError, KeyError) as error:
        summary["status"] = "provenance_or_execution_error"
        summary["error"] = str(error)
        code = 2
    if root is not None:
        _write_json(root / "summary.json", summary)
        # A compact exchange artifact omits full paths/argv/stderr and file lists.
        short = {key: value for key, value in summary.items() if key != "results"}
        short["results"] = [{
            "name": result["name"], "returncode": result["returncode"],
            "timed_out": result["timed_out"], "plan_memory_fatal": result["plan_memory_fatal"],
            "compile_complete": result["compile_complete"],
            "first_diagnostic": result["stderr_excerpt"][:1],
            "start_error": result["start_error"],
            "output_count": len(result["output_files"]),
        } for result in summary["results"]]
        _write_json(root / "short-summary.json", short)
        print(f"Output: {root}")
    print(f"Status: {summary['status']}")
    for key in ("compiler_sha256", "ir_sha256", "symbol", "error"):
        if key in summary:
            print(f"{key}: {summary[key]}")
    for result in summary["results"]:
        print(f"{result['name']}: returncode={result['returncode']}, "
              f"timeout={result['timed_out']}, compile_complete={result['compile_complete']}")
        if result["stderr_excerpt"]:
            print(f"  {result['stderr_excerpt'][0][:300]}")
        if result["start_error"]:
            print(f"  start_error: {result['start_error'][:300]}")
    print("No device code executed; this is not NPU acceptance.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
