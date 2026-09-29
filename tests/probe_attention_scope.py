"""Run six attention compiler probes in separate processes, retaining local evidence.

This diagnostic does not change the production kernel or establish NPU acceptance.
The parent imports only the standard library and existing stdlib diagnostic helpers.
Run from the package parent: ``python -m msa_triton.tests.probe_attention_scope``.
CPU execution requires ``--device cpu --interpreter`` and only checks the harness
and probe arithmetic. Each child receives its own compilation cache and IR dump.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

from .probe_attention_compile_replay import _disable_core_dump, _stop_group
from .probe_npu_isolated import _classify, _load_report, _write_json


CASES = (
    "tree-d128", "qkv-d128", "qk-d128", "qv-d128", "split-kv-d128",
    "tile-sum-kahan-d128",
)
SOURCE_FILES = (
    "triton/sparse_attention.py", "tests/probe_attention_cases.py",
    "tests/probe_attention_scope.py", "tests/probe_npu_isolated.py",
    "tests/probe_attention_compile_replay.py", "tests/probe_attention_ablation.py",
    "tests/test_attention_reduction.py", "tests/test_triton_attention.py",
    "tests/reference_fp64.py",
)


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return seconds


def _git(package: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(package), *arguments], capture_output=True,
            text=True, timeout=10, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _run(command: list[str], *, cwd: Path, env: dict, log: Path, timeout: float) -> dict:
    started = time.monotonic()
    result = {"command": command, "cwd": str(cwd), "exitcode": None,
              "status": "completed", "log": str(log)}
    with log.open("w") as stream:
        stream.write(f"cwd: {cwd}\ncommand: {shlex.join(command)}\n\n")
        stream.flush()
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=_disable_core_dump,
            )
        except OSError as error:
            result.update(status="error", reason=f"process_start: {error}")
            stream.write(f"{error}\n")
        else:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
                _stop_group(process)
            except KeyboardInterrupt:
                result["status"] = "interrupted"
                _stop_group(process)
            except BaseException:
                _stop_group(process)
                raise
            result["exitcode"] = process.returncode
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def _describe(result: dict, report: dict | None, node: str) -> dict:
    status = _classify(result, report, node)
    log = Path(result["log"]).read_text(errors="replace")
    stages = re.findall(r"^STAGE=([^\r\n]+)", log, flags=re.MULTILINE)
    detail = log + "\n" + "\n".join(
        item.get("detail") or "" for item in (report or {}).get("test_reports", [])
    )
    runtime_fault = any(marker in detail.lower() for marker in (
        "507035", "the vector core execution is abnormal", "vector core exception",
        "scalar instruction accesses an invalid gm address", "gm address accessed by scalar",
        "illegal gm", "illegal memory access",
    ))
    if result["status"] in {"timeout", "interrupted"}:
        reason = result["status"]
    elif runtime_fault:
        reason, status = "npu_runtime_fault", "failed"
    elif status == "passed":
        reason = "passed"
    elif "PlanMemory Traverse IR Failed" in detail:
        reason = "compile_plan_memory"
    elif "MLIRCompilationError" in detail or "CompilationError" in detail:
        reason = "compile_error"
    elif "AssertionError" in detail and status == "failed":
        reason = "assertion_failure"
    else:
        reason = result.get("reason", status)
    return dict(result, status=status, reason=reason, runtime_fault=runtime_fault,
                last_stage=stages[-1] if stages else "none", stages=stages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--interpreter", action="store_true")
    parser.add_argument("--timeout", type=_positive_seconds, default=180,
                        help="maximum seconds per case, including compilation (default: 180)")
    parser.add_argument("--output-dir", type=Path, help="new directory; default: /tmp/msa-att-scope-*")
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("POSIX process-group control is required")
    if (args.device == "cpu") != args.interpreter:
        parser.error("CPU requires --interpreter; NPU must not use --interpreter")

    package = Path(__file__).resolve().parents[1]
    root = package.parent
    hashes = {}
    for relative in SOURCE_FILES:
        try:
            hashes[relative] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        except OSError as error:
            parser.error(f"source manifest unavailable: {error}")
    if args.output_dir:
        output = args.output_dir.expanduser().absolute()
        try:
            output.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            parser.error(f"output directory must be new: {error}")
    else:
        output = Path(tempfile.mkdtemp(prefix="msa-att-scope-"))

    env = os.environ.copy()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", MSA_TEST_DEVICE=args.device,
               ASCEND_LAUNCH_BLOCKING="1", TRITON_KERNEL_DUMP="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    for key in ("PYTEST_ADDOPTS", "TRITON_INTERPRET", "TRITON_KERNEL_OVERRIDE"):
        env.pop(key, None)
    if args.interpreter:
        env["TRITON_INTERPRET"] = "1"
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "python_version": sys.version, "cwd": str(root), "source_sha256": hashes,
        "git_sha": _git(package, "rev-parse", "HEAD"),
        "git_status": _git(package, "status", "--short"),
        "environment": {key: env.get(key) for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "MSA_TEST_DEVICE",
            "TRITON_INTERPRET", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "TRITON_KERNEL_OVERRIDE",
        )},
        "timeout_seconds": args.timeout, "device": args.device, "cases": list(CASES),
        "npu_acceptance": False,
        "note": "Probe outcomes only. Fresh processes do not reset unhealthy hardware.",
    }
    _write_json(output / "manifest.json", manifest)
    summary = {"status": "running", "results": [], "not_run": list(CASES)}
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    print(f"Output: {output}", flush=True)
    print(f"Source: git={manifest['git_sha']} probe_sha256={hashes['tests/probe_attention_cases.py']}", flush=True)
    started = time.monotonic()
    try:
        for case in CASES:
            directory = output / case
            directory.mkdir()
            for child in ("cache", "dump"):
                (directory / child).mkdir()
            report_path = directory / "pytest.json"
            node = f"msa_triton/tests/probe_attention_cases.py::test_probe[{case}]"
            child_env = dict(env, MSA_ISOLATED_REPORT_PATH=str(report_path),
                             TRITON_CACHE_DIR=str(directory / "cache"),
                             TRITON_DUMP_DIR=str(directory / "dump"))
            command = [sys.executable, "-m", "pytest", "-p", "msa_triton.tests.probe_npu_isolated",
                       f"--rootdir={root}", f"--basetemp={directory / 'pytest-tmp'}",
                       "-o", "addopts=", "-x", "-q", "-s", "--tb=long", node]
            result = _run(command, cwd=root, env=child_env, log=directory / "run.log", timeout=args.timeout)
            result = _describe(result, _load_report(report_path), node)
            result.update(case=case, node=node, report=str(report_path))
            _write_json(directory / "result.json", result)
            summary["results"].append(result)
            summary["not_run"].remove(case)
            print(f"{case}: {result['status']} reason={result['reason']} STAGE={result['last_stage']}", flush=True)
            _write_json(summary_path, summary)
            if result["runtime_fault"] or result["status"] == "interrupted":
                summary["status"] = "stopped_runtime_fault" if result["runtime_fault"] else "interrupted"
                break
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    if summary["status"] == "running":
        summary["status"] = "passed" if all(r["status"] == "passed" for r in summary["results"]) else "not_passed"
    summary.update(counts=dict(Counter(r["status"] for r in summary["results"])),
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   duration_seconds=round(time.monotonic() - started, 3))
    _write_json(summary_path, summary)
    print(f"Status: {summary['status']} counts={summary['counts']} not_run={len(summary['not_run'])}", flush=True)
    return 130 if summary["status"] == "interrupted" else 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
