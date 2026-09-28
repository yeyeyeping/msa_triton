r"""Run each pytest node in a fresh process and retain independent NPU evidence.

Run from the package's parent directory after activating the target environment::

    ASCEND_RT_VISIBLE_DEVICES=15 python -m msa_triton.tests.probe_npu_isolated \
        msa_triton/tests/test_triton_score.py --repeat 5 --output-dir /tmp/msa-score

``--dry-run`` collects on CPU and prints commands without executing device tests.
``--device cpu --interpreter`` exercises this runner on a CPU-only machine.
The parent imports only the standard library. Collection uses CPU/interpreter
settings in a separate process; each execution uses the requested device.

A new process isolates a poisoned runtime context, but cannot reset faulty NPU
hardware. Passing skips, xfails, no-tests, or a dry run off as acceptance is
intentionally disallowed. Results describe independent test executions, not a
count of independent kernel defects. This module also acts as a small pytest
reporting plugin in the child processes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time


_COLLECTION_REPORTS: list[dict] = []
_TEST_REPORTS: list[dict] = []


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pytest_collectreport(report):
    """Pytest plugin hook; do not import pytest or torch into the parent."""
    if report.outcome != "passed":
        _COLLECTION_REPORTS.append({
            "node": report.nodeid, "outcome": report.outcome,
            "detail": str(report.longrepr),
        })


def pytest_runtest_logreport(report):
    _TEST_REPORTS.append({
        "node": report.nodeid,
        "phase": report.when,
        "outcome": report.outcome,
        "wasxfail": getattr(report, "wasxfail", None),
        "detail": str(report.longrepr) if report.longrepr else None,
    })


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get("MSA_ISOLATED_REPORT_PATH")
    if destination:
        _write_json(Path(destination), {
            "exitstatus": int(exitstatus),
            "nodes": [item.nodeid for item in session.items],
            "selectors": [str(item.path) + "::" + item.nodeid.split("::", 1)[1]
                          for item in session.items],
            "collection_reports": _COLLECTION_REPORTS,
            "test_reports": _TEST_REPORTS,
        })


def _stop_process(process: subprocess.Popen) -> None:
    """Stop compiler descendants as well as the timed-out pytest process."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def _run(command: list[str], *, cwd: Path, env: dict, log: Path, timeout: float) -> dict:
    start = time.monotonic()
    timed_out = False
    interrupted = False
    with log.open("w") as stream:
        stream.write(f"cwd: {cwd}\ncommand: {shlex.join(command)}\n\n")
        stream.flush()
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            stream.write(f"Unable to start process: {error}\n")
            return {
                "command": command, "exitcode": None,
                "duration_seconds": time.monotonic() - start,
                "status": "error", "reason": str(error), "log": str(log),
            }
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process(process)
        except KeyboardInterrupt:
            interrupted = True
            _stop_process(process)
    return {
        "command": command,
        "exitcode": process.returncode,
        "duration_seconds": time.monotonic() - start,
        "status": "interrupted" if interrupted else "timeout" if timed_out else "completed",
        "log": str(log),
    }


def _load_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _classify(result: dict, report: dict | None, node: str) -> str:
    if result["status"] != "completed":
        return result["status"]
    if report is None:
        return "error"
    if report.get("collection_reports"):
        return "error"
    if not report.get("nodes"):
        return "no_tests"
    if report["nodes"] != [node]:
        return "error"
    phases = report.get("test_reports", [])
    if any(item["outcome"] == "failed" for item in phases):
        return "failed" if result["exitcode"] == 1 else "error"
    if any(item["wasxfail"] is not None for item in phases):
        return "xfailed" if any(item["outcome"] == "skipped" for item in phases) else "xpassed"
    if any(item["outcome"] == "skipped" for item in phases):
        return "skipped"
    if (result["exitcode"] == 0 and len(phases) == 3
            and {item["phase"] for item in phases} == {"setup", "call", "teardown"}
            and all(item["outcome"] == "passed" and item["node"] == node for item in phases)):
        return "passed"
    return "error"


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", default=["msa_triton/tests"],
                        help="pytest paths or exact node IDs, relative to the package parent")
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--interpreter", action="store_true",
                        help="explicitly enable TRITON_INTERPRET=1; only valid with --device cpu")
    parser.add_argument("--repeat", type=_positive_integer, default=1,
                        help="repeat the whole node list; every execution gets a fresh process")
    parser.add_argument("--timeout", type=_positive_integer, default=600,
                        help="timeout in seconds for each child, including collection (default: 600)")
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for logs/JSON; defaults to a unique directory in /tmp")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child processes")
    parser.add_argument("--no-blocking", action="store_true",
                        help="unset ASCEND_LAUNCH_BLOCKING (default: force 1 for NPU diagnosis)")
    parser.add_argument("--dry-run", action="store_true", help="collect and print commands without running tests")
    args = parser.parse_args(argv)
    if args.device == "cpu" and not args.interpreter:
        parser.error("--device cpu requires --interpreter; CPU is a harness check, not NPU acceptance")
    if args.device == "npu" and args.interpreter:
        parser.error("--interpreter is not valid with --device npu")

    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.expanduser().resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="msa-isolated-"))
    if args.output_dir:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"output directory already exists; use a new directory: {output}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH", "")]))
    # External pytest plugins/options must not silently filter or alter acceptance.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env["MSA_TEST_DEVICE"] = args.device
    env.pop("TRITON_INTERPRET", None)
    if args.interpreter:
        env["TRITON_INTERPRET"] = "1"
    if args.device == "npu" and not args.no_blocking:
        env["ASCEND_LAUNCH_BLOCKING"] = "1"
    else:
        env.pop("ASCEND_LAUNCH_BLOCKING", None)
    base = [args.python, "-m", "pytest", "-p", "msa_triton.tests.probe_npu_isolated",
            f"--rootdir={root}", "-o", "addopts="]
    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(root), "device": args.device, "dry_run": args.dry_run,
        "repeat": args.repeat, "timeout_seconds": args.timeout,
        "environment": {key: env.get(key) for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "MSA_TEST_DEVICE", "TRITON_INTERPRET")},
        "status": "collecting", "results": [],
        "note": "Independent processes do not imply independent defects or reset failed device hardware.",
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    print(f"Output: {output}", flush=True)

    collect_env = env.copy()
    collect_env.update(MSA_TEST_DEVICE="cpu", TRITON_INTERPRET="1",
                       MSA_ISOLATED_REPORT_PATH=str(output / "collection.json"))
    collect_env.pop("ASCEND_LAUNCH_BLOCKING", None)
    collection = _run(base + ["--collect-only", "-q", *args.targets], cwd=root, env=collect_env,
                      log=output / "collection.log", timeout=args.timeout)
    manifest = _load_report(output / "collection.json")
    summary["collection"] = collection
    if (collection["status"] != "completed" or collection["exitcode"] != 0 or not manifest
            or manifest.get("collection_reports") or not manifest.get("nodes")):
        summary["status"] = "collection_failed"
        _write_json(summary_path, summary)
        print(f"Collection did not produce a complete test list; see {output / 'collection.log'}", file=sys.stderr)
        return 130 if collection["status"] == "interrupted" else 2
    selections = list(dict.fromkeys(zip(manifest["nodes"], manifest["selectors"])))
    summary["nodes"] = [node for node, _ in selections]
    summary["status"] = "running"
    _write_json(summary_path, summary)
    total = len(selections) * args.repeat
    print(f"Collected {len(selections)} nodes; {total} independent executions planned.", flush=True)

    for repeat in range(1, args.repeat + 1):
        for node, selector in selections:
            ordinal = len(summary["results"]) + 1
            digest = hashlib.sha256(node.encode()).hexdigest()[:12]
            stem = f"{ordinal:05d}-r{repeat}-{digest}"
            command = base + ["-x", "--tb=short", "-q", "-s", selector]
            if args.dry_run:
                result = {"command": command, "exitcode": None, "duration_seconds": 0,
                          "status": "dry_run"}
                print(shlex.join(command), flush=True)
            else:
                report_path = output / f"{stem}.json"
                child_env = dict(env, MSA_ISOLATED_REPORT_PATH=str(report_path))
                result = _run(command, cwd=root, env=child_env, log=output / f"{stem}.log", timeout=args.timeout)
                result["status"] = _classify(result, _load_report(report_path), node)
                result["report"] = str(report_path)
                print(f"[{ordinal}/{total}] {result['status']}: {node} (repeat {repeat})", flush=True)
            result.update(node=node, repeat=repeat)
            summary["results"].append(result)
            summary["counts"] = dict(Counter(item["status"] for item in summary["results"]))
            summary["status"] = "interrupted" if result["status"] == "interrupted" else "running"
            _write_json(summary_path, summary)
            if result["status"] == "interrupted":
                print(f"Interrupted; partial results: {summary_path}", file=sys.stderr)
                return 130
    passed = all(item["status"] == "passed" for item in summary["results"])
    summary["status"] = "dry_run" if args.dry_run else "passed" if passed else "not_passed"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(summary_path, summary)
    print(f"{summary['status']}: {summary.get('counts', {})}; summary: {summary_path}", flush=True)
    return 0 if passed or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
