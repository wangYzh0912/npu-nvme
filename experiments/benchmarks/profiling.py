#!/usr/bin/env python3
"""External profiling helpers for the WP1 experiments.

The benchmark processes already emit the authoritative monotonic event
timeline.  This module collects process-level evidence around a benchmark
without changing its I/O implementation: ``perf stat`` and ``strace -c`` are
best-effort counters, while ``perf record`` is retained as a raw profile for
call-path inspection.
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


def command_available(name):
    return shutil.which(name) is not None


def child_usage():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def run(argv, output_dir, label, timeout=None):
    """Run one command and collect optional external profiler outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [str(item) for item in argv]
    record = {
        "label": label,
        "argv": argv,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tools": {name: command_available(name)
                  for name in ("perf", "strace")},
    }
    before = child_usage()

    def execute(command, stdout_path, stderr_path):
        started = time.monotonic_ns()
        try:
            completed = subprocess.run(
                command, stdout=stdout_path.open("w", encoding="utf-8"),
                stderr=stderr_path.open("w", encoding="utf-8"),
                timeout=timeout, check=False, text=True)
            return completed.returncode, time.monotonic_ns() - started, None
        except subprocess.TimeoutExpired as error:
            return -124, time.monotonic_ns() - started, repr(error)
        except OSError as error:
            return -1, time.monotonic_ns() - started, repr(error)

    # Build each command separately to keep the event list readable and avoid
    # relying on shell quoting.
    commands = [("program", argv, output_dir / "program.stdout",
                 output_dir / "program.stderr")]
    if record["tools"]["perf"]:
        commands.append(("perf_stat", ["perf", "stat", "-x,",
                         "-e", "task-clock,cycles,instructions",
                         "-e", "context-switches,cpu-migrations,page-faults",
                         *argv], output_dir / "perf_stat.stdout",
                         output_dir / "perf_stat.stderr"))
        commands.append(("perf_record", ["perf", "record", "-F", "99",
                         "-g", "--call-graph", "dwarf", "-o",
                         str(output_dir / "perf.data"), *argv],
                        output_dir / "perf_record.stdout",
                        output_dir / "perf_record.stderr"))
    if record["tools"]["strace"]:
        commands.append(("strace_summary", ["strace", "-f", "-c", "-o",
                         str(output_dir / "strace_summary.txt"), *argv],
                        output_dir / "strace.stdout",
                        output_dir / "strace.stderr"))

    results = []
    for index, (item_label, command, stdout_path, stderr_path) in enumerate(commands, 1):
        # A profiled command is run once per tool. Keep every invocation's
        # stdout/stderr and raw profile instead of overwriting the previous
        # invocation's evidence.
        suffix = f"_{index:02d}"
        stdout_path = output_dir / f"{item_label}{suffix}.stdout"
        stderr_path = output_dir / f"{item_label}{suffix}.stderr"
        command = list(command)
        command = [str(output_dir / f"perf.data{suffix}")
                   if item == str(output_dir / "perf.data") else item
                   for item in command]
        command = [str(output_dir / f"strace_summary{suffix}.txt")
                   if item == str(output_dir / "strace_summary.txt") else item
                   for item in command]
        code, elapsed_ns, error = execute(command, stdout_path, stderr_path)
        results.append({"label": item_label, "argv": command,
                        "returncode": code, "elapsed_ms": elapsed_ns / 1e6,
                        "error": error})
    record.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commands": results, "child_usage_before": before,
                   "child_usage_after": child_usage()})
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="command")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after the options")
    result = run(args.command, args.output_dir, args.label, args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    program = next(item for item in result["commands"]
                   if item["label"] == "program")
    raise SystemExit(0 if program["returncode"] == 0 else program["returncode"])


if __name__ == "__main__":
    main()
