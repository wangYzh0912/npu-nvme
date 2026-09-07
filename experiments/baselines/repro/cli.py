"""Command line entry point for the unified baseline reproduction."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .adapters import ADAPTERS
from .protocol import AdapterError, DependencyBlocked
from .runner import prepare as prepare_run
from .runner import restore as restore_run
from .runner import run as train_run
from .state_bridge import write_json
from .inventory import build_inventory, storage_evidence


REQUIRED = (
    "project_commit", "model", "seed", "input_tokens", "batch_size",
    "dropout", "warmup_steps", "formal_steps", "checkpoint_every",
    "capture_mode", "continue_steps", "loss_rtol", "loss_atol",
    "npu_device", "numa_node", "fs_test_dir", "raw_pci",
    "raw_test_authorized", "worker_python_bytecheckpoint",
    "worker_python_fastpersist", "results_root",
    "same_physical_storage_verified",
)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED if config.get(field) in (None, "")]
    if missing:
        raise ValueError("config missing required fields: " + ", ".join(missing))
    if config["project_commit"] != "f3c086157d2ce65d8b686cd938874b1b3541ec5d":
        raise ValueError("project_commit must be the locked f3c0861 baseline")
    if config["capture_mode"] != "frozen":
        raise ValueError("first implementation only supports frozen capture")
    return config


def env_snapshot(config):
    def command(argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True,
                                  check=False, timeout=10).__dict__
        except Exception as error:
            return {"error": repr(error)}
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repo_commit": config["project_commit"],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--porcelain"],
                                      capture_output=True, text=True).stdout,
        "hardware": {
            "npu_device": config["npu_device"], "numa_node": config["numa_node"],
            "raw_pci": config["raw_pci"],
            "raw_pci_info": command(["lspci", "-s", config["raw_pci"], "-nnk"]),
            "findmnt": command(["findmnt", "-no", "TARGET,SOURCE,FSTYPE,OPTIONS", "-T", config["fs_test_dir"]]),
            "lsblk": command(["lsblk", "-o", "NAME,PATH,MODEL,SERIAL,PKNAME,FSTYPE,MOUNTPOINT"]),
            "storage_evidence": storage_evidence(config),
        },
    }


def preflight(config, selected, runtime_probe=False):
    root = Path(config["results_root"]) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    output = {"config": config, "environment": env_snapshot(config), "adapters": {}}
    for name in selected:
        try:
            status = ADAPTERS[name].preflight(config)
            if runtime_probe and name == "ours" and status.get("status") == "ready":
                status = ADAPTERS[name].runtime_probe(config)
            output["adapters"][name] = status
        except Exception as error:
            output["adapters"][name] = {"adapter": name, "status": "build_failed",
                                         "reason": repr(error)}
    write_json(root / "preflight.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def adapter_evidence(name):
    adapter = ADAPTERS[name]
    lines = [
        f"# Adapter: {name}",
        f"# Port class: {adapter.kind}",
        f"# Locked upstream core invoked: {getattr(adapter, 'upstream_core_invoked', None)}",
        "# Mechanisms preserved:",
    ]
    lines.extend(f"# - {item}" for item in getattr(
        adapter, "mechanisms_preserved", ()))
    lines.append("# Platform substitutions:")
    lines.extend(f"# - {item}" for item in getattr(
        adapter, "platform_substitutions", ()))
    return "\n".join(lines) + "\n"


def is_formal_semantic_port(row):
    checkpoints = row.get("checkpoints", ())
    return (
        row.get("status") == "trend_measured"
        and row.get("kind") in {
            "npu-semantic-port", "host-adapted-semantic-port"}
        and int(row.get("formal_steps") or 0) == 30
        and int(row.get("checkpoint_count") or 0) == 10
        and (row.get("restore") or {}).get("status") == "pass"
        and len(checkpoints) == 10
        and all(checkpoint.get("state") == "PERSISTED"
                for checkpoint in checkpoints)
    )


def run_one(config, name, mode="run"):
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter: {name}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["results_root"]) / name / f"{name}_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", env_snapshot(config))
    lock_path = Path(__file__).with_name("upstream.lock.json")
    if lock_path.exists():
        shutil.copy2(lock_path, run_dir / "upstream.lock.json")
    worker_lock = Path(__file__).with_name("worker_environment.lock.json")
    if worker_lock.exists():
        shutil.copy2(worker_lock, run_dir / "worker_environment.lock.json")
    schema_path = Path(config["results_root"]) / "prepare" / "state_schema.json"
    if schema_path.exists():
        shutil.copy2(schema_path, run_dir / "state_schema.json")
    (run_dir / "adapter_diff.patch").write_text(
        adapter_evidence(name), encoding="utf-8")
    (run_dir / "stdout.log").touch()
    (run_dir / "stderr.log").touch()
    try:
        status = ADAPTERS[name].preflight(config)
        if status.get("status") != "ready":
            failure_status = status.get("status", "not_attempted")
            failure = {"status": failure_status, "adapter": name,
                       "error": status.get("reason"), "preflight": status}
            write_json(run_dir / "failure.json", failure)
            return failure
        if mode == "run":
            # Both phases run outside this coordinator.  The restore process
            # is not started until the source interpreter has fully exited.
            source_cmd = [sys.executable, "-m",
                          "experiments.baselines.repro.cli", "source",
                          "--config", str(run_dir / "config.json"),
                          "--adapter", name, "--run-dir", str(run_dir)]
            source_proc = subprocess.run(
                source_cmd, capture_output=True, text=True, check=False)
            (run_dir / "stdout.log").write_text(
                source_proc.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(
                source_proc.stderr, encoding="utf-8")
            source_path = run_dir / "source.json"
            if source_proc.returncode or not source_path.exists():
                failure = {
                    "status": "roundtrip_failed", "adapter": name,
                    "stage": "source", "returncode": source_proc.returncode,
                    "error": "source subprocess did not complete",
                }
                write_json(run_dir / "failure.json", failure)
                return failure
            result = json.loads(source_path.read_text(encoding="utf-8"))
            if name == "none":
                result["restore"] = {"status": "not_applicable",
                                      "reason": "training baseline has no checkpoint"}
                write_json(run_dir / "result.json", result)
                return result
            restore_cmd = [sys.executable, "-m",
                           "experiments.baselines.repro.cli", "restore",
                           "--config", str(run_dir / "config.json"),
                           "--adapter", name, "--run-dir", str(run_dir)]
            restore_proc = subprocess.run(
                restore_cmd, capture_output=True, text=True, check=False)
            (run_dir / "restore.stdout.log").write_text(
                restore_proc.stdout, encoding="utf-8")
            (run_dir / "restore.stderr.log").write_text(
                restore_proc.stderr, encoding="utf-8")
            restore_path = run_dir / "restore.json"
            if restore_path.exists():
                restore = json.loads(restore_path.read_text())
            else:
                restore = {"status": "restore_failed",
                           "error": "restore subprocess produced no restore.json",
                           "returncode": restore_proc.returncode}
            result["restore"] = restore
            write_json(run_dir / "result.json", result)
            return result
        raise ValueError(f"unsupported mode {mode}")
    except DependencyBlocked as error:
        failure = {"status": "dependency_blocked", "adapter": name,
                   "error": repr(error), "next_step": "prepare locked dependency"}
        write_json(run_dir / "failure.json", failure)
        return failure
    except Exception as error:
        failure = {"status": "build_failed", "adapter": name,
                   "error": repr(error)}
        write_json(run_dir / "failure.json", failure)
        return failure


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command_name in ("preflight", "prepare", "smoke", "run", "source",
                         "restore", "inventory", "suite", "summarize"):
        cmd = sub.add_parser(command_name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--all", action="store_true")
        cmd.add_argument("--adapter")
        cmd.add_argument("--continue-on-failure", action="store_true")
        cmd.add_argument("--eligible-only", action="store_true")
        cmd.add_argument("--serial", action="store_true")
        cmd.add_argument("--restore-each", action="store_true")
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--steps", type=int)
        cmd.add_argument("--checkpoint-every", type=int)
        cmd.add_argument("--generation", default="latest-committed")
        cmd.add_argument("--continue-steps", type=int)
        cmd.add_argument("--run-dir")
        cmd.add_argument("--source-run", action="append", default=[])
        cmd.add_argument("--output")
        cmd.add_argument("--mode", choices=("timing", "verify"), default="verify")
        cmd.add_argument("--repeat-index", type=int)
        cmd.add_argument("--runtime-probe", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    # Command-line overrides are intentionally explicit so short G2 smoke runs
    # can reuse the same fixture without silently changing the checked-in plan.
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        config = dict(config)
        config["formal_steps"] = args.steps
    if args.checkpoint_every is not None:
        if args.checkpoint_every <= 0:
            parser.error("--checkpoint-every must be positive")
        config = dict(config)
        config["checkpoint_every"] = args.checkpoint_every
    if args.continue_steps is not None and args.command != "restore":
        if args.continue_steps < 0:
            parser.error("--continue-steps must be non-negative")
        config = dict(config)
        config["continue_steps"] = args.continue_steps
    names = list(ADAPTERS) if args.all or not args.adapter else [args.adapter]
    if args.command == "preflight":
        report = preflight(config, names, runtime_probe=args.runtime_probe)
        statuses = [row.get("status") for row in report.get("adapters", {}).values()]
        return 0 if statuses and all(status in ("ready", "trend_measured") for status in statuses) else 1
    if args.command == "prepare":
        root = Path(config["results_root"]) / "prepare"
        write_json(root / "config.json", config)
        write_json(root / "environment.json", env_snapshot(config))
        lock_path = Path(__file__).with_name("upstream.lock.json")
        if lock_path.exists():
            shutil.copy2(lock_path, root / "upstream.lock.json")
        print(json.dumps(prepare_run(config, root), indent=2, sort_keys=True))
        return 0
    if args.command in ("smoke", "suite"):
        if args.eligible_only:
            statuses = preflight(config, names).get("adapters", {})
            names = [name for name in names
                     if statuses.get(name, {}).get("status") in ("ready", "trend_measured")]
            if not names:
                print(json.dumps({"status": "no_eligible_adapters"}))
                return 1
        if args.dry_run:
            print(json.dumps({"adapters": names, "formal_steps": config["formal_steps"],
                              "checkpoint_count": config["formal_steps"] // config["checkpoint_every"],
                              "serial": True}, indent=2))
            return 0
        failures = 0
        for name in names:
            result = run_one(config, name)
            print(json.dumps(result, sort_keys=True))
            if result.get("status") not in ("trend_measured", "ready"):
                failures += 1
                if not args.continue_on_failure:
                    break
        return 1 if failures and not args.continue_on_failure else 0
    if args.command == "run":
        if not args.adapter:
            parser.error("run requires --adapter")
        result = run_one(config, args.adapter)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "trend_measured" else 1
    if args.command == "source":
        if not args.adapter or not args.run_dir:
            parser.error("source requires --adapter and --run-dir")
        try:
            result = train_run(config, args.adapter, args.run_dir)
        except Exception as error:
            write_json(Path(args.run_dir) / "failure.json",
                       {"status": "roundtrip_failed", "adapter": args.adapter,
                        "stage": "source", "error": repr(error)})
            raise
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "trend_measured" else 1
    if args.command == "restore":
        if not args.adapter or not args.run_dir:
            parser.error("restore requires --adapter and --run-dir")
        try:
            result = restore_run(config, args.adapter, args.run_dir,
                                 generation=args.generation,
                                 continue_steps=args.continue_steps,
                                 output_path=args.output,
                                 mode=args.mode,
                                 repeat_index=args.repeat_index)
        except Exception as error:
            failure_path = Path(args.output) if args.output else Path(args.run_dir) / "restore.json"
            if not failure_path.exists():
                write_json(failure_path,
                       {"status": "restore_failed", "adapter": args.adapter,
                        "error": repr(error), "mode": args.mode,
                        "repeat_index": args.repeat_index})
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "pass" else 1
    if args.command == "inventory":
        if args.source_run:
            source_runs = {}
            for value in args.source_run:
                if "=" not in value:
                    parser.error("--source-run must be adapter=run_dir")
                name, path = value.split("=", 1)
                if name not in ADAPTERS:
                    parser.error(f"unknown adapter: {name}")
                source_runs[name] = path
        else:
            if not args.adapter or not args.run_dir:
                parser.error("inventory requires --adapter and --run-dir")
            source_runs = {args.adapter: args.run_dir}
        output = Path(args.output) if args.output else Path(config["results_root"]) / "checkpoint_inventory.json"
        if output.exists():
            parser.error(f"refusing to overwrite inventory: {output}")
        inventory = build_inventory(config, source_runs)
        write_json(output, inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        root = Path(config["results_root"])
        summary = {"config": config, "runs": []}
        for path in sorted(root.glob("*/**/result.json")):
            row = json.loads(path.read_text())
            row["result_path"] = str(path)
            # Older smoke records predate the explicit storage_backend field;
            # classify them from the adapter name for a useful, non-empty
            # summary while leaving their raw JSON untouched.
            row["legacy_unclassified"] = False
            summary["runs"].append(row)
        summary["matrix"] = [
            {"adapter": row.get("adapter"), "status": row.get("status"),
             "storage_backend": row.get("storage_backend") or
             ("filesystem" if row.get("adapter") == "mindspore_sync" else
              "none" if row.get("adapter") == "none" else "unknown"),
             "fresh_restore": (row.get("restore") or {}).get("status",
                                "not_applicable"),
             "formal_steps": row.get("formal_steps"),
             "checkpoint_count": row.get("checkpoint_count"),
             "result_path": row.get("result_path")}
            for row in summary["runs"]
            if (not row.get("legacy_unclassified") and
                (row.get("adapter") != "ours" or
                 row.get("storage_backend") == "raw_spdk"))
        ]
        latest_formal = {}
        for row in summary["runs"]:
            if not is_formal_semantic_port(row):
                continue
            adapter = row["adapter"]
            current = latest_formal.get(adapter)
            if current is None or row["result_path"] > current["result_path"]:
                latest_formal[adapter] = row
        summary["formal_semantic_ports"] = [
            {
                "adapter": row["adapter"],
                "kind": row["kind"],
                "upstream_core_invoked": row.get("upstream_core_invoked"),
                "formal_steps": row["formal_steps"],
                "checkpoint_count": row["checkpoint_count"],
                "total_wall_seconds": row.get("total_wall_seconds"),
                "state_bytes": row.get("state_bytes"),
                "fresh_restore": row["restore"]["status"],
                "byte_exact": row["restore"].get("byte_exact"),
                "max_loss_deviation": max(
                    row["restore"].get("loss_deviation", [0.0])),
                "mechanisms_preserved": row.get("mechanisms_preserved", []),
                "platform_substitutions": row.get("platform_substitutions", []),
                "result_path": row["result_path"],
            }
            for row in sorted(latest_formal.values(),
                              key=lambda item: item["adapter"])
        ]
        write_json(root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
