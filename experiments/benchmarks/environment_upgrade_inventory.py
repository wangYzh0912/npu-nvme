#!/usr/bin/env python3
"""Capture the active environment before an isolated model-stack upgrade.

This is intentionally read-only.  It records the executable/library paths and
device layout that must remain reproducible while a candidate environment is
tested beside the current one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def command(argv, timeout=15):
    try:
        proc = subprocess.run(argv, text=True, capture_output=True,
                              check=False, timeout=timeout)
        return {"argv": argv, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "returncode": -1, "error": repr(error)}


def path_info(path):
    item = Path(path)
    result = {"path": str(path), "exists": None}
    try:
        stat = item.stat()
        result.update(exists=True, resolved=str(item.resolve()),
                      size=stat.st_size if item.is_file() else None)
    except FileNotFoundError:
        result["exists"] = False
    except OSError as error:
        result["stat_error"] = repr(error)
    return result


def file_record(path):
    result = path_info(path)
    try:
        raw = Path(path).read_bytes()
        result.update(sha256=hashlib.sha256(raw).hexdigest(),
                      content=raw.decode("utf-8", errors="replace"))
    except OSError as error:
        result["read_error"] = repr(error)
    return result


def blockers_for(result):
    blockers = []
    devices = result["devices"]
    if not devices["npu_nodes"]:
        blockers.append("no NPU device nodes are visible in this execution environment")
    if not devices["uio_nodes"]:
        blockers.append("no UIO device nodes are visible; raw NVMe gates cannot run")
    if devices["npu"].get("returncode") != 0:
        blockers.append("NPU runtime health could not be verified")
    if result["packages"].get("returncode") != 0:
        blockers.append("selected Python environment could not be inspected")
    if result["toolkit"]["firmware_version"].get("read_error"):
        blockers.append("firmware version is not readable")
    for model in result["storage"]["models"]:
        if model["config"].get("read_error"):
            blockers.append(f"model config is not readable: {model['path']}")
    for probe in result.get("network", {}).values():
        if probe.get("returncode") != 0:
            blockers.append("official compatibility references or Git remote are inaccessible")
            break
    return blockers


def package_versions(python):
    script = (
        "import importlib.metadata as m, json; "
        "names=['mindspore','mindformers','numpy','torch','spdk']; "
        "out={}; "
        "\nfor n in names:\n"
        "  try: out[n]=m.version(n)\n"
        "  except m.PackageNotFoundError: out[n]=None\n"
        "print(json.dumps(out, sort_keys=True))"
    )
    result = command([python, "-c", script])
    result["python"] = python
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--check-network", action="store_true",
                        help="perform bounded read-only Git/official documentation probes")
    parser.add_argument("--models", nargs="*", default=(
        "/models/Qwen3-0.6B", "/models/Qwen3-1.7B", "/models/Qwen3-4B",
        "/models/Qwen3-8B"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new inventory path")

    toolkit = Path(os.environ.get("ASCEND_HOME_PATH",
                   "/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux"))
    version_root = Path(os.environ.get("NPU_NVME_CANN_VERSION_ROOT",
                        str(toolkit.parent if toolkit.name == "aarch64-linux" else toolkit)))
    npu_nodes = [str(p) for p in Path("/dev").glob("davinci[0-9]*")]
    library = Path(os.environ.get("NPU_NVME_LIBRARY_PATH",
                                 str(args.repo / "build_out/lib/libnpu_nvme.so")))

    env_keys = ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "SPDK_ROOT_DIR",
                "SOC_VERSION", "NPU_NVME_DPDK_ARGS")
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(args.repo.resolve()),
        "python": path_info(args.python),
        "collector_python_version": platform.python_version(),
        "python_version": command([args.python, "-I", "-c",
                                   "import platform; print(platform.python_version())"]),
        "environment_id": os.environ.get("NPU_NVME_ENVIRONMENT_ID"),
        "platform": {"system": platform.system(), "release": platform.release(),
                      "machine": platform.machine()},
        "environment": {key: os.environ.get(key) for key in env_keys},
        "packages": package_versions(args.python),
        "git": {
            "status": command(["git", "-C", str(args.repo), "status", "--short"]),
            "head": command(["git", "-C", str(args.repo), "log", "-1", "--oneline"]),
            "submodules": command(["git", "-C", str(args.repo), "submodule", "status"]),
        },
        "toolkit": {
            "selected": path_info(toolkit),
            "runtime_version": file_record(version_root / "runtime/version.info"),
            "hccl_version": file_record(version_root / "hccl/version.info"),
            "driver_version": file_record("/usr/local/Ascend/driver/version.info"),
            "firmware_version": file_record("/usr/local/Ascend/firmware/version.info"),
            "system_env": file_record("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
            "ascend_root": path_info("/usr/local/Ascend/ascend-toolkit"),
            "set_env": path_info("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
            "latest_lib": path_info("/usr/local/Ascend/ascend-toolkit/latest/lib64"),
            "opp": path_info("/usr/local/Ascend/ascend-toolkit/latest/opp"),
        },
        "build": {
            "library": path_info(library),
            "spdk": path_info(args.repo / "third_party/spdk"),
            "dpdk_fixed_ring": path_info(args.repo / "build/dpdk_fix/librte_mempool_ring_fixed.a"),
        },
        "devices": {
            "npu_nodes": npu_nodes,
            "uio_nodes": [str(p) for p in Path("/dev").glob("uio*")],
            "npu": command(["npu-smi", "info"]) if npu_nodes else {
                "returncode": None, "status": "blocked", "reason": "no NPU devices exposed"},
            "nvme_raw": command(["lspci", "-s", "83:00.0", "-nnk"]),
            "nvme_fs": command(["lspci", "-s", "84:00.0", "-nnk"]),
            "uio": command(["bash", "-lc", "ls -l /dev/uio* 2>&1"]),
            "drivers": {
                pci83: path_info(f"/sys/bus/pci/devices/{pci83}/driver")
                for pci83 in ("0000:83:00.0", "0000:84:00.0")
            },
        },
        "storage": {
            "models_mount": command(["findmnt", "-no", "TARGET,SOURCE,FSTYPE,OPTIONS", "-T", "/models"]),
            "home_df": command(["df", "-h", "/home"]),
            "models_df": command(["df", "-h", "/models"]),
            "models": [{**path_info(path), "config": file_record(Path(path) / "config.json")}
                       for path in args.models],
        },
        "memory": {
            "meminfo": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None,
            "hugepages": command(["grep", "-E", "Huge(Page|pages)", "/proc/meminfo"]),
        },
        "runtime_paths": {
            "ldd_library": command(["ldd", str(library)]),
            "command_line": shlex.join(sys.argv),
        },
    }
    if args.check_network:
        result["network"] = {
            "mindformers_install": command([
                "curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "10",
                "https://www.mindspore.cn/mindformers/docs/zh-CN/r1.8.0/installation.html"], timeout=12),
            "remote": command(["git", "-C", str(args.repo), "-c",
                "core.sshCommand=ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8",
                "ls-remote", "origin", "HEAD"], timeout=12),
        }
    result["blockers"] = blockers_for(result)
    result["status"] = "blocked" if result["blockers"] else "inventory_complete"
    # An inventory, even a successful one, is not a CANN compatibility or
    # full-state model acceptance test.
    result["compatibility"] = "not_determined"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "timestamp": result["timestamp"],
                      "status": result["status"], "blockers": result["blockers"]},
                     sort_keys=True), flush=True)
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
