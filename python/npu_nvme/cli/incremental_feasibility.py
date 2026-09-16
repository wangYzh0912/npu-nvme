"""CLI supervisor for the incremental-checkpoint phase-one campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from npu_nvme.experiments.config import load


ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _driver(pci):
    path = Path("/sys/bus/pci/devices") / pci / "driver"
    return path.resolve().name if path.exists() else None


def _model_identity(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = []
    for name in sorted(set(index["weight_map"].values())):
        shard = (path / name).resolve()
        if path.resolve() not in shard.parents:
            raise ValueError("model shard escapes model directory")
        shards.append({"name": name, "bytes": shard.stat().st_size, "sha256": digest_file(shard)})
    expected = {"main": (4096, 36), "auxiliary": (2560, 36)}
    return config, shards, expected


def preflight(config):
    authorization = json.loads(Path(config["authorization"]).read_text())
    if (authorization.get("write_authorized") is not True or
            authorization.get("pci_addr") != config["raw_pci_addr"] or
            authorization.get("offset") != 0 or authorization.get("length") != "device_capacity_bytes"):
        raise ValueError("whole-device raw authorization differs")
    if authorization.get("protected_pci_addr") != config["protected_pci_addr"]:
        raise ValueError("protected PCI identity differs")
    models = {}
    expected = {"main": (4096, 36), "auxiliary": (2560, 36)}
    for role, path in config["models"].items():
        model, shards, _ = _model_identity(path)
        identity = (model.get("hidden_size"), model.get("num_hidden_layers"))
        if model.get("model_type") != "qwen3" or identity != expected[role]:
            raise ValueError(f"{role} model identity differs")
        models[role] = {"path": path, "config": model, "shards": shards}
    library = Path(config["library"])
    manifest = Path(config["environment_manifest"])
    if not library.is_file() or not manifest.is_file():
        raise FileNotFoundError("selected runtime is incomplete")
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
    idle = all(f"No running processes found in NPU {rank}" in smi for rank in range(8))
    lease = Path(config["hardware_lock"]).with_suffix(".lease.json")
    result = {
        "schema_version": 1, "status": "pass", "monotonic_ns": time.monotonic_ns(),
        "config_sha256": config["config_sha256"], "models": models,
        "runtime": {"manifest": str(manifest), "library": str(library),
                    "library_sha256": digest_file(library)},
        "storage": {"raw_pci_addr": config["raw_pci_addr"],
                    "raw_driver": _driver(config["raw_pci_addr"]),
                    "protected_pci_addr": config["protected_pci_addr"],
                    "protected_driver": _driver(config["protected_pci_addr"]),
                    "authorization": authorization},
        "devices_idle": idle, "pending_lease": str(lease) if lease.exists() else None,
        "npu_info": smi,
    }
    if result["storage"]["raw_driver"] != "uio_pci_generic":
        raise RuntimeError("raw PCI device is not bound to uio_pci_generic")
    if result["storage"]["protected_driver"] != "nvme":
        raise RuntimeError("protected PCI device is not bound to nvme")
    if not idle or lease.exists():
        raise RuntimeError("hardware is occupied or has an unreconciled lease")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "campaign", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/incremental_feasibility.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("p0", "p1", "p2", "p3", "p4", "p5", "p6", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load(args.config, ROOT)
    if args.command == "preflight":
        result = preflight(config); write(args.output, result)
        print(json.dumps(result, indent=2)); return 0
    if args.command == "campaign":
        from npu_nvme.experiments.campaign import run
        return run(config, args.output, phase=args.phase, resume=args.resume)
    from npu_nvme.experiments.report import render
    return render(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
