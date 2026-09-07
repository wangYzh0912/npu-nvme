"""Checkpoint inventory and storage-endpoint evidence for recovery comparisons."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                             check=False, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout,
                "stderr": proc.stderr}
    except Exception as error:
        return {"error": repr(error)}


def storage_evidence(config):
    raw = config.get("raw_pci")
    fs_path = config.get("fs_test_dir")
    return {
        "fs_test_dir": fs_path,
        "findmnt": _command(["findmnt", "-no", "TARGET,SOURCE,FSTYPE,OPTIONS", "-T", fs_path]),
        "lsblk": _command(["lsblk", "-o", "NAME,PATH,MODEL,SERIAL,PKNAME,FSTYPE,MOUNTPOINT"]),
        "raw_pci": raw,
        "raw_pci_info": _command(["lspci", "-s", raw, "-nnk"]),
        "same_physical_storage_verified": bool(config.get("same_physical_storage_verified")),
    }


def build_inventory(config, source_runs):
    project_root = Path(config.get("project_root", "."))
    prepare_root = project_root / "results/baseline-gpt2-repro-20260906/prepare"
    fixture = prepare_root / "initial_state/metadata.json"
    batches = prepare_root / "batches.npz"
    rows = []
    for adapter, run_path in source_runs.items():
        run_dir = Path(run_path)
        source_path = run_dir / "source.json"
        row = {"adapter": adapter, "run_dir": str(run_dir.resolve()),
               "status": "missing", "checkpoint_count": 0}
        if not source_path.exists():
            row["reason"] = "source.json missing"
            rows.append(row)
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        checkpoints = source.get("checkpoints", [])
        row.update({"status": "present",
                    "formal_steps": source.get("formal_steps"),
                    "state_bytes": source.get("state_bytes"),
                    "checkpoint_count": len(checkpoints),
                    "storage_backend": source.get("storage_backend"),
                    "checkpoints": []})
        for checkpoint in checkpoints:
            generation = int(checkpoint["generation"])
            if adapter == "ours":
                directory = Path(config["fs_test_dir"]) / "repro_checkpoints" / run_dir.name / f"generation_{generation:06d}"
            else:
                directory = Path(config["fs_test_dir"]) / "repro_checkpoints" / run_dir.name / f"generation_{generation:06d}"
            meta_candidates = [directory / "metadata.json", directory / "bridge.json"]
            meta_path = next((path for path in meta_candidates if path.exists()), None)
            item = {"generation": generation, "step": checkpoint.get("step"),
                    "directory": str(directory), "exists": directory.exists(),
                    "metadata": str(meta_path) if meta_path else None}
            if meta_path:
                item["metadata_sha256"] = _sha256(meta_path)
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    item["metadata_generation"] = metadata.get("generation")
                    item["metadata_step"] = metadata.get("step")
                    item["state_sha256"] = metadata.get("sha256")
                    item["schema_fields"] = len(metadata.get("schema", {}).get("fields", []))
                except Exception as error:
                    item["metadata_error"] = repr(error)
            payloads = [path for path in directory.glob("*") if path.is_file() and not path.name.endswith(".json")]
            item["payload_files"] = [{"path": str(path), "bytes": path.stat().st_size} for path in payloads]
            row["checkpoints"].append(item)
        rows.append(row)
    return {
        "runner": {
            "git_head": _command(["git", "-C", str(project_root), "rev-parse", "HEAD"]),
            "git_status": _command(["git", "-C", str(project_root), "status", "--porcelain"]),
            "project_commit": config.get("project_commit"),
        },
        "inputs": {
            "fixture_metadata": str(fixture),
            "fixture_metadata_sha256": _sha256(fixture) if fixture.exists() else None,
            "batches": str(batches),
            "batches_sha256": _sha256(batches) if batches.exists() else None,
        },
        "storage": storage_evidence(config), "methods": rows,
    }
