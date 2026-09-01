"""Small durable helpers shared by the long-running FULL validation runners."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checked_stdout(result, label):
    """Extract a stable command value or fail instead of recording ambiguity."""
    if result.get("returncode") != 0:
        raise RuntimeError(f"{label} failed: {result.get('stderr', '').strip()}")
    value = result.get("stdout", "").strip()
    if not value:
        raise RuntimeError(f"{label} returned empty output")
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    default=str), encoding="utf-8")
    os.replace(temporary, path)


def open_campaign(path, commit, config, resume=False):
    path = Path(path)
    digest = config_digest(config)
    if path.exists():
        campaign = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError(f"campaign already exists; use --resume: {path}")
        if campaign.get("commit") != commit:
            raise RuntimeError("resume commit differs from campaign commit")
        if campaign.get("config_digest") != digest:
            raise RuntimeError("resume configuration differs from campaign")
        return campaign
    campaign = {
        "schema_version": 1,
        "commit": commit,
        "config_digest": digest,
        "config": config,
        "created_unix_ns": time.time_ns(),
        "updated_unix_ns": time.time_ns(),
        "entries": {},
    }
    atomic_json(path, campaign)
    return campaign


def update_entry(path, campaign, key, status, **fields):
    campaign["entries"][key] = {
        **campaign["entries"].get(key, {}),
        **fields,
        "status": status,
        "updated_unix_ns": time.time_ns(),
    }
    campaign["updated_unix_ns"] = time.time_ns()
    atomic_json(path, campaign)


def completed_result(campaign, key, result_path):
    entry = campaign.get("entries", {}).get(key, {})
    path = Path(result_path)
    if entry.get("status") != "pass" or not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return result if result.get("status") == "pass" else None
