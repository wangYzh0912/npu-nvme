"""Strict configuration for the incremental-checkpoint phase-one campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _path(value, repo, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path")
    path = Path(value)
    return str((path if path.is_absolute() else repo / path).resolve())


def resolve(value, repo):
    repo = Path(repo).resolve()
    value = dict(value)
    required = {
        "schema_version", "environment_manifest", "library", "hardware_lock",
        "authorization", "protected_pci_addr", "raw_pci_addr", "models",
        "topology", "devices", "shadow_devices", "sequence_length",
        "micro_batch_size", "formal_steps", "warmup_steps",
        "performance_repeats", "extra_repeats_on_instability",
        "baseline_spread_limit", "block_elements", "small_parameter_elements",
        "top_ratios", "hbm_staging_bytes", "transport_chunk_bytes",
        "transport_probe_granularities", "max_inflight", "hbm_reserve_bytes",
        "slowdown_reference_lines", "seed", "loss_rtol", "loss_atol",
    }
    if set(value) != required:
        raise ValueError("incremental settings differ: " + repr(sorted(set(value) ^ required)))
    if value["schema_version"] != 1:
        raise ValueError("incremental settings require schema_version 1")
    for name in (
        "sequence_length", "micro_batch_size", "formal_steps", "warmup_steps",
        "performance_repeats", "block_elements", "small_parameter_elements",
        "hbm_staging_bytes", "transport_chunk_bytes", "max_inflight",
        "hbm_reserve_bytes",
    ):
        _positive_int(value[name], name)
    if type(value["extra_repeats_on_instability"]) is not int or value["extra_repeats_on_instability"] < 0:
        raise ValueError("extra_repeats_on_instability must be nonnegative")
    if value["small_parameter_elements"] != value["block_elements"]:
        raise ValueError("phase one requires the small-parameter boundary to equal one block")
    if value["max_inflight"] != 1:
        raise ValueError("phase one permits exactly one incomplete increment")
    if value["topology"] != {"tp": 4, "dp": 1, "pp": 1}:
        raise ValueError("phase one requires TP4/DP1/PP1")
    if value["devices"] != [0, 1, 2, 3] or value["shadow_devices"] != [4, 5, 6, 7]:
        raise ValueError("phase-one device assignment differs")
    for name in ("baseline_spread_limit", "loss_rtol", "loss_atol"):
        if type(value[name]) not in (int, float) or not 0 < value[name] < 1:
            raise ValueError(f"invalid fraction: {name}")
    ratios = value["top_ratios"]
    if ratios != [0.05, 0.10, 0.20]:
        raise ValueError("phase one requires Top 5/10/20 percent")
    if value["slowdown_reference_lines"] != [0.01, 0.03, 0.05]:
        raise ValueError("phase-one slowdown reference lines differ")
    granularities = value["transport_probe_granularities"]
    if granularities != [262144, 4194304]:
        raise ValueError("phase-one transport granularities differ")
    if any(value["hbm_staging_bytes"] % size for size in granularities):
        raise ValueError("HBM staging must be divisible by both transport granularities")
    models = value["models"]
    if set(models) != {"main", "auxiliary"}:
        raise ValueError("main and auxiliary models are required")
    result = dict(value)
    for name in ("environment_manifest", "library", "hardware_lock", "authorization"):
        result[name] = _path(value[name], repo, name)
    result["models"] = {name: _path(path, repo, f"models.{name}") for name, path in models.items()}
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def load(path, repo):
    return resolve(json.loads(Path(path).read_text()), repo)

