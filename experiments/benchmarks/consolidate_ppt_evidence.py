#!/usr/bin/env python3
"""Consolidate already completed E0 measurements into the PPT evidence schema.

This command is deliberately an importer, not a benchmark runner.  It copies
references and derived statistics, never reuses failed samples, and labels old
measurements as historical/cross-disk where their protocol differs from the
new E1 protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from ppt_evidence import (EvidenceBundle, environment_snapshot, sha256_file,
                          stats)  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source(path, *, label, metadata=None):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path), "label": label,
            "metadata": metadata or {}}


def add_metric(table, *, experiment, label, path, metric, value, unit="ms",
               metadata=None):
    table.setdefault(experiment, []).append({
        "label": label, "metric": metric, "value": float(value), "unit": unit,
        "source": source(path, label="historical source", metadata=metadata),
    })


def result_metric(table, experiment, label, path, summary_key, *, metric=None,
                  metadata=None):
    data = read_json(path)
    if data.get("status") != "pass":
        return False
    summary = data.get("summary", {}).get(summary_key)
    if not isinstance(summary, dict) or summary.get("mean") is None:
        return False
    add_metric(table, experiment=experiment, label=label, path=path,
               metric=metric or summary_key, value=summary["mean"],
               metadata={"source_n": summary.get("n"), **(metadata or {})})
    return True


def add_existing_run(bundle, table, path, group, *, metadata=None):
    path = Path(path)
    data = read_json(path)
    bundle.add_sample({"status": data.get("status"), "group": group,
                       "source_path": str(path),
                       "source_sha256": sha256_file(path),
                       "source_samples": data.get("samples"),
                       "metadata": metadata or {}})
    table.setdefault("runs", []).append({
        "group": group, "path": str(path), "sha256": sha256_file(path),
        "status": data.get("status"), "samples": data.get("samples"),
    })


def collect(root, bundle=None):
    table = {"runs": []}
    wp2 = root / "results/wp2-20260825/evidence"
    wp1 = root / "results/wp1-20260825/evidence"

    # E0-1: same-device historical 256 MiB comparison.  v2 has a complete
    # 10-sample formal section; the original files are retained as provenance.
    for name, label in (("same_device_fs_256m_v2.jsonl", "Buffered FS"),
                        ("same_device_fs_odirect_256m_v2.jsonl", "O_DIRECT"),
                        ("same_device_spdk_256m_v2.jsonl", "SPDK")):
        path = wp1 / name
        if not path.exists():
            continue
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # The historical file begins with a pretty-printed config;
                # formal samples and the final summary are one JSON object per
                # line and remain importable.
                continue
        summary = next((r.get("summary") for r in records
                        if isinstance(r, dict) and "summary" in r), {})
        for key in ("write_persist", "read", "end_to_end"):
            value = summary.get(key, {}).get("mean")
            if value is not None:
                add_metric(table, experiment="E0-1", label=label, path=path,
                           metric=key, value=value,
                           metadata={"source_n": summary[key].get("n"),
                                     "persist_boundary": "historical fdatasync/flush"})

    model13 = wp2 / "model_13b"
    # E0-2: scalar versus aggregated submission.
    for name, label in (("model_gpt2_13b_a3_scalar", "scalar submission"),
                        ("model_gpt2_13b_a2_p4", "batched submission")):
        path = model13 / name / "result.json"
        for key in ("write_ms", "read_ms"):
            result_metric(table, "E0-2", label, path, key,
                          metric=key.replace("_ms", ""),
                          metadata={"model": "gpt2_13b", "pipeline_depth": 4})

    # E0-3: depth matrix, all successful formal runs.
    depth_names = {1: "model_gpt2_13b_a4_d1", 2: "model_gpt2_13b_a4_d2_v2",
                   4: "model_gpt2_13b_a2_p4", 8: "model_gpt2_13b_a4_d8",
                   16: "model_gpt2_13b_a4_d16_v2"}
    for depth, name in depth_names.items():
        path = model13 / name / "result.json"
        for key in ("write_ms", "read_ms"):
            result_metric(table, "E0-3", f"depth={depth}", path, key,
                          metric=key.replace("_ms", ""),
                          metadata={"model": "gpt2_13b", "pipeline_depth": depth})

    # E0-4: chunk matrix.  4 MiB is the E0-3 depth=4 control and is included
    # explicitly to make the chart self-contained.
    chunk_names = {65536: "a5_extreme_summary.json", 262144: "a5_extreme_summary.json",
                   1048576: "model_gpt2_13b_a5_1m", 4194304: "model_gpt2_13b_a2_p4",
                   16777216: "model_gpt2_13b_a5_16m"}
    for chunk, name in chunk_names.items():
        path = model13 / name if name.endswith("result.json") else model13 / name / "result.json"
        if name == "a5_extreme_summary.json":
            path = wp2 / "../model_13b/a5_extreme_summary.json"
            data = read_json(path)
            item = next((x for x in data.get("results", [])
                         if x.get("chunk_size_bytes") == chunk), None)
            if item:
                for key in ("write_mean_ms", "read_mean_ms"):
                    add_metric(table, experiment="E0-4", label=f"chunk={chunk}",
                               path=path, metric=key.replace("_mean_ms", ""),
                               value=item[key], metadata={"model": "gpt2_13b",
                               "chunk_size": chunk, "source_n": data.get("repetitions")})
            continue
        for key in ("write_ms", "read_ms"):
            result_metric(table, "E0-4", f"chunk={chunk}", path, key,
                          metric=key.replace("_ms", ""),
                          metadata={"model": "gpt2_13b", "chunk_size": chunk})

    # E0-5: source summary is the authoritative slow-disk HBM slot matrix.
    slow = root / "results/wp2-20260826/a9_hbm/summary.json"
    if slow.exists():
        data = read_json(slow)
        for slot, item in data.get("slow_disk", {}).get("results", {}).items():
            for key in ("slot_wait_mean_ms", "end_to_end_mean_ms"):
                if item.get(key) is not None:
                    add_metric(table, experiment="E0-5", label=f"slots={slot}",
                               path=slow, metric=key, value=item[key],
                               metadata={"slot_count": int(slot),
                                         "delay_ms": data["slow_disk"].get("delay_ms", 5000),
                                         "interpretation": "foreground wait; not physical write time"})

    # E0-6: recovery and fault sources are retained as run-level evidence.
    for path, group in ((root / "results/wp3-closeout-20260826/r0_cpu/result.json", "FULL+Delta recovery"),
                        (root / "results/wp3-closeout-20260826/i6_ring/I6_RAW_RING_20260826_173634_47ca1918/result.json", "ring/recovery"),
                        (root / "results/wp2-20260825/evidence/model_xl/a8_formal/result.json", "metadata/ACK")):
        if path.exists() and bundle is not None:
            add_existing_run(bundle, table, path, group)

    # E0-7: import complete trajectory and policy decision files as provenance;
    # the detailed samples remain in their original directories.
    for path in sorted((root / "results/incremental-next-20260828").glob("trajectory-*/**/result.json")):
        if bundle is not None:
            add_existing_run(bundle, table, path, "trajectory estimate")
    for path in sorted((root / "results/incremental-next-20260828").glob("decision-*/result.json")):
        if bundle is not None:
            add_existing_run(bundle, table, path, "R2 decision")

    for experiment, rows in list(table.items()):
        if experiment == "runs":
            continue
        for row in rows:
            row["value_stats"] = stats([r["value"] for r in rows
                                         if r["metric"] == row["metric"]
                                         and r.get("value") is not None])
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    table = collect(root)
    bundle = EvidenceBundle("E0", {
        "model": "historical mixed models", "seed": "historical",
        "mode": "import-only", "protocol": "pre-PPT evidence; not rerun",
        "device_policy": "historical sources; E1 uses 84 FS and 83 SPDK",
    }, repo_root=root, environment=environment_snapshot(
        pci="historical sources", npu="historical", numa="historical",
        repo_root=root))
    # collect() needs a bundle only for E0-6/E0-7; rebuild those source records
    # in the same run after creating it.
    table = collect_with_bundle(root, bundle)
    (bundle.raw_dir / "source_manifest.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {"model": "historical mixed models", "mode": "import-only",
               "logical_bytes": None, "physical_bytes": None,
               "fault_results": "see raw/source_manifest.json",
               "source_count": len(table.get("runs", [])) + sum(
                   len(v) for k, v in table.items() if k != "runs")}
    result = bundle.finalize(metrics=metrics, status="pass")
    print(json.dumps({"run_dir": str(bundle.run_dir), "result": result},
                     indent=2, sort_keys=True))


def collect_with_bundle(root, bundle):
    """Collect once while preserving the bundle's run-level source samples."""
    # The metric collector is side-effect free; the source run additions are
    # duplicated here only because EvidenceBundle is intentionally created
    # after the data scan to keep its commit snapshot at command execution.
    return collect(root, bundle=bundle)


if __name__ == "__main__":
    main()
