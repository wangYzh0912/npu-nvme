"""Render a compact machine-readable and Markdown campaign summary."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
from npu_nvme.runtime.training_catalog import read_checked


def render(campaign):
    campaign = Path(campaign).resolve()
    manifest = json.loads((campaign / "campaign.json").read_text())
    rows = []
    rejected = []
    for path in sorted((campaign / "runs").glob("*")):
        outcome=path/'result.json'
        if not outcome.exists() or json.loads(outcome.read_text()).get('validation_status')!='pass':
            rejected.append(dict(run=path.name,error='supervisor did not validate this run'))
            continue
        rank_reports = []
        for rank in range(4):
            file = path / f"rank_{rank}" / "training.json"
            if file.exists():
                try:
                    rank_reports.append(read_checked(file))
                except (ValueError, KeyError) as error:
                    rejected.append(dict(run=path.name, rank=rank, error=str(error)))
        if len(rank_reports) != 4:
            rejected.append(dict(run=path.name, error='incomplete rank reports'))
            continue
        if any(row.get('status') != 'pass' or 'formal_end_ns' not in row.get('incremental', {}) for row in rank_reports):
            rejected.append(dict(run=path.name, error='run failed or formal interval incomplete',
                                 ranks=[dict(rank=row['rank'], status=row['status'], error=row.get('error')) for row in rank_reports]))
            continue
        experiment = rank_reports[0].get("incremental", {})
        formal = (max(row["incremental"]["formal_end_ns"] for row in rank_reports) -
                  min(row["incremental"]["formal_begin_ns"] for row in rank_reports)) / 1e9
        rows.append({"run": path.name, "group": experiment.get("group"), "formal_seconds": formal,
                     "profiled": experiment.get('profiled', False),
                     "role": 'auxiliary' if path.name.startswith('auxiliary-') else 'main',
                     "canonical": '-canonical' in path.name,
                     "probe": any(r.get('probe') for r in experiment.get('steps',[])),
                     "all_done_seconds":(max(r['incremental'].get('all_done_ns',r['incremental']['formal_end_ns']) for r in rank_reports)-min(r['incremental']['formal_begin_ns'] for r in rank_reports))/1e9,
                     "peak_hbm_bytes_per_rank": [r['incremental'].get('memory_peak_bytes') for r in rank_reports],
                     "drain_seconds": max(row["incremental"].get("drain_ns", 0) for row in rank_reports) / 1e9,
                     "payload_bytes": sum(sum(step.get("payload_bytes", 0) for step in row["incremental"].get("steps", []))
                                          for row in rank_reports)})
    groups = {}
    for row in rows:
        if not (row['profiled'] or row['canonical'] or row['probe']):
            groups.setdefault(row['role']+':'+row['group'], []).append(row['formal_seconds'])
    summary = {group: {"samples": values, "median_seconds": statistics.median(values),
                        "spread_fraction": (max(values)-min(values))/statistics.median(values)
                        if values else None}
               for group, values in groups.items()}
    probe = campaign / 'score-probe' / 'result.json'
    workload = {"scope": "weights", "dtype_bytes": 4,
                "score_ops_per_element": 3, "score_reads_bytes_per_element": 8,
                "candidate_blocks": None, "note": "parameter inventory required for K; no full model copy made"}
    if probe.exists(): workload["device_score_probe"] = json.loads(probe.read_text())
    budget = campaign/'resource-budget-main.json'
    profile = campaign/'profile-summary-rank0.json'
    load = campaign/'load-boundary.json';copy = campaign/'copy-boundary.json'
    document = {"manifest": manifest, "runs": rows, "rejected_runs": rejected, "groups": summary,
                "phase_status": {name: row['status'] for name,row in manifest['phases'].items()},
                "workload_estimate": workload,
                "resource_budget": json.loads(budget.read_text()) if budget.exists() else None,
                "load_boundary": json.loads(load.read_text()) if load.exists() else None,
                "copy_boundary": json.loads(copy.read_text()) if copy.exists() else None,
                "resource_metrics": json.loads(profile.read_text()) if profile.exists() else {
                    "vector_cube_timeline": "missing: profiler trace not collected",
                    "kernel_occupancy": "missing: tool metric unavailable",
                    "hbm_bandwidth": "missing: synchronized timeline not collected"}}
    (campaign / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    lines = ["# Incremental Checkpoint Phase 1", "", f"Campaign status: `{manifest['status']}`", "",
             "Only complete, checksum-verified rank reports enter the table. Profiled runs are excluded from timing summaries.", "",
             "| Group | Runs | Median formal seconds | Spread |", "|---|---:|---:|---:|"]
    for group, value in sorted(summary.items()):
        lines.append(f"| {group} | {len(value['samples'])} | {value['median_seconds']:.6f} | {value['spread_fraction']:.2%} |")
    lines += ["", "## Phase Status", ""]
    lines += [f"- {name}: {row['status']}" for name,row in manifest['phases'].items()]
    lines += ["", f"Rejected or incomplete runs: {len(rejected)}.",
              "", "Profiler task intervals do not prove core occupancy or available bandwidth."]
    if profile.exists():
        value=json.loads(profile.read_text())
        lines += ["", "## Resource Timeline", "",
                  "| State | Fraction |", "|---|---:|",
                  *[f"| {name} | {fraction:.4%} |" for name,fraction in value['fractions'].items()],
                  "", f"Cube-only candidate windows: {value['candidate_count']}; longest {value['candidate_longest']:.3f} us."]
    (campaign / "REPORT.md").write_text("\n".join(lines) + "\n")
    return 0
