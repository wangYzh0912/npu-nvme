#!/usr/bin/env python3
"""Parse A1 PMU CSVs — run via sudo to read msprof output."""
import csv, json, os, glob
from collections import defaultdict

BASE = "/home/user7/npu-nvme/output/profiling_vec/A1/PROF_000001_20260609232800974_DDHQNKRHNCDGGEPA/mindstudio_profiler_output"

# ── op_statistic ──
stat_file = os.path.join(BASE, "op_statistic_20260610025708.csv")
by_core = defaultdict(lambda: {"total_time_us": 0.0, "kernel_count": 0, "ops": []})

with open(stat_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        core = row.get("Core Type", "")
        op_name = row.get("OP Type", "")
        count = int(row.get("Count", 0))
        total_us = float(row.get("Total Time(us)", 0))
        ratio = float(row.get("Ratio(%)", 0))
        if total_us <= 0:
            continue
        d = by_core[core]
        d["total_time_us"] += total_us
        d["kernel_count"] += count
        d["ops"].append({
            "op": op_name, "count": count,
            "total_us": round(total_us, 2),
            "ratio_pct": round(ratio, 2)
        })

total_op_time = sum(v["total_time_us"] for v in by_core.values())
total_kernels = sum(v["kernel_count"] for v in by_core.values())
print("Total op time: {:.2f}s [{:.0f}us]".format(total_op_time/1e6, total_op_time))
print("Total kernels: {}".format(total_kernels))
print()

for ct in sorted(by_core.keys()):
    d = by_core[ct]
    pct = d["total_time_us"] / total_op_time * 100
    ops_sorted = sorted(d["ops"], key=lambda o: o["total_us"], reverse=True)
    print("  {}: {:.2f}s ({:.1f}%), {} kernels".format(
        ct, d["total_time_us"]/1e6, pct, d["kernel_count"]))
    for o in ops_sorted[:12]:
        print("    {:<48} x{:<8} {:>10.1f}ms  {:>6.2f}%".format(
            o["op"], o["count"], o["total_us"]/1e3, o["ratio_pct"]))
    print()

aic_time = by_core.get("AI_CORE", {}).get("total_time_us", 0)
aiv_time = by_core.get("AI_VECTOR_CORE", {}).get("total_time_us", 0)
print("Cube time:  {:.2f}s ({:.1f}%)".format(aic_time/1e6, aic_time/total_op_time*100))
print("Vector time: {:.2f}s ({:.1f}%)".format(aiv_time/1e6, aiv_time/total_op_time*100))

# ── op_summary for PMU ratios ──
sum_file = os.path.join(BASE, "op_summary_20260610025708.csv")
aic_mac_sum = 0.0; aic_n = 0
aiv_vec_sum = 0.0; aiv_scalar_sum = 0.0; aiv_n = 0

with open(sum_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        core = row.get("Core Type", "")
        try:
            if core == "AI_CORE":
                r = float(row.get("aic_mac_ratio", "0"))
                aic_mac_sum += r; aic_n += 1
            elif core == "AI_VECTOR_CORE":
                vr = float(row.get("aiv_vec_ratio", "0"))
                sr = float(row.get("aiv_scalar_ratio", "0"))
                aiv_vec_sum += vr; aiv_scalar_sum += sr
                aiv_n += 1
        except (ValueError, TypeError):
            pass

aic_mac_pct = (aic_mac_sum / max(aic_n, 1)) * 100
aiv_vec_pct = (aiv_vec_sum / max(aiv_n, 1)) * 100
aiv_scalar_pct = (aiv_scalar_sum / max(aiv_n, 1)) * 100
vec_eff = aiv_vec_pct + aiv_scalar_pct
vec_idle_ms = aiv_time / 1e6 * (1 - vec_eff / 100) if vec_eff < 99 else 0

print("\n  PMU Cycle-level (from op_summary):")
print("  Cube aic_mac_ratio:     {:.2f}%".format(aic_mac_pct))
print("  Vector aiv_vec_ratio:   {:.2f}%".format(aiv_vec_pct))
print("  Vector aiv_scalar_ratio: {:.2f}%".format(aiv_scalar_pct))
print("  Vector effective util:  {:.2f}%".format(vec_eff))
print("  Vector idle:            {:.2f}%".format(100 - vec_eff))
print("  Vector idle estimate:   {:.0f}ms".format(vec_idle_ms))

# Save JSON
result = {
    "test": "A1",
    "model": "GPT-2 XL 48L",
    "total_op_time_us": round(total_op_time, 2),
    "aic_time_pct": round(aic_time/total_op_time*100, 2) if total_op_time else 0,
    "aiv_time_pct": round(aiv_time/total_op_time*100, 2) if total_op_time else 0,
    "cube_mac_ratio_pct": round(aic_mac_pct, 2),
    "vec_vec_ratio_pct": round(aiv_vec_pct, 2),
    "vec_scalar_ratio_pct": round(aiv_scalar_pct, 2),
    "vec_eff_util_pct": round(vec_eff, 2),
    "vec_idle_pct": round(100 - vec_eff, 2),
    "vec_idle_ms_est": round(vec_idle_ms, 2),
    "by_core_type": {
        ct: {
            "total_time_us": round(d["total_time_us"], 2),
            "pct": round(d["total_time_us"]/total_op_time*100, 2),
            "kernels": d["kernel_count"],
            "top_ops": sorted(d["ops"], key=lambda o: o["total_us"], reverse=True)[:12]
        }
        for ct, d in by_core.items()
    }
}

out_path = "/home/user7/npu-nvme/experiments/output/phase1a_a1_pmu.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to phase1a_a1_pmu.json")
