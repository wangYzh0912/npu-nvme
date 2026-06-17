#!/usr/bin/env python3
"""Parse A2_50 op_summary CSV and compare with A1 baseline."""
import csv, json, os
from collections import defaultdict

A2_CSV = "/home/user7/npu-nvme/output/profiling_vec/A2_50/PROF_000001_20260610145353086_AQLPENGJRNGCDHIB/mindstudio_profiler_output/op_summary_20260610184049.csv"
A1_PMU = "/home/user7/npu-nvme/experiments/output/phase1a_a1_pmu.json"
OUT = "/home/user7/npu-nvme/experiments/output/phase1a_a2_50_pmu.json"

# Load A1 baseline
with open(A1_PMU) as f:
    a1 = json.load(f)

# Parse A2
aic_count = 0; aiv_count = 0; aicpu_count = 0
aic_mac_sum = 0.0; aic_scalar_sum = 0.0
aiv_vec_sum = 0.0; aiv_scalar_sum = 0.0
aic_dur_sum = 0.0; aiv_dur_sum = 0.0; aicpu_dur_sum = 0.0

# Per-op-type tracking for delta ops
delta_ops = defaultdict(lambda: {"count": 0, "dur_us": 0.0, "core_type": None})
all_op_types = defaultdict(lambda: {"count": 0, "dur_us": 0.0, "core_type": None})

with open(A2_CSV) as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"CSV header: {len(header)} columns")
    # Map: 7=Task Type, 8=Task Start Time(us), 9=Task Duration(us)
    # 24=aic_mac_ratio, 25=aic_scalar_time(us), 26=aic_scalar_ratio
    # 34=aiv_vec_ratio, 35=aiv_scalar_time(us), 36=aiv_scalar_ratio
    for row in reader:
        if len(row) < 42:
            continue
        try:
            core = row[7].strip()
            op_name = row[4].strip()
            op_type = row[5].strip()
            dur_us = float(row[9].strip())
        except (IndexError, ValueError):
            continue

        all_op_types[op_type]["count"] += 1
        all_op_types[op_type]["dur_us"] += dur_us
        all_op_types[op_type]["core_type"] = core

        # Track "delta" related ops: Sub, ReduceSum, Cast, Add, Reshape, Concat, ZerosLike
        if op_type in ("Sub", "ReduceSum", "Cast", "Add", "Reshape", "Concat", "ZerosLike", "OnesLike"):
            delta_ops[op_type]["count"] += 1
            delta_ops[op_type]["dur_us"] += dur_us
            delta_ops[op_type]["core_type"] = core

        if core == "AI_CORE":
            aic_count += 1; aic_dur_sum += dur_us
            # Column indices: 24=aic_mac_ratio, 26=aic_scalar_ratio
            # 37=aiv_vec_ratio, 39=aiv_scalar_ratio
            # A2_50 CSV: 46 columns
            # 24=aic_mac_ratio, 26=aic_scalar_ratio
            # 37=aiv_vec_ratio, 39=aiv_scalar_ratio
            try:
                aic_mac_sum += float(row[24].strip())
                aic_scalar_sum += float(row[26].strip())
            except (ValueError, IndexError):
                pass
        elif core == "AI_VECTOR_CORE":
            aiv_count += 1; aiv_dur_sum += dur_us
            try:
                aiv_vec_sum += float(row[37].strip())
                aiv_scalar_sum += float(row[39].strip())
            except (ValueError, IndexError):
                pass
        elif core == "AI_CPU":
            aicpu_count += 1; aicpu_dur_sum += dur_us

aic_mac_pct  = (aic_mac_sum  / max(aic_count, 1)) * 100
aic_scal_pct = (aic_scalar_sum / max(aic_count, 1)) * 100
aiv_vec_pct  = (aiv_vec_sum  / max(aiv_count, 1)) * 100
aiv_scal_pct = (aiv_scalar_sum / max(aiv_count, 1)) * 100

cube_eff = aic_mac_pct
vec_eff  = aiv_vec_pct + aiv_scal_pct
vec_idle_pct = 100 - vec_eff
vec_idle_ms = aiv_dur_sum / 1000 * (1 - vec_eff / 100)
aic_time_ms = aic_dur_sum / 1000
aiv_time_ms = aiv_dur_sum / 1000
total_op_time = aic_dur_sum + aiv_dur_sum + aicpu_dur_sum
aic_time_pct = aic_dur_sum / total_op_time * 100 if total_op_time else 0
aiv_time_pct = aiv_dur_sum / total_op_time * 100 if total_op_time else 0

print("=" * 70)
print("A2_50 PMU Analysis (50 params injected)")
print("=" * 70)
print()
print(f"  Total kernels: {aic_count + aiv_count + aicpu_count}")
print(f"  AI_CORE:       {aic_count:>8} rows, {aic_dur_sum/1e6:.2f}s")
print(f"  AI_VECTOR_CORE:{aiv_count:>8} rows, {aiv_dur_sum/1e6:.2f}s")
print(f"  AI_CPU:        {aicpu_count:>8} rows, {aicpu_dur_sum/1e6:.2f}s")
print()
print("  AI_CORE:")
print(f"    MAC ratio avg:    {aic_mac_pct:.2f}%")
print(f"    Scalar ratio avg: {aic_scal_pct:.2f}%")
print(f"    Cube effective:   {cube_eff:.2f}%")
print()
print("  AI_VECTOR_CORE:")
print(f"    vec_ratio avg:    {aiv_vec_pct:.2f}%")
print(f"    scalar_ratio avg: {aiv_scal_pct:.2f}%")
print(f"    Vector effective: {vec_eff:.2f}%")
print(f"    Vector idle:      {vec_idle_pct:.2f}%")
print(f"    Vector idle est:  {vec_idle_ms:.0f}ms")
print()
print("  Core time distribution:")
print(f"    Cube:   {aic_time_ms:.0f}ms ({aic_time_pct:.1f}%)")
print(f"    Vector: {aiv_time_ms:.0f}ms ({aiv_time_pct:.1f}%)")
print(f"    CPU:    {aicpu_dur_sum/1000:.0f}ms ({aicpu_dur_sum/total_op_time*100:.1f}%)")
print()

# Compare delta-related ops
print("-" * 70)
print("Delta-related ops (Sub/ReduceSum/Cast/Add/Reshape/Concat/ZerosLike/OnesLike):")
print("-" * 70)
for op_type, info in sorted(delta_ops.items()):
    print(f"  {op_type:15s}: {info['count']:>6d} ops, {info['dur_us']/1000:>8.2f}ms, Core={info['core_type']}")
print()

# Comparison table
print("-" * 70)
print("A1 vs A2_50 Comparison")
print("-" * 70)
print(f"  {'Metric':<30s} {'A1 (baseline)':>15s} {'A2_50 (inject)':>15s} {'Delta':>10s}")
print(f"  {'---':<30s} {'---':>15s} {'---':>15s} {'---':>10s}")
print(f"  {'AI_CORE count':<30s} {a1['aic_rows']:>15d} {aic_count:>15d} {aic_count - a1['aic_rows']:+10d}")
print(f"  {'AI_VECTOR count':<30s} {a1['aiv_rows']:>15d} {aiv_count:>15d} {aiv_count - a1['aiv_rows']:+10d}")
print(f"  {'Cube time (ms)':<30s} {a1['aic_time_ms']:>15.1f} {aic_time_ms:>15.1f} {aic_time_ms - a1['aic_time_ms']:+10.1f}")
print(f"  {'Vector time (ms)':<30s} {a1['aiv_time_ms']:>15.1f} {aiv_time_ms:>15.1f} {aiv_time_ms - a1['aiv_time_ms']:+10.1f}")
print(f"  {'Cube time %':<30s} {a1['aic_time_pct']:>15.1f} {aic_time_pct:>15.1f} {aic_time_pct - a1['aic_time_pct']:+10.2f}")
print(f"  {'Vector time %':<30s} {a1['aiv_time_pct']:>15.1f} {aiv_time_pct:>15.1f} {aiv_time_pct - a1['aiv_time_pct']:+10.2f}")
print(f"  {'Cube MAC util %':<30s} {a1['aic_mac_ratio_pct']:>15.2f} {aic_mac_pct:>15.2f} {aic_mac_pct - a1['aic_mac_ratio_pct']:+10.2f}")
print(f"  {'Cube scalar util %':<30s} {a1['aic_scalar_ratio_pct']:>15.2f} {aic_scal_pct:>15.2f} {aic_scal_pct - a1['aic_scalar_ratio_pct']:+10.2f}")
print(f"  {'Vector vec util %':<30s} {a1['aiv_vec_ratio_pct']:>15.2f} {aiv_vec_pct:>15.2f} {aiv_vec_pct - a1['aiv_vec_ratio_pct']:+10.2f}")
print(f"  {'Vector scalar util %':<30s} {a1['aiv_scalar_ratio_pct']:>15.2f} {aiv_scal_pct:>15.2f} {aiv_scal_pct - a1['aiv_scalar_ratio_pct']:+10.2f}")
print(f"  {'Vector eff util %':<30s} {a1['vec_eff_util_pct']:>15.2f} {vec_eff:>15.2f} {vec_eff - a1['vec_eff_util_pct']:+10.2f}")
print(f"  {'Vector idle %':<30s} {a1['vec_idle_pct']:>15.2f} {vec_idle_pct:>15.2f} {vec_idle_pct - a1['vec_idle_pct']:+10.2f}")
print(f"  {'Vector idle est (ms)':<30s} {a1['vec_idle_ms_est']:>15.1f} {vec_idle_ms:>15.1f} {vec_idle_ms - a1['vec_idle_ms_est']:+10.1f}")

# Save output
a2_result = {
    "test": "A2_50",
    "model": "GPT-2 XL 48L",
    "inject_params": 50,
    "aic_rows": aic_count,
    "aiv_rows": aiv_count,
    "aicpu_rows": aicpu_count,
    "aic_mac_ratio_pct": round(aic_mac_pct, 2),
    "aic_scalar_ratio_pct": round(aic_scal_pct, 2),
    "cube_eff_util_pct": round(cube_eff, 2),
    "aiv_vec_ratio_pct": round(aiv_vec_pct, 2),
    "aiv_scalar_ratio_pct": round(aiv_scal_pct, 2),
    "vec_eff_util_pct": round(vec_eff, 2),
    "vec_idle_pct": round(vec_idle_pct, 2),
    "vec_idle_ms_est": round(vec_idle_ms, 2),
    "aic_time_ms": round(aic_time_ms, 2),
    "aiv_time_ms": round(aiv_time_ms, 2),
    "aic_time_pct": round(aic_time_pct, 2),
    "aiv_time_pct": round(aiv_time_pct, 2),
    "aicpu_time_ms": round(aicpu_dur_sum/1000, 2),
    "total_op_time_ms": round(total_op_time/1000, 2),
    # Baseline values for comparison
    "a1_vec_eff_util_pct": a1["vec_eff_util_pct"],
    "a1_vec_idle_pct": a1["vec_idle_pct"],
    "a1_vec_idle_ms_est": a1["vec_idle_ms_est"],
    "a1_cube_eff_util_pct": a1["cube_eff_util_pct"],
    "a1_aic_time_ms": a1["aic_time_ms"],
    "a1_aiv_time_ms": a1["aiv_time_ms"],
    "vec_eff_delta_pp": round(vec_eff - a1["vec_eff_util_pct"], 2),
    "cube_eff_delta_pp": round(cube_eff - a1["cube_eff_util_pct"], 2),
    "vec_idle_delta_pct": round(vec_idle_pct - a1["vec_idle_pct"], 2),
    "aic_time_delta_ms": round(aic_time_ms - a1["aic_time_ms"], 2),
    "aiv_time_delta_ms": round(aiv_time_ms - a1["aiv_time_ms"], 2),
    # Delta ops summary
    "delta_ops": {k: dict(v) for k, v in delta_ops.items()},
}

with open(OUT, "w") as f:
    json.dump(a2_result, f, indent=2)
print(f"\nSaved to {OUT}")
