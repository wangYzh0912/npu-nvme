#!/usr/bin/env python3
"""Parse A1 op_summary CSV properly — column index based since field count varies."""
import csv, json, os
from collections import defaultdict

SUM_FILE = "/home/user7/npu-nvme/output/profiling_vec/A1/PROF_000001_20260609232800974_DDHQNKRHNCDGGEPA/mindstudio_profiler_output/op_summary_20260610025708.csv"

# Column positions from header inspection:
#  0: Device_id
#  ...
#  7: Core Type
#  8: Task Start Time(us)
#  9: Task Duration(us)
# 10: Task Wait Time(us)
# ...
# 21: aicore_time(us)
# 22: aic_total_cycles
# 23: aic_mac_time(us)
# 24: aic_mac_ratio
# 25: aic_scalar_time(us)
# 26: aic_scalar_ratio
# ...
# 31: aiv_time(us)
# 32: aiv_total_cycles
# 33: aiv_vec_time(us)
# 34: aiv_vec_ratio
# 35: aiv_scalar_time(us)
# 36: aiv_scalar_ratio
# ...
# 41: aiv_icache_miss_rate
# 42: cube_utilization(%)

aic_count = 0; aiv_count = 0
aic_mac_sum = 0.0; aic_scalar_sum = 0.0
aiv_vec_sum = 0.0; aiv_scalar_sum = 0.0

# Duration stats per core
aic_dur_sum = 0.0; aiv_dur_sum = 0.0

with open(SUM_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("Device_id"):
            continue
        # The CSV has tab separators between columns
        # Simple split on tab
        parts = line.split("\t")
        if len(parts) < 42:
            continue
        try:
            core = parts[7].strip().rstrip(",")
            dur_us = float(parts[9].strip().rstrip(","))
        except (IndexError, ValueError):
            continue

        if core == "AI_CORE":
            aic_count += 1; aic_dur_sum += dur_us
            try:
                aic_mac_sum += float(parts[24].strip().rstrip(","))
                aic_scalar_sum += float(parts[26].strip().rstrip(","))
            except (ValueError, IndexError):
                pass
        elif core == "AI_VECTOR_CORE":
            aiv_count += 1; aiv_dur_sum += dur_us
            try:
                aiv_vec_sum += float(parts[34].strip().rstrip(","))
                aiv_scalar_sum += float(parts[36].strip().rstrip(","))
            except (ValueError, IndexError):
                pass

aic_mac_pct  = (aic_mac_sum  / max(aic_count, 1)) * 100
aic_scal_pct = (aic_scalar_sum / max(aic_count, 1)) * 100
aiv_vec_pct  = (aiv_vec_sum  / max(aiv_count, 1)) * 100
aiv_scal_pct = (aiv_scalar_sum / max(aiv_count, 1)) * 100

cube_eff = aic_mac_pct
vec_eff  = aiv_vec_pct + aiv_scal_pct
vec_idle_ms = aiv_dur_sum / 1000 * (1 - vec_eff / 100)

print("A1 PMU from op_summary ({} AI_CORE rows, {} AI_VECTOR_CORE rows)".format(aic_count, aiv_count))
print()
print("  AI_CORE:")
print("    Total duration:  {:.2f}s".format(aic_dur_sum / 1e6))
print("    mac_ratio avg:   {:.2f}%".format(aic_mac_pct))
print("    scalar_ratio avg: {:.2f}%".format(aic_scal_pct))
print("    Cube effective:  {:.2f}%".format(cube_eff))
print()
print("  AI_VECTOR_CORE:")
print("    Total duration:  {:.2f}s".format(aiv_dur_sum / 1e6))
print("    vec_ratio avg:   {:.2f}%".format(aiv_vec_pct))
print("    scalar_ratio avg: {:.2f}%".format(aiv_scal_pct))
print("    Vector effective: {:.2f}%".format(vec_eff))
print("    Vector idle:      {:.2f}%".format(100 - vec_eff))
print("    Vector idle est:  {:.0f}ms".format(vec_idle_ms))

# Save
out = {
    "test": "A1", "model": "GPT-2 XL",
    "aic_row_count": aic_count, "aiv_row_count": aiv_count,
    "aic_dur_us": round(aic_dur_sum, 2), "aiv_dur_us": round(aiv_dur_sum, 2),
    "aic_mac_ratio_pct": round(aic_mac_pct, 2),
    "aic_scalar_ratio_pct": round(aic_scal_pct, 2),
    "cube_eff_util_pct": round(cube_eff, 2),
    "aiv_vec_ratio_pct": round(aiv_vec_pct, 2),
    "aiv_scalar_ratio_pct": round(aiv_scal_pct, 2),
    "vec_eff_util_pct": round(vec_eff, 2),
    "vec_idle_pct": round(100 - vec_eff, 2),
    "vec_idle_ms_est": round(vec_idle_ms, 2),
}
with open("/home/user7/npu-nvme/experiments/output/phase1a_a1_pmu.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to phase1a_a1_pmu.json")
