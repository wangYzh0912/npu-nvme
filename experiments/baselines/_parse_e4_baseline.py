#!/usr/bin/env python3
"""Parse op_summary CSV — standalone, no mindspore import needed."""
import csv

csv_path = "/home/user7/npu-nvme/output/profiling_vec/E4_BASE/PROF_000001_20260616175009350_CQIDICIIFLPNHOLC/mindstudio_profiler_output/op_summary_20260616175618.csv"

with open(csv_path, encoding='utf-8-sig') as f:
    lines = list(f)

reader = csv.reader(lines)
header = next(reader)
col = {h.strip(): i for i, h in enumerate(header)}

total_ops = 0; total_us = 0.0
aic_us = 0.0; aiv_us = 0.0; aicpu_us = 0.0
aic_count = 0; aiv_count = 0; aicpu_count = 0
aic_mac_weighted = 0.0; aiv_vec_weighted = 0.0

for row in reader:
    if len(row) < max(col.values()) + 1:
        continue
    try:
        task = row[col['Task Type']].strip()
        dur = float(row[col['Task Duration(us)']].strip())
    except:
        continue
    total_ops += 1
    total_us += dur
    if 'AI_CORE' in task and 'AI_VECTOR' not in task:
        aic_count += 1; aic_us += dur
        try: aic_mac_weighted += float(row[col.get('aic_mac_ratio',24)].strip()) * dur
        except: pass
    elif 'AI_VECTOR' in task:
        aiv_count += 1; aiv_us += dur
        try: aiv_vec_weighted += float(row[col.get('aiv_vec_ratio',37)].strip()) * dur
        except: pass
    elif 'AICPU' in task:
        aicpu_count += 1; aicpu_us += dur

print(f"Total: {total_ops} ops, {total_us/1e6:.3f}s")
if aic_us > 0:
    print(f"Cube:  {aic_count:5d} ops, {aic_us/1e6:.3f}s ({aic_us/total_us*100:.1f}%), mac_util_avg={aic_mac_weighted/aic_us:.1f}%")
if aiv_us > 0:
    print(f"Vector:{aiv_count:5d} ops, {aiv_us/1e6:.3f}s ({aiv_us/total_us*100:.1f}%), vec_util_avg={aiv_vec_weighted/aiv_us:.1f}%")
    idle_pct = 1 - aiv_vec_weighted / aiv_us / 100.0
    print(f"VectorIdleMs={aiv_us/1000*idle_pct:.0f}ms, VectorALUutil={aiv_vec_weighted/aiv_us:.1f}%, VectorIDLEpct={idle_pct*100:.1f}%")
if aicpu_us > 0:
    print(f"AICPU: {aicpu_count:5d} ops, {aicpu_us/1e6:.3f}s ({aicpu_us/total_us*100:.1f}%)")
