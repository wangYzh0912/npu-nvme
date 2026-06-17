#!/usr/bin/env python3
"""Parse E4_BASE and E4_I3 op_summary CSVs, produce comparison report."""
import csv, os, json

def parse_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        lines = list(f)
    reader = csv.reader(lines)
    header = next(reader)
    col = {h.strip(): i for i, h in enumerate(header)}
    st = {"total_ops": 0, "total_us": 0.0, "aic_us": 0.0, "aiv_us": 0.0, "aicpu_us": 0.0,
          "aic_count": 0, "aiv_count": 0, "aicpu_count": 0,
          "aic_mac_w": 0.0, "aiv_vec_w": 0.0,
          "delta_ops": {}}
    delta_set = {"Sub","Mul","ReduceSum","Cast","Reshape","Concat","ZerosLike","OnesLike","Add"}
    for row in reader:
        if len(row) < max(col.values())+1: continue
        try:
            task = row[col['Task Type']].strip()
            op_type = row[col['OP Type']].strip()
            dur = float(row[col['Task Duration(us)']].strip())
        except: continue
        st["total_ops"] += 1; st["total_us"] += dur
        if 'AI_CORE' in task and 'AI_VECTOR' not in task:
            st["aic_count"] += 1; st["aic_us"] += dur
            try: st["aic_mac_w"] += float(row[col.get('aic_mac_ratio',24)].strip()) * dur
            except: pass
        elif 'AI_VECTOR' in task:
            st["aiv_count"] += 1; st["aiv_us"] += dur
            try: st["aiv_vec_w"] += float(row[col.get('aiv_vec_ratio',37)].strip()) * dur
            except: pass
        elif 'AICPU' in task:
            st["aicpu_count"] += 1; st["aicpu_us"] += dur
        if op_type in delta_set:
            d = st["delta_ops"].setdefault(op_type, {"count":0,"dur_us":0.0,"core":task})
            d["count"] += 1; d["dur_us"] += dur
    if st["aic_us"] > 0: st["aic_mac_pct"] = st["aic_mac_w"] / st["aic_us"]
    if st["aiv_us"] > 0: st["aiv_vec_pct"] = st["aiv_vec_w"] / st["aiv_us"]
    return st

BASE = "/home/user7/npu-nvme/output/profiling_vec/E4_BASE/PROF_000001_20260616175009350_CQIDICIIFLPNHOLC/mindstudio_profiler_output/op_summary_20260616175618.csv"
I3   = "/home/user7/npu-nvme/output/profiling_vec/E4_I3/PROF_000001_20260616175852479_BMMDQDACKAPEKCPA/mindstudio_profiler_output/op_summary_20260616180500.csv"

base = parse_csv(BASE)
i3 = parse_csv(I3)

print("="*75)
print("E4 FINAL: GPT-2 XL 24L — PMU Core Utilization Comparison")
print("="*75)
print(f"{'Metric':<35} {'BASELINE':>15} {'I3_BATCHED':>15} {'Delta'}")
print(f"{'-'*75}")

for label, bv, iv, fmt in [
    ("Total ops", base["total_ops"], i3["total_ops"], "d"),
    ("Total op time (s)", base["total_us"]/1e6, i3["total_us"]/1e6, ".3f"),
    ("Cube ops", base["aic_count"], i3["aic_count"], "d"),
    ("Cube time (s)", base["aic_us"]/1e6, i3["aic_us"]/1e6, ".3f"),
    ("Cube MAC util avg (%)", base.get("aic_mac_pct",0), i3.get("aic_mac_pct",0), ".1f"),
    ("Vector ops", base["aiv_count"], i3["aiv_count"], "d"),
    ("Vector time (s)", base["aiv_us"]/1e6, i3["aiv_us"]/1e6, ".3f"),
    ("Vector ALU util avg (%)", base.get("aiv_vec_pct",0), i3.get("aiv_vec_pct",0), ".1f"),
    ("AICPU ops", base["aicpu_count"], i3["aicpu_count"], "d"),
    ("AICPU time (s)", base["aicpu_us"]/1e6, i3["aicpu_us"]/1e6, ".3f"),
]:
    fmt_str = f"{{v:{fmt}}}" if isinstance(fmt, str) else str
    delta_str = ""
    if isinstance(bv, (int, float)) and isinstance(iv, (int, float)):
        if isinstance(bv, float) and bv > 0:
            delta_str = f"{(iv-bv)/bv*100:+.1f}%"
        elif isinstance(bv, int) and bv > 0:
            delta_str = f"{iv-bv:+d}"
    print(f"{label:<35} {bv:{fmt}} {iv:{fmt}} {delta_str}")

print(f"\n{'='*75}")
print("Delta-related Ops Core Attribution")
print(f"{'='*75}")
print(f"{'Op Type':<15} {'BASE_count':>12} {'BASE_core':>18} {'I3_count':>12} {'I3_core':>18}")
for ot in sorted(set(list(base["delta_ops"].keys()) + list(i3["delta_ops"].keys()))):
    bd = base["delta_ops"].get(ot, {"count":0,"core":"-"})
    id_ = i3["delta_ops"].get(ot, {"count":0,"core":"-"})
    delta_str = ""
    if bd["count"] > 0:
        delta_str = f"{(id_['count']-bd['count'])/bd['count']*100:+.0f}%"
    print(f"{ot:<15} {bd['count']:>12d} {bd['core']:>18} {id_['count']:>12d} {id_['core']:>18} ({delta_str})")

# Save
out = os.path.join(os.path.dirname(__file__), "..", "output", "phase5_e4_pmu_comparison.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"experiment":"Phase 5 E4 PMU Comparison","model":"GPT-2 XL 48L, 24L delta",
               "baseline":{k:str(v) if isinstance(v,float) else v for k,v in base.items()},
               "i3_batched":{k:str(v) if isinstance(v,float) else v for k,v in i3.items()}}, f, indent=2)
print(f"\n-> {out}")
