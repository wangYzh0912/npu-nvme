#!/usr/bin/env python3
"""Analyze exported msprof task timelines into Vector/Cube/HBM windows."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def number(value):
    try: return float(str(value).strip())
    except (TypeError,ValueError): return 0.0

def read_pmu(paths):
    rows=[]; has_vector_ratio=False; has_hbm_metrics=False
    ratio_fields=("aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio",
                  "aiv_vec_int32_ratio", "aiv_vec_misc_ratio")
    hbm_fields=("aiv_main_mem_read_bw(GB/s)",
                "aic_main_mem_read_bw(GB/s)",
                "aiv_main_mem_write_bw(GB/s)",
                "aic_main_mem_write_bw(GB/s)")
    for path in paths:
        with Path(path).open(newline="") as stream:
            reader=csv.DictReader(stream)
            has_vector_ratio = has_vector_ratio or any(k in (reader.fieldnames or []) for k in ratio_fields)
            has_hbm_metrics = has_hbm_metrics or any(
                k in (reader.fieldnames or []) for k in hbm_fields)
            for row in reader:
                start=number(row.get("Task Start Time(us)")); duration=number(row.get("Task Duration(us)"))
                if not start or duration<=0: continue
                rows.append({"start":start,"stop":start+duration,
                    "vector":sum(number(row.get(k)) for k in ("aiv_vec_fp32_ratio","aiv_vec_fp16_ratio","aiv_vec_int32_ratio","aiv_vec_misc_ratio")),
                    "cube":sum(number(row.get(k)) for k in ("aic_mac_fp16_ratio","aic_mac_int8_ratio")),
                    "hbm_read_gb_s":number(row.get("aiv_main_mem_read_bw(GB/s)"))+number(row.get("aic_main_mem_read_bw(GB/s)")),
                    "hbm_write_gb_s":number(row.get("aiv_main_mem_write_bw(GB/s)"))+number(row.get("aic_main_mem_write_bw(GB/s)"))})
    return rows, has_vector_ratio, has_hbm_metrics

def read_hbm_average(paths):
    reads=[]; writes=[]
    for path in paths:
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("Metric", "")).strip().lower() != "average":
                    continue
                reads.append(number(row.get("Read(MB/s)")) / 1000.0)
                writes.append(number(row.get("Write(MB/s)")) / 1000.0)
    return ((float(np.mean(reads)) if reads else None),
            (float(np.mean(writes)) if writes else None))

def read_tasks(paths):
    rows=[]
    for path in paths:
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    start=float(row.get("task_start(us)", "0").strip())
                    stop=float(row.get("task_stop(us)", "0").strip())
                except (ValueError, AttributeError):
                    continue
                if stop <= start: continue
                kind=str(row.get("kernel_type", "")).upper()
                core="vector" if "AIVEC" in kind or "VECTOR" in kind else (
                    "cube" if "AIC" in kind or "CUBE" in kind else "other")
                rows.append((start, stop, core, row.get("kernel_name", "")))
    return sorted(rows)

def windows(rows, width_us, pmu=(), vector_ratio_available=True,
            hbm_metrics_available=False):
    if not rows: return []
    begin=min(x[0] for x in rows); end=max(x[1] for x in rows)
    count=int(np.ceil((end-begin)/width_us)); keys=("vector","cube","other")
    active={key:np.zeros(count,dtype=np.float64) for key in keys}
    weighted={key:np.zeros(count,dtype=np.float64) for key in
              ("vector","cube","hbm_read_gb_s","hbm_write_gb_s")}
    def add_interval(start,stop,target,value=1.0):
        first=max(0,int((start-begin)//width_us)); last=min(count-1,int((stop-begin)//width_us))
        for index in range(first,last+1):
            left=begin+index*width_us; right=min(end,left+width_us)
            target[index]+=max(0.0,min(right,stop)-max(left,start))*value
    for start,stop,core,_ in rows: add_interval(start,stop,active[core])
    for item in pmu:
        for key in weighted: add_interval(item["start"],item["stop"],weighted[key],item[key])
    output=[]
    for index in range(count):
        cursor=begin+index*width_us; finish=min(end,cursor+width_us); duration=finish-cursor
        output.append({"start_us":cursor,"end_us":finish,
            "vector_active_fraction":min(1.0,active["vector"][index]/duration),
            "cube_active_fraction":min(1.0,active["cube"][index]/duration),
            "vector_util":min(1.0,weighted["vector"][index]/duration) if pmu and vector_ratio_available else None,
            "cube_util":min(1.0,weighted["cube"][index]/duration) if pmu and vector_ratio_available else None,
            "hbm_read_gb_s":weighted["hbm_read_gb_s"][index]/duration if pmu and hbm_metrics_available else None,
            "hbm_write_gb_s":weighted["hbm_write_gb_s"][index]/duration if pmu and hbm_metrics_available else None,
            "other_util":min(1.0,active["other"][index]/duration)})
    return output

def main():
    p=argparse.ArgumentParser(); p.add_argument("--task-time", nargs="+", type=Path, required=True); p.add_argument("--op-summary",nargs="*",type=Path,default=[]); p.add_argument("--steps-json",type=Path,default=None)
    p.add_argument("--hbm", nargs="*", type=Path, default=[]); p.add_argument("--window-us", type=float, default=10000)
    p.add_argument("--output", type=Path, required=True); args=p.parse_args()
    rows=read_tasks(args.task_time)
    pmu, vector_ratio_available, hbm_metrics_available=read_pmu(args.op_summary)
    hbm_device_read, hbm_device_write=read_hbm_average(args.hbm)
    bins=windows(rows,args.window_us,pmu,vector_ratio_available,
                 hbm_metrics_available)
    alignment="unavailable"
    if args.steps_json and args.steps_json.exists() and bins:
        steps=json.loads(args.steps_json.read_text()).get("samples",[]); cursor=bins[0]["start_us"]; ranges=[]
        for step in steps:
            finish=cursor+float(step["step_ms"])*1000.0; ranges.append((cursor,finish,int(step["step"]))); cursor=finish
        for window in bins:
            middle=(window["start_us"]+window["end_us"])/2; window["step"]=next((step for start,stop,step in ranges if start<=middle<stop),None)
        alignment="duration-normalized to child formal step durations; device and host clocks have no exported common epoch"
    vals=[b["vector_util"] for b in bins if b["vector_util"] is not None]; cube=[b["cube_util"] for b in bins if b["cube_util"] is not None]
    summary={"task_files":[str(x) for x in args.task_time],"windows":len(bins),
             "vector_mean":float(np.mean(vals)) if vals else None,
             "vector_p95":float(np.percentile(vals,95)) if vals else None,
             "below_20_fraction":float(np.mean(np.asarray(vals)<.2)) if vals else None,
             "below_30_fraction":float(np.mean(np.asarray(vals)<.3)) if vals else None,
             "below_50_fraction":float(np.mean(np.asarray(vals)<.5)) if vals else None,
             "cube_busy_vector_low_fraction":float(np.mean([(c>.5 and v<.3) for c,v in zip(cube,vals)])) if vals else None,
             "hbm_read_gb_s_mean":float(np.mean([b["hbm_read_gb_s"] for b in bins])) if pmu and hbm_metrics_available else None,
             "hbm_write_gb_s_mean":float(np.mean([b["hbm_write_gb_s"] for b in bins])) if pmu and hbm_metrics_available else None,
             "hbm_device_average_read_gb_s":hbm_device_read,
             "hbm_device_average_write_gb_s":hbm_device_write,
             "hbm_metric_interpretation":"device averages come from msprof hbm.csv Average rows; legacy hbm_*_mean fields are duration-weighted per-op projections, and neither is an HBM utilization percentage",
             "op_summary_files":[str(x) for x in args.op_summary],"hbm_files":[str(x) for x in args.hbm],
             "step_alignment":alignment,
             "step_alignment_valid":False,
             "metric_interpretation":"legacy vector_util fields are duration-weighted ArithmeticUtilization issue ratios projected onto device wall-clock; they are not whole-device Vector occupancy",
             "utilization_source": ("ArithmeticUtilization PMU issue ratios; not whole-device occupancy"
                                    if vector_ratio_available else
                                    ("PMU export has no arithmetic ratios; Vector utilization unavailable"
                                     if pmu else "unavailable; activity fractions are not relabelled as utilization"))}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({"summary":summary,"windows":bins},indent=2)+"\n")
    print(json.dumps(summary,sort_keys=True))
if __name__ == "__main__": main()
