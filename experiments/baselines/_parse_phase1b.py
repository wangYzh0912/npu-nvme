#!/usr/bin/env python3
"""
Phase 1b PMU Parser — parses msprof op_summary CSV and extracts:
  - AI_CORE / AI_VECTOR_CORE kernel counts and times
  - MAC/Vector/Scalar utilization ratios
  - Vector idle time estimate
  - Delta ops core-type verification (for inject runs)

Usage: python _parse_phase1b.py <label> [--baseline-pmu baseline.json]
  Input:  /home/user7/npu-nvme/output/profiling_vec/<label>/.../op_summary*.csv
  Output: /home/user7/npu-nvme/experiments/output/phase1b_<label>_pmu.json
"""
import csv, json, os, sys, glob, argparse
from collections import defaultdict

REPO = "/home/user7/npu-nvme"
PROF_BASE = os.path.join(REPO, "output", "profiling_vec")
OUT_DIR = os.path.join(REPO, "experiments", "output")


def find_csv(label):
    """Find the op_summary CSV in msprof output directory."""
    d = os.path.join(PROF_BASE, label)
    if not os.path.isdir(d):
        # try glob
        matches = glob.glob(os.path.join(PROF_BASE, f"*{label}*"))
        for m in matches:
            if os.path.isdir(m):
                d = m
                break
    pattern = os.path.join(d, "**", "op_summary*.csv")
    csvs = sorted(glob.glob(pattern, recursive=True))
    if not csvs:
        # Also try device/** pattern
        pattern2 = os.path.join(d, "device", "*", "op_summary*.csv")
        csvs = sorted(glob.glob(pattern2, recursive=False))
    if not csvs:
        print(f"  WARNING: No op_summary CSV found under {d}")
        print(f"  Searched: {pattern}")
        # List what we did find
        all_csvs = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        print(f"  Available CSVs ({len(all_csvs)}):")
        for c in all_csvs[:10]:
            print(f"    {os.path.relpath(c, PROF_BASE)}")
    return csvs


def parse_csv(label):
    """Parse all CSV files from a profiling run."""
    csv_files = find_csv(label)
    if not csv_files:
        return None

    aic_count = 0; aiv_count = 0; aicpu_count = 0
    aic_dur_sum = 0.0; aiv_dur_sum = 0.0; aicpu_dur_sum = 0.0
    aic_mac_sum = 0.0; aic_scalar_sum = 0.0
    aiv_vec_sum = 0.0; aiv_scalar_sum = 0.0

    delta_ops = defaultdict(lambda: {"count": 0, "dur_us": 0.0, "core_type": None})
    all_op_types = defaultdict(lambda: {"count": 0, "dur_us": 0.0, "core_type": None})

    # Known delta-related op types
    DELTA_OPS = {"Sub", "ReduceSum", "Cast", "Add", "Reshape", "Concat", "ZerosLike", "OnesLike"}

    for fp in csv_files:
        print(f"  Parsing: {os.path.relpath(fp, PROF_BASE)}", flush=True)
        with open(fp) as f:
            reader = csv.reader(f)
            header = next(reader)
            # Detect column indices from header
            # Expected: ... Task Type, Kernel Name/Op Type, Duration(us)
            # Plus: aic_mac_ratio, aic_scalar_ratio, aiv_vec_ratio, aiv_scalar_ratio
            col_map = {}
            for i, h in enumerate(header):
                h_clean = h.strip().lower()
                col_map[h_clean] = i

            # Known column names we care about
            task_type_col = col_map.get("task type", col_map.get("core type", 7))
            op_type_col   = col_map.get("op type", col_map.get("kernel name", 5))
            dur_col       = col_map.get("task duration(us)", col_map.get("duration(us)", 9))

            # PMU ratio columns
            mac_ratio_col = col_map.get("aic_mac_fp16_ratio", col_map.get("aic_mac_ratio", 24))
            scalar_ratio_col = col_map.get("aic_scalar_ratio", 26)
            vec_ratio_col = col_map.get("aiv_vec_fp16_ratio", col_map.get("aiv_vec_ratio", 37))
            aiv_scalar_ratio_col = col_map.get("aiv_scalar_ratio", 39)

            for row in reader:
                try:
                    core = row[task_type_col].strip()
                    op_type = row[op_type_col].strip()
                    dur_us = float(row[dur_col].strip())
                except (IndexError, ValueError):
                    continue

                # Aggregate per core type
                if core == "AI_CORE":
                    aic_count += 1; aic_dur_sum += dur_us
                    try:
                        aic_mac_sum += float(row[mac_ratio_col].strip())
                        aic_scalar_sum += float(row[scalar_ratio_col].strip())
                    except (ValueError, IndexError):
                        pass
                elif core == "AI_VECTOR_CORE":
                    aiv_count += 1; aiv_dur_sum += dur_us
                    try:
                        aiv_vec_sum += float(row[vec_ratio_col].strip())
                        aiv_scalar_sum += float(row[aiv_scalar_ratio_col].strip())
                    except (ValueError, IndexError):
                        pass
                elif core == "AI_CPU":
                    aicpu_count += 1; aicpu_dur_sum += dur_us

                # Track op types
                all_op_types[op_type]["count"] += 1
                all_op_types[op_type]["dur_us"] += dur_us
                all_op_types[op_type]["core_type"] = core

                if op_type in DELTA_OPS:
                    delta_ops[op_type]["count"] += 1
                    delta_ops[op_type]["dur_us"] += dur_us
                    delta_ops[op_type]["core_type"] = core

    total_op_time = aic_dur_sum + aiv_dur_sum + aicpu_dur_sum
    if total_op_time == 0:
        return {"error": "no valid data", "csv_files": len(csv_files)}

    # Compute utilizations
    aic_mac_pct  = (aic_mac_sum / max(aic_count, 1)) * 100
    aic_scal_pct = (aic_scalar_sum / max(aic_count, 1)) * 100
    aiv_vec_pct  = (aiv_vec_sum / max(aiv_count, 1)) * 100
    aiv_scal_pct = (aiv_scalar_sum / max(aiv_count, 1)) * 100

    cube_eff = aic_mac_pct
    vec_eff  = aiv_vec_pct + aiv_scal_pct
    vec_idle_pct = max(0, 100 - vec_eff)
    vec_idle_ms = aiv_dur_sum / 1000 * (vec_idle_pct / 100)

    aic_time_ms = aic_dur_sum / 1000
    aiv_time_ms = aiv_dur_sum / 1000
    aic_time_pct = aic_dur_sum / total_op_time * 100
    aiv_time_pct = aiv_dur_sum / total_op_time * 100

    # Top ops by core type (for quick verification)
    aic_top_ops = sorted(
        [(ot, info) for ot, info in all_op_types.items() if info["core_type"] == "AI_CORE"],
        key=lambda x: x[1]["dur_us"], reverse=True)[:10]
    aiv_top_ops = sorted(
        [(ot, info) for ot, info in all_op_types.items() if info["core_type"] == "AI_VECTOR_CORE"],
        key=lambda x: x[1]["dur_us"], reverse=True)[:10]

    result = {
        "label": label,
        "aic_kernels": aic_count,
        "aiv_kernels": aiv_count,
        "aicpu_kernels": aicpu_count,
        "aic_time_ms": round(aic_time_ms, 2),
        "aiv_time_ms": round(aiv_time_ms, 2),
        "aicpu_time_ms": round(aicpu_dur_sum / 1000, 2),
        "aic_time_pct": round(aic_time_pct, 2),
        "aiv_time_pct": round(aiv_time_pct, 2),
        "aic_mac_util_pct": round(aic_mac_pct, 2),
        "aic_scalar_util_pct": round(aic_scal_pct, 2),
        "cube_eff_util_pct": round(cube_eff, 2),
        "aiv_vec_util_pct": round(aiv_vec_pct, 2),
        "aiv_scalar_util_pct": round(aiv_scal_pct, 2),
        "vec_eff_util_pct": round(vec_eff, 2),
        "vec_idle_pct": round(vec_idle_pct, 2),
        "vec_idle_ms_est": round(vec_idle_ms, 2),
        "total_op_time_ms": round(total_op_time / 1000, 2),
        "aic_top_ops": [{"op": ot, "count": info["count"], "dur_ms": round(info["dur_us"]/1000, 2)}
                        for ot, info in aic_top_ops],
        "aiv_top_ops": [{"op": ot, "count": info["count"], "dur_ms": round(info["dur_us"]/1000, 2)}
                        for ot, info in aiv_top_ops],
        "delta_ops": {k: {"count": v["count"], "dur_ms": round(v["dur_us"]/1000, 2), "core": v["core_type"]}
                      for k, v in delta_ops.items()},
        "csv_files": len(csv_files),
    }

    print(f"\n  [{label}] PMU Summary:")
    print(f"    AI_CORE:       {aic_count:>8d} kernels, {aic_time_ms:>8.1f}ms ({aic_time_pct:.1f}%)")
    print(f"    AI_VECTOR_CORE:{aiv_count:>8d} kernels, {aiv_time_ms:>8.1f}ms ({aiv_time_pct:.1f}%)")
    print(f"    Cube eff:   {cube_eff:.2f}% (MAC)")
    print(f"    Vector eff: {vec_eff:.2f}% (vec+scalar)")
    print(f"    Vector idle: {vec_idle_pct:.2f}% ≈ {vec_idle_ms:.0f}ms")

    return result


def compare(baseline_pmu, inject_pmu):
    """Print comparison table between baseline and inject run."""
    if not baseline_pmu or not inject_pmu:
        return

    b, i = baseline_pmu, inject_pmu
    print(f"\n{'='*80}")
    print(f"  {b.get('label','baseline')} vs {i.get('label','inject')} Comparison")
    print(f"{'='*80}")

    rows = [
        ("AI_CORE kernels",       b.get("aic_kernels",0),       i.get("aic_kernels",0)),
        ("AI_VECTOR kernels",     b.get("aiv_kernels",0),       i.get("aiv_kernels",0)),
        ("Cube time (ms)",        b.get("aic_time_ms",0),       i.get("aic_time_ms",0)),
        ("Vector time (ms)",      b.get("aiv_time_ms",0),       i.get("aiv_time_ms",0)),
        ("Cube MAC util %",       b.get("aic_mac_util_pct",0),  i.get("aic_mac_util_pct",0)),
        ("Vector vec util %",     b.get("aiv_vec_util_pct",0),  i.get("aiv_vec_util_pct",0)),
        ("Vector scalar util %",  b.get("aiv_scalar_util_pct",0), i.get("aiv_scalar_util_pct",0)),
        ("Vector eff util %",     b.get("vec_eff_util_pct",0),  i.get("vec_eff_util_pct",0)),
        ("Vector idle %",         b.get("vec_idle_pct",0),      i.get("vec_idle_pct",0)),
        ("Vector idle ms",        b.get("vec_idle_ms_est",0),   i.get("vec_idle_ms_est",0)),
    ]
    print(f"  {'Metric':<28s} {'Baseline':>14s} {'Inject':>14s} {'Delta':>12s}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*12}")
    for name, bv, iv in rows:
        d = iv - bv
        unit = "" if "%" in name or "idle" in name.lower() else ""
        print(f"  {name:<28s} {bv:>14.2f} {iv:>14.2f} {d:>+12.2f}")

    # Key verdicts
    cube_delta = abs(i.get("aic_mac_util_pct",0) - b.get("aic_mac_util_pct",0))
    print(f"\n  Cube util change: {cube_delta:+.2f}pp {'✅ unchanged' if cube_delta < 2 else '⚠️ changed'}")

    # Delta ops core-type verification
    dops = i.get("delta_ops", {})
    if dops:
        print(f"\n  Delta ops Core Type:")
        for op_name, info in sorted(dops.items()):
            print(f"    {op_name:15s}: {info['count']:>6d} ops, {info['dur_ms']:>8.2f}ms, Core={info['core']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Experiment label (e.g. V5_baseline)")
    parser.add_argument("--baseline-pmu", default=None, help="Path to baseline PMU JSON for comparison")
    parser.add_argument("--compare", default=None, help="Label of inject run to compare against baseline")
    parser.add_argument("--step-ms", type=float, default=None, help="Average step time in ms (from training JSON)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse the primary label
    pmu = parse_csv(args.label)

    # If loading baseline for comparison
    baseline_pmu = None
    if args.baseline_pmu:
        try:
            with open(args.baseline_pmu) as f:
                baseline_pmu = json.load(f)
        except FileNotFoundError:
            print(f"Warning: baseline PMU not found at {args.baseline_pmu}")

    # If comparing
    if args.compare:
        inject_pmu = pmu
        baseline_pmu = parse_csv(args.compare)
        compare(baseline_pmu, inject_pmu)

    # Add step time if provided
    if args.step_ms is not None and pmu:
        pmu["avg_step_ms"] = args.step_ms

    # Save output
    out_path = os.path.join(OUT_DIR, f"phase1b_{args.label}_pmu.json")
    with open(out_path, "w") as f:
        json.dump(pmu, f, indent=2)
    print(f"\n[OK] PMU data → {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
