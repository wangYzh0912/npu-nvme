#!/usr/bin/env python3
"""
Step 1b: Device-level PMU Profiling — GPT-2 XL Pure Training
==============================================================
Measures device-level Cube MAC and Vector ALU utilization via
msprof sample-based PMU counters (ArithmeticUtilization metrics).

Background:
  Step 1a (task-based msprof) gives per-operator internal efficiency
  (aic_mac_ratio / aiv_vec_ratio). These are NOT device-level utilization.

  Device-level utilization requires msprof's sample-based AI Core profiling
  with the ArithmeticUtilization metric group, which samples PMU counters
  at configurable frequency (default 100 Hz).

Metrics:
  S1b.1: Cube MAC utilization (device-level, %)
  S1b.2: Vector ALU utilization (device-level, %)
  S1b.3: Vector idle ratio (= 100% - Vector ALU util)
  S1b.4: Core time distribution (AIC % vs AIV % vs AICPU %)
  S1b.5: Per-step time (sanity check vs Step 1a baseline)

Usage:
  bash _run.sh 12           # 12-step benchmark

Output:
  experiments/output/benchmark/step1b_pmu.json
"""

import os, sys, time, json, argparse, csv, glob as gb, re, math

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

OUTPUT_DIR = os.path.join(REPO, "experiments", "output", "benchmark")
DEVICE_ID = 1
SEQ_LEN = 1024


# ═══════════════════════════════════════════════════════════════════
# PMU CSV Parser (sample-based ArithmeticUtilization)
# ═══════════════════════════════════════════════════════════════════

def parse_sample_pmu_csvs(prof_dir):
    """
    Parse msprof sample-based PMU output.

    In sample-based mode, msprof produces additional CSV files with
    device-level PMU counter samples. The exact file names depend on
    the CANN version. We search for:
      - device_*/pmu_*.csv  (sample-based counter data)
      - op_summary_*.csv    (fallback: task-based, for core distribution)
    """
    result = {
        "cube_mac_util_pct": None,
        "vector_alu_util_pct": None,
        "vector_idle_pct": None,
        "aic_time_pct": None,
        "aiv_time_pct": None,
        "aicpu_time_pct": None,
        "total_kernel_instances": None,
        "aic_ops": None,
        "aiv_ops": None,
        "source": None,
    }

    # ── Strategy 1: Parse msprof sample-based utilization CSVs ──
    # CANN 8.0 RC3 produces:
    #   mindstudio_profiler_output/ai_core_utilization_*.csv
    #   mindstudio_profiler_output/ai_vector_core_utilization_*.csv
    # Each has per-core ratios and an "Average" row at the bottom.
    aic_csvs = []
    aiv_csvs = []
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if 'ai_core_utilization' in f and f.endswith('.csv'):
                aic_csvs.append(os.path.join(root, f))
            elif 'ai_vector_core_utilization' in f and f.endswith('.csv'):
                aiv_csvs.append(os.path.join(root, f))

    # Parse AIC utilization
    for fp in aic_csvs:
        try:
            with open(fp, encoding='utf-8-sig') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("Core ID", "").strip() == "Average":
                        mac_fp16 = float(row.get("mac_fp16_ratio", 0))
                        vec_fp16 = float(row.get("vec_fp16_ratio", 0))
                        vec_fp32 = float(row.get("vec_fp32_ratio", 0))
                        vec_misc = float(row.get("vec_misc_ratio", 0))
                        result["cube_mac_util_pct"] = round(mac_fp16 * 100, 2)
                        result["aic_vec_total_pct"] = round((vec_fp16 + vec_fp32 + vec_misc) * 100, 2)
                        break
        except Exception as e:
            print(f"  WARNING: Failed to parse {fp}: {e}")

    # Parse AIV utilization
    for fp in aiv_csvs:
        try:
            with open(fp, encoding='utf-8-sig') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("Core ID", "").strip() == "Average":
                        vec_fp16 = float(row.get("vec_fp16_ratio", 0))
                        vec_fp32 = float(row.get("vec_fp32_ratio", 0))
                        vec_misc = float(row.get("vec_misc_ratio", 0))
                        total_vec = vec_fp16 + vec_fp32 + vec_misc
                        result["vector_alu_util_pct"] = round(total_vec * 100, 2)
                        result["vector_idle_pct"] = round((1.0 - total_vec) * 100, 2)
                        break
        except Exception as e:
            print(f"  WARNING: Failed to parse {fp}: {e}")

    if aic_csvs or aiv_csvs:
        result["source"] = f"sample-based PMU (aic={len(aic_csvs)} csvs, aiv={len(aiv_csvs)} csvs)"

    # ── Strategy 2: Parse op_summary for core time distribution ──
    op_csvs = []
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith('op_summary') and f.endswith('.csv'):
                op_csvs.append(os.path.join(root, f))

    if op_csvs:
        csv_path = sorted(op_csvs)[-1]
        aic_dur = aiv_dur = aicpu_dur = total_dur = 0.0
        aic_ops = aiv_ops = aicpu_ops = total_ops = 0

        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    task_type = (row.get("Task Type") or "").strip()
                    dur_us = float(row.get("Task Duration(us)", 0))
                except (ValueError, TypeError):
                    continue
                if dur_us <= 0:
                    continue

                total_ops += 1
                total_dur += dur_us
                if "AI_CORE" in task_type and "AI_VECTOR" not in task_type:
                    aic_ops += 1
                    aic_dur += dur_us
                elif "AI_VECTOR" in task_type:
                    aiv_ops += 1
                    aiv_dur += dur_us
                elif "AICPU" in task_type:
                    aicpu_ops += 1
                    aicpu_dur += dur_us

        if total_dur > 0:
            result["aic_time_pct"] = round(aic_dur / total_dur * 100, 2)
            result["aiv_time_pct"] = round(aiv_dur / total_dur * 100, 2)
            result["aicpu_time_pct"] = round(aicpu_dur / total_dur * 100, 2)
            result["total_kernel_instances"] = total_ops
            result["aic_ops"] = aic_ops
            result["aiv_ops"] = aiv_ops

        if result["source"] is None:
            result["source"] = f"task-based op_summary ({total_ops} kernel instances)"

    return result


# ═══════════════════════════════════════════════════════════════════
# Training Benchmark (minimal — just needs stable steps for PMU)
# ═══════════════════════════════════════════════════════════════════

def run_training(steps, device_id):
    """Run GPT-2 XL training, return timing + loss."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[S1b] Loading GPT-2 XL, compiling GRAPH_MODE...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.set_seed(42)
    ms.common.set_seed(42)

    t0 = time.perf_counter()
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    t_model = time.perf_counter() - t0

    params = list(model.trainable_params())
    total_elems = sum(int(p.size) for p in params)
    total_fp16_gb = total_elems * 2 / 1e9
    print(f"  Model init: {t_model:.1f}s, {len(params)} params, {total_fp16_gb:.2f} GB FP16")

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    class TrainCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad(); self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            return ops.Depend()(loss, self.opt(grads))

    cell = TrainCell(model, opt)
    ms_model = ms.Model(cell)
    t_cell = time.perf_counter() - t0

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(steps)

    step_times_ms = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            step_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  Training {steps} steps (sink_size=1)...")
    ms_model.train(epoch=steps, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=1)

    compile_epoch = step_times_ms[0] if step_times_ms else 0
    warm = step_times_ms[1:] if len(step_times_ms) > 1 else [0]
    avg_step = float(np.mean(warm)) if warm else 0

    print(f"  Compile: {compile_epoch:.0f}ms  Avg step: {avg_step:.1f}ms (n={len(warm)})")

    return {
        "model_init_s": round(t_model, 1),
        "cell_build_s": round(t_cell, 1),
        "compile_epoch_ms": compile_epoch,
        "warm_steps": len(warm),
        "avg_step_ms": avg_step,
        "all_step_times_ms": [round(t, 1) for t in step_times_ms],
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 1b: Device-level PMU Profiling")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--parse-only", action="store_true",
                        help="Parse existing msprof output (post-processing)")
    parser.add_argument("--profiler-dir", type=str, default=None,
                        help="Path to msprof PROF_* directory")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "step1b_pmu.json"))
    args = parser.parse_args()

    if args.parse_only:
        prof_dir = args.profiler_dir
        if not prof_dir:
            prof_base = os.path.join(REPO, "output", "profiling_vec", "step1b")
            dirs = sorted(gb.glob(os.path.join(prof_base, "PROF_*")))
            if dirs:
                prof_dir = dirs[-1]
                print(f"Auto-detected: {prof_dir}")
            else:
                print(f"ERROR: --profiler-dir required")
                return 1

        print(f"Parsing PMU from: {prof_dir}")
        pmu = parse_sample_pmu_csvs(prof_dir)

        # Load partial timing
        partial = os.path.join(OUTPUT_DIR, "step1b_benchmark_partial.json")
        if os.path.exists(partial):
            with open(partial) as f:
                results = json.load(f)
        else:
            results = {"experiment": "Step 1b: Device-level PMU Profiling", "model": "GPT-2 XL"}

        results["pmu"] = pmu

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("STEP 1b DEVICE-LEVEL PMU")
        print(f"{'='*60}")
        if pmu.get("cube_mac_util_pct"):
            print(f"  Cube MAC util:     {pmu['cube_mac_util_pct']:.1f}%")
        if pmu.get("vector_alu_util_pct"):
            print(f"  Vector ALU util:   {pmu['vector_alu_util_pct']:.1f}%")
        if pmu.get("vector_idle_pct"):
            print(f"  Vector idle:       {pmu['vector_idle_pct']:.1f}%")
        print(f"  Source:            {pmu.get('source', 'unknown')}")
        if pmu.get("aic_time_pct"):
            print(f"  AIC time:  {pmu['aic_time_pct']}%  "
                  f"AIV time: {pmu['aiv_time_pct']}%  "
                  f"kernels: {pmu['total_kernel_instances']}")
        print(f"  → Saved: {args.output}")
        print(f"{'='*60}")
    else:
        timing = run_training(args.steps, args.device_id)
        results = {
            "experiment": "Step 1b: Device-level PMU Profiling",
            "model": "GPT-2 XL (48L/1600d)",
            "config": {
                "seq_len": SEQ_LEN, "batch_size": 1, "steps": args.steps,
                "sink_size": 1, "mode": "GRAPH_MODE",
                "optimizer": "AdamWeightDecay", "learning_rate": 1e-5,
                "device_id": args.device_id,
            },
            "timing": timing,
        }
        partial = os.path.join(OUTPUT_DIR, "step1b_benchmark_partial.json")
        with open(partial, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Partial results → {partial}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
