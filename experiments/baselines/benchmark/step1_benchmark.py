#!/usr/bin/env python3
"""
Step 1 Benchmark: GPT-2 XL Pure Training Baseline
==================================================
Measures pure training performance without any checkpoint ops.

Metrics:
  S1.1: Per-step wall-clock time (sink_size=1)
  S1.2: GRAPH_MODE compile time
  S1.3: Cube Engine utilization (msprof CSV parse)
  S1.4: Vector Engine utilization + idle ratio (msprof CSV parse)
  S1.5: HBM usage before/during/after (background npu-smi watcher)
  S1.6: GE graph compilation info — kernel instances, core distribution
  S1.7: SPDK raw BW (reuse Phase 5 S5 result)

Two-phase usage:
  Phase 1 (under msprof):
    msprof --output=<prof_dir> -- python step1_benchmark.py --steps 20 --device-id 1
    → spawns HBM watcher, runs training, saves partial timing JSON

  Phase 2 (parse msprof output):
    python step1_benchmark.py --parse-only --profiler-dir <PROF_*> --output <final.json>
    → parses msprof CSV, merges with timing data, produces final benchmark

Quick test (no profiler):
    python step1_benchmark.py --steps 5 --device-id 1 --no-hbm-watch
"""

import os, sys, time, json, argparse, csv, glob as gb, subprocess, signal, re, math

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
# HBM Watcher
# ═══════════════════════════════════════════════════════════════════

class HbmWatcher:
    """Background subprocess that polls npu-smi every second."""

    def __init__(self, log_path, device_id=1):
        self.log_path = log_path
        self.device_id = device_id
        self._proc = None

    def start(self):
        script = (
            f'while true; do '
            f'echo "TS $(date +%s.%N)"; '
            f'npu-smi info -t usages -i {self.device_id} 2>&1; '
            f'sleep 1; '
            f'done'
        )
        self._proc = subprocess.Popen(
            ['bash', '-c', script],
            stdout=open(self.log_path, 'w'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,  # detach from parent process group
        )
        return self

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait()


def parse_hbm_log(log_path):
    """Parse npu-smi watcher log, extract HBM usage stats."""
    if not os.path.exists(log_path):
        return None

    pct_vals = []
    total_mb = 65536
    with open(log_path, 'r') as f:
        for line in f:
            # Format: "HBM Usage Rate(%)              : 33"
            m_pct = re.search(r'HBM (?:Capacity|Usage)\s*\(.*?\)\s*:\s*(\d+)', line, re.IGNORECASE)
            if m_pct:
                val = int(m_pct.group(1))
                if 'capacity' in line.lower():
                    total_mb = val
                else:
                    pct_vals.append(val)

    if not pct_vals:
        return {"error": "No memory usage lines found in log"}

    return {
        "samples": len(pct_vals),
        "min_pct": min(pct_vals),
        "max_pct": max(pct_vals),
        "avg_pct": round(sum(pct_vals) / len(pct_vals), 1),
        "first_pct": pct_vals[0],
        "last_pct": pct_vals[-1],
        "total_hbm_mb": total_mb,
        "peak_pct": max(pct_vals),
        "peak_mb": round(max(pct_vals) / 100 * total_mb, 0),
    }


# ═══════════════════════════════════════════════════════════════════
# msprof CSV Parser (S1.3, S1.4, S1.6)
# ═══════════════════════════════════════════════════════════════════

def parse_msprof_csv(csv_path):
    """Parse msprof op_summary CSV, return core utilization + GE graph stats."""
    stats = {
        "total_kernel_instances": 0,
        "total_dur_us": 0.0,
        # Core distribution
        "aic_count": 0, "aic_dur_us": 0.0,       # AI_CORE = Cube
        "aiv_count": 0, "aiv_dur_us": 0.0,        # AI_VECTOR_CORE
        "aicpu_count": 0, "aicpu_dur_us": 0.0,    # AICPU
        # Weighted sums for utilization
        "aic_mac_weighted_sum": 0.0,
        "aic_scalar_weighted_sum": 0.0,
        "aiv_vec_weighted_sum": 0.0,
        "aiv_scalar_weighted_sum": 0.0,
        "aic_mac_active_us": 0.0,
        "aiv_vec_active_us": 0.0,
        # Op type histogram (for S1.6 GE graph info)
        "op_type_counts": {},
        "op_type_dur_us": {},
        # Delta-related ops specifically
        "delta_ops": {},
    }

    delta_op_types = {
        "Sub", "Mul", "ReduceSum", "Cast", "Reshape", "Concat",
        "ZerosLike", "OnesLike", "Add", "Abs", "ReduceMax", "Div",
        "Round", "ClipByValue", "TopK", "Gather", "ScatterUpdate",
    }

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h.strip(): i for i, h in enumerate(header)}

        for row in reader:
            if len(row) < max(col.values()) + 1:
                continue
            try:
                task_type = row[col.get("Task Type", 7)].strip()
                op_type   = row[col.get("OP Type", 5)].strip()
                op_name   = row[col.get("Op Name", 4)].strip()
                dur_us    = float(row[col.get("Task Duration(us)", 9)].strip())
            except (IndexError, ValueError, KeyError):
                continue

            stats["total_kernel_instances"] += 1
            stats["total_dur_us"] += dur_us

            # Op type histogram
            stats["op_type_counts"][op_type] = stats["op_type_counts"].get(op_type, 0) + 1
            stats["op_type_dur_us"][op_type] = stats["op_type_dur_us"].get(op_type, 0.0) + dur_us

            # Core type
            is_aic  = "AI_CORE" in task_type and "AI_VECTOR" not in task_type
            is_aiv  = "AI_VECTOR" in task_type
            is_aicpu = "AICPU" in task_type

            if is_aic:
                stats["aic_count"] += 1
                stats["aic_dur_us"] += dur_us
                try:
                    mac_r    = float(row[col.get("aic_mac_ratio", 24)].strip())
                    scalar_r = float(row[col.get("aic_scalar_ratio", 26)].strip())
                    stats["aic_mac_weighted_sum"]    += mac_r * dur_us
                    stats["aic_scalar_weighted_sum"] += scalar_r * dur_us
                    stats["aic_mac_active_us"]       += mac_r / 100.0 * dur_us
                except (ValueError, KeyError):
                    pass
            elif is_aiv:
                stats["aiv_count"] += 1
                stats["aiv_dur_us"] += dur_us
                try:
                    vec_r    = float(row[col.get("aiv_vec_ratio", 37)].strip())
                    scalar_r = float(row[col.get("aiv_scalar_ratio", 39)].strip())
                    stats["aiv_vec_weighted_sum"]    += vec_r * dur_us
                    stats["aiv_scalar_weighted_sum"] += scalar_r * dur_us
                    stats["aiv_vec_active_us"]       += vec_r / 100.0 * dur_us
                except (ValueError, KeyError):
                    pass
            elif is_aicpu:
                stats["aicpu_count"] += 1
                stats["aicpu_dur_us"] += dur_us

            # Delta-related ops
            if op_type in delta_op_types:
                d = stats["delta_ops"].setdefault(op_type, {
                    "count": 0, "dur_us": 0.0, "core_type": task_type, "names": set()
                })
                d["count"] += 1
                d["dur_us"] += dur_us
                d["names"].add(op_name[:80])

    # Compute weighted utilizations
    if stats["aic_dur_us"] > 0:
        stats["aic_mac_util_pct"]    = round(stats["aic_mac_weighted_sum"] / stats["aic_dur_us"], 2)
        stats["aic_scalar_util_pct"] = round(stats["aic_scalar_weighted_sum"] / stats["aic_dur_us"], 2)
    if stats["aiv_dur_us"] > 0:
        stats["aiv_vec_util_pct"]    = round(stats["aiv_vec_weighted_sum"] / stats["aiv_dur_us"], 2)
        stats["aiv_scalar_util_pct"] = round(stats["aiv_scalar_weighted_sum"] / stats["aiv_dur_us"], 2)

    # Convert sets to lists for JSON
    for ot in stats["delta_ops"]:
        stats["delta_ops"][ot]["names"] = sorted(stats["delta_ops"][ot]["names"])

    # Sort op type histograms by count desc
    stats["op_type_counts"] = dict(
        sorted(stats["op_type_counts"].items(), key=lambda x: -x[1]))
    stats["op_type_dur_us"] = dict(
        sorted(stats["op_type_dur_us"].items(), key=lambda x: -x[1]))

    return stats


# ═══════════════════════════════════════════════════════════════════
# Training + Benchmark
# ═══════════════════════════════════════════════════════════════════

def run_benchmark(steps, device_id, no_hbm_watch=False):
    """Run pure GPT-2 XL training, return timing + loss + HBM data."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hbm_log = os.path.join(OUTPUT_DIR, "step1_hbm_usage.log")

    # ── S1.5a: HBM before ──
    print("\n[S1.5] HBM usage before model load...")
    hbm_before = None
    try:
        hbm_before = os.popen(f"npu-smi info -t usages -i {device_id} 2>/dev/null").read().strip()
        print(f"  {hbm_before}")
    except Exception:
        print("  (npu-smi not available)")

    # ── Start HBM watcher ──
    watcher = None
    if not no_hbm_watch:
        print(f"\n  Starting HBM watcher → {hbm_log}")
        watcher = HbmWatcher(hbm_log, device_id).start()
        time.sleep(1)  # let it capture first sample

    try:
        # ── S1.2: Model load + compile ──
        print(f"\n[S1.2] Loading GPT-2 XL and compiling in GRAPH_MODE...")
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
        print(f"  Model init: {t_model:.1f}s")

        params = list(model.trainable_params())
        total_elems = sum(int(p.size) for p in params)
        total_fp16_mb = total_elems * 2 / (1024 * 1024)
        total_fp16_gb = total_fp16_mb / 1024
        print(f"  Trainable params: {len(params)}")
        print(f"  Total elements:   {total_elems:,} ({total_fp16_mb:.0f} MB = {total_fp16_gb:.2f} GB FP16)")

        # Layer statistics (for S1.6)
        layer_elems = {}
        for p in params:
            m = re.search(r'backbone\.blocks\.(\d+)\.', p.name)
            lid = int(m.group(1)) if m else -1
            layer_elems[lid] = layer_elems.get(lid, 0) + int(p.size)
        n_layers = len([k for k in layer_elems if k >= 0])

        opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

        class TrainCell(nn.Cell):
            def __init__(self, net, opt):
                super().__init__(auto_prefix=False)
                self.net = net
                self.net.set_grad()
                self.opt = opt
                self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            def construct(self, *inp):
                loss, grads = self.gf(*inp)
                return ops.Depend()(loss, self.opt(grads))

        cell = TrainCell(model, opt)
        ms_model = ms.Model(cell)
        t_cell = time.perf_counter() - t0
        print(f"  Cell build: {t_cell:.1f}s")

        # ── Dataset ──
        ds_train = ms.dataset.MindDataset(
            REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
            shuffle=True)
        ds_train = ds_train.batch(1, drop_remainder=True).take(steps)

        # ── S1.1: Train with timing ──
        print(f"\n[S1.1] Training {steps} steps (sink_size=1 — epoch per step)...")

        step_times_ms = []
        loss_values = []

        class BenchmarkCB(ms.Callback):
            def __init__(self):
                self.t_epoch_begin = None
                self.step_count = 0
            def on_train_epoch_begin(self, rc):
                self.t_epoch_begin = time.perf_counter()
            def on_train_epoch_end(self, rc):
                elapsed_ms = (time.perf_counter() - self.t_epoch_begin) * 1000
                self.step_count += 1
                step_times_ms.append(elapsed_ms)
                cb_params = rc.original_args()
                try:
                    net_out = cb_params.net_outputs
                    if hasattr(net_out, 'asnumpy'):
                        loss_values.append(float(net_out.asnumpy().flatten()[0]))
                    elif isinstance(net_out, (list, tuple)):
                        loss_values.append(float(net_out[0].asnumpy().flatten()[0]))
                except Exception:
                    pass

        cb = BenchmarkCB()

        t_train = time.perf_counter()
        ms_model.train(epoch=steps, train_dataset=ds_train, callbacks=[cb],
                       dataset_sink_mode=True, sink_size=1)
        t_train_end = time.perf_counter()

        total_train_s = t_train_end - t_train

        # ── Compute step stats ──
        if len(step_times_ms) >= 2:
            compile_epoch_ms = step_times_ms[0]
            warm_ms = step_times_ms[1:]
        else:
            compile_epoch_ms = step_times_ms[0] if step_times_ms else 0
            warm_ms = step_times_ms or [0]

        avg_step_ms = float(np.mean(warm_ms)) if warm_ms else 0
        std_step_ms = float(np.std(warm_ms)) if warm_ms else 0
        min_step_ms = float(np.min(warm_ms)) if warm_ms else 0
        max_step_ms = float(np.max(warm_ms)) if warm_ms else 0

        print(f"\n  Compile (epoch 0): {compile_epoch_ms:.0f}ms")
        print(f"  Warm steps: {len(warm_ms)}")
        print(f"  Avg step:   {avg_step_ms:.1f}ms")
        print(f"  Std step:   {std_step_ms:.1f}ms")
        print(f"  Min step:   {min_step_ms:.1f}ms")
        print(f"  Max step:   {max_step_ms:.1f}ms")
        print(f"  Total:      {total_train_s:.1f}s")
        if warm_ms:
            print(f"  COV:        {std_step_ms/avg_step_ms*100:.1f}%")

        # ── S1.5b: HBM after ──
        print(f"\n[S1.5] HBM usage after training...")
        hbm_after = None
        try:
            hbm_after = os.popen(f"npu-smi info -t usages -i {device_id} 2>/dev/null").read().strip()
            print(f"  {hbm_after}")
        except Exception:
            print("  (npu-smi not available)")

        # ── Loss ──
        avg_loss = float(np.mean(loss_values)) if loss_values else 0
        print(f"\n  Loss: init={loss_values[0] if loss_values else 'N/A'}  "
              f"final={loss_values[-1] if loss_values else 'N/A'}  avg={avg_loss:.4f}")

        # ── Build partial results ──
        results = {
            "experiment": "Step 1 Benchmark: GPT-2 XL Pure Training",
            "model": "GPT-2 XL (48L/1600d)",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "seq_len": SEQ_LEN,
                "batch_size": 1,
                "steps": steps,
                "sink_size": 1,
                "mode": "GRAPH_MODE",
                "learning_rate": 1e-5,
                "optimizer": "AdamWeightDecay",
                "n_trainable_params": len(params),
                "n_transformer_layers": n_layers,
                "total_elems": total_elems,
                "fp16_mb": round(total_fp16_mb, 1),
                "fp16_gb": round(total_fp16_gb, 2),
                "device_id": device_id,
            },
            "S1.1_perf": {
                "compile_epoch_ms": compile_epoch_ms,
                "warm_steps": len(warm_ms),
                "avg_step_ms": avg_step_ms,
                "std_step_ms": std_step_ms,
                "min_step_ms": min_step_ms,
                "max_step_ms": max_step_ms,
                "cov_pct": round(std_step_ms/avg_step_ms*100, 1) if avg_step_ms > 0 else 0,
                "total_train_s": round(total_train_s, 3),
                "all_step_times_ms": [round(t, 1) for t in step_times_ms],
            },
            "S1.2_compile": {
                "model_init_s": round(t_model, 1),
                "cell_build_s": round(t_cell, 1),
                "first_epoch_with_compile_ms": compile_epoch_ms,
            },
            "S1.5_hbm": {
                "before": hbm_before,
                "during_log_file": hbm_log,
                "after": hbm_after,
            },
            "S1_loss": {
                "init": loss_values[0] if loss_values else None,
                "final": loss_values[-1] if loss_values else None,
                "avg": round(avg_loss, 6),
            },
        }

        out = os.path.join(OUTPUT_DIR, "step1_benchmark_partial.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Partial results → {out}")

    finally:
        # ── Stop HBM watcher ──
        if watcher:
            watcher.stop()
            hbm_stats = parse_hbm_log(hbm_log)
            if hbm_stats and "samples" in hbm_stats:
                print(f"\n  HBM watcher: {hbm_stats['samples']} samples, "
                      f"min={hbm_stats['min_pct']}%, max={hbm_stats['max_pct']}%, "
                      f"avg={hbm_stats['avg_pct']}% (peak {hbm_stats['peak_mb']:.0f} MB / {hbm_stats['total_hbm_mb']} MB)")
                # Write HBM stats to a separate file
                with open(hbm_log.replace('.log', '.json'), 'w') as f:
                    json.dump(hbm_stats, f, indent=2)
            elif hbm_stats and "error" in hbm_stats:
                print(f"\n  HBM watcher parse error: {hbm_stats['error']}")
            else:
                print(f"\n  HBM watcher: no valid data parsed from {hbm_log}")

    return results


# ═══════════════════════════════════════════════════════════════════
# SPDK BW Reference (S1.7)
# ═══════════════════════════════════════════════════════════════════

def load_spdk_bw_reference():
    """Load SPDK raw BW from Phase 5 S5 result."""
    s5_path = os.path.join(REPO, "experiments", "output", "phase5_s5_spdk_raw_bw.json")
    if os.path.exists(s5_path):
        try:
            with open(s5_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════
# Post-processing: merge msprof CSV with benchmark data
# ═══════════════════════════════════════════════════════════════════

def parse_mode(profiler_dir, output_path):
    """Parse msprof CSV and merge with partial benchmark results."""

    # ── Find CSV ──
    csv_pattern = os.path.join(profiler_dir, "mindstudio_profiler_output", "op_summary_*.csv")
    csv_files = sorted(gb.glob(csv_pattern))
    if not csv_files:
        print(f"ERROR: No op_summary CSV found at {csv_pattern}")
        print(f"  Contents of {profiler_dir}:")
        for root, dirs, files in os.walk(profiler_dir):
            for f in files:
                print(f"    {os.path.join(root, f)}")
        return None
    csv_path = csv_files[-1]  # newest
    print(f"\nParsing msprof CSV: {csv_path}")
    print(f"  File size: {os.path.getsize(csv_path):,} bytes")

    # ── Parse CSV ──
    pmu = parse_msprof_csv(csv_path)

    print(f"\n  Core Distribution:")
    total_dur = pmu["total_dur_us"]
    aic_pct = pmu["aic_dur_us"] / total_dur * 100 if total_dur > 0 else 0
    aiv_pct = pmu["aiv_dur_us"] / total_dur * 100 if total_dur > 0 else 0
    aicpu_pct = pmu["aicpu_dur_us"] / total_dur * 100 if total_dur > 0 else 0
    print(f"    AI_CORE (Cube):  {pmu['aic_count']:6d} ops  {pmu['aic_dur_us']/1e6:.2f}s  ({aic_pct:.1f}%)")
    print(f"    AI_VECTOR:       {pmu['aiv_count']:6d} ops  {pmu['aiv_dur_us']/1e6:.2f}s  ({aiv_pct:.1f}%)")
    print(f"    AICPU:           {pmu['aicpu_count']:6d} ops  {pmu['aicpu_dur_us']/1e6:.2f}s  ({aicpu_pct:.1f}%)")
    print(f"    TOTAL:           {pmu['total_kernel_instances']:6d} ops  {total_dur/1e6:.2f}s")

    print(f"\n  Core Utilization (weighted avg):")
    if "aic_mac_util_pct" in pmu:
        print(f"    Cube MAC util:    {pmu['aic_mac_util_pct']:.1f}%")
        print(f"    Cube Scalar util: {pmu['aic_scalar_util_pct']:.1f}%")
    if "aiv_vec_util_pct" in pmu:
        print(f"    Vector ALU util:  {pmu['aiv_vec_util_pct']:.1f}%")
        print(f"    Vector Scalar:    {pmu['aiv_scalar_util_pct']:.1f}%")

    print(f"\n  Top-10 Op Types by Count:")
    for i, (op, cnt) in enumerate(list(pmu["op_type_counts"].items())[:10]):
        dur = pmu["op_type_dur_us"].get(op, 0)
        print(f"    {i+1:2d}. {op:25s} {cnt:6d} ops  {dur/1e3:8.2f}ms")

    # ── Load partial results ──
    partial_path = os.path.join(OUTPUT_DIR, "step1_benchmark_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            results = json.load(f)
        print(f"\n  Merged with: {partial_path}")
    else:
        results = {"experiment": "Step 1 Benchmark", "model": "GPT-2 XL"}
        print(f"\n  WARNING: No partial results at {partial_path} — PMU-only output")

    # ── Load HBM stats ──
    hbm_json = os.path.join(OUTPUT_DIR, "step1_hbm_usage.json")
    if os.path.exists(hbm_json):
        with open(hbm_json) as f:
            results["S1.5_hbm"]["stats"] = json.load(f)

    # ── Add PMU data ──
    results["S1.3_cube"] = {
        "aic_total_ops": pmu["aic_count"],
        "aic_total_dur_s": round(pmu["aic_dur_us"] / 1e6, 4),
        "aic_mac_util_pct": pmu.get("aic_mac_util_pct"),
        "aic_scalar_util_pct": pmu.get("aic_scalar_util_pct"),
        "aic_active_pct_of_total": round(aic_pct, 1),
    }
    results["S1.4_vector"] = {
        "aiv_total_ops": pmu["aiv_count"],
        "aiv_total_dur_s": round(pmu["aiv_dur_us"] / 1e6, 4),
        "aiv_vec_util_pct": pmu.get("aiv_vec_util_pct"),
        "aiv_scalar_util_pct": pmu.get("aiv_scalar_util_pct"),
        "aiv_active_pct_of_total": round(aiv_pct, 1),
    }
    results["S1.6_ge_graph"] = {
        "total_kernel_instances": pmu["total_kernel_instances"],
        "total_dur_s": round(pmu["total_dur_us"] / 1e6, 4),
        "aicpu_ops": pmu["aicpu_count"],
        "aicpu_dur_s": round(pmu["aicpu_dur_us"] / 1e6, 4),
        "top_op_types": dict(list(pmu["op_type_counts"].items())[:20]),
        "delta_related_ops": {ot: {"count": d["count"], "dur_us": round(d["dur_us"], 1),
                                    "core_type": d["core_type"]}
                              for ot, d in sorted(pmu["delta_ops"].items())},
    }

    # ── Add S1.7 SPDK BW reference ──
    spdk_ref = load_spdk_bw_reference()
    if spdk_ref:
        results["S1.7_spdk_bw"] = spdk_ref
        print(f"  S1.7 SPDK BW: loaded from phase5_s5_spdk_raw_bw.json")
    else:
        results["S1.7_spdk_bw"] = {"note": "Reuse Phase 5 S5 result; file not found, re-run phase5_s5_spdk_raw_bw.py"}

    # ── Add msprof source reference ──
    results["_meta"] = {
        "msprof_csv": csv_path,
        "profiler_dir": profiler_dir,
        "parse_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Write final ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Final results → {output_path}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print("STEP 1 BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:          GPT-2 XL (48L/1600d)")
    print(f"  Trainable params: {results['config']['n_trainable_params']}")
    print(f"  Model size (FP16): {results['config']['fp16_gb']:.2f} GB")
    print(f"  GE kernel instances: {pmu['total_kernel_instances']:,}")
    print(f"  Compile+first step:  {results['S1.2_compile']['first_epoch_with_compile_ms']:.0f} ms")
    print(f"  Avg step (warm):     {results['S1.1_perf']['avg_step_ms']:.1f} ms ±{results['S1.1_perf']['std_step_ms']:.1f}")
    print(f"  COV:                 {results['S1.1_perf']['cov_pct']}%")
    if "aic_mac_util_pct" in pmu:
        print(f"  Cube MAC util:       {pmu['aic_mac_util_pct']:.1f}%")
    if "aiv_vec_util_pct" in pmu:
        print(f"  Vector ALU util:     {pmu['aiv_vec_util_pct']:.1f}%")
    if "stats" in results.get("S1.5_hbm", {}):
        s = results["S1.5_hbm"]["stats"]
        print(f"  HBM peak:            {s.get('peak_pct','?')}% ({s.get('peak_mb','?')} MB / {s.get('total_hbm_mb','?')} MB)")
    print(f"{'='*70}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 1 Benchmark: GPT-2 XL Pure Training")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--no-hbm-watch", action="store_true",
                        help="Disable background HBM watcher")
    parser.add_argument("--parse-only", action="store_true",
                        help="Parse msprof CSV and merge results (post-processing mode)")
    parser.add_argument("--profiler-dir", type=str, default=None,
                        help="Path to msprof PROF_* directory (for --parse-only)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "step1_benchmark.json"),
                        help="Final output JSON path (for --parse-only)")
    args = parser.parse_args()

    if args.parse_only:
        if not args.profiler_dir:
            # Try auto-detect
            prof_base = os.path.join(REPO, "output", "profiling_vec", "step1")
            prof_dirs = sorted(gb.glob(os.path.join(prof_base, "PROF_*")))
            if prof_dirs:
                args.profiler_dir = prof_dirs[-1]
                print(f"Auto-detected profiler dir: {args.profiler_dir}")
            else:
                print(f"ERROR: --profiler-dir required (or no PROF_* found under {prof_base})")
                return 1
        parse_mode(args.profiler_dir, args.output)
    else:
        run_benchmark(args.steps, args.device_id, args.no_hbm_watch)

    return 0


if __name__ == "__main__":
    sys.exit(main())
