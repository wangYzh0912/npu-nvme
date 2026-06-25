#!/usr/bin/env python3
"""
Phase 1a: GE Scheduling Verification — A1 (no-inject) vs A2 (200-param inject).

Strategy: per-step wall time measured via Python callback, PMU collected via msprof CLI wrapper.
The msprof approach is more reliable than the Python Profiler API in this MS version.

Output:
  experiments/output/phase1a_a1.json  — A1 (baseline)
  experiments/output/phase1a_a2.json  — A2 (200-param inject)
  output/profiling_vec/A1/           — msprof raw data
  output/profiling_vec/A2/           — msprof raw data

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /home/user7/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/phase1a_pmu_verify.py --exp A1,A2'
"""
import os, sys, time, json, math, glob, csv, shutil, argparse, subprocess
from collections import defaultdict

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

DEVICE_ID = 1
SEQ_LEN = 1024
SINK_SIZE = 4
TOTAL_STEPS = 12
EPOCHS = 3

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
PROF_DIR_BASE = os.path.join(REPO, "output", "profiling_vec")
MSPROF = "/usr/local/Ascend/ascend-toolkit/latest/bin/msprof"


def parse_pmu_csvs(csv_files):
    """Parse MindSpore PMU CSV files."""
    if not csv_files:
        return {"error": "no_csv_found"}

    by_core = defaultdict(lambda: {
        "total_time_us": 0.0, "kernel_count": 0,
        "ops": defaultdict(lambda: {"count": 0, "total_us": 0.0}),
        "pmu_fields": defaultdict(float),
        "ratio_count": 0,
    })

    total_op_time_us = 0.0
    for fp in csv_files:
        try:
            with open(fp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    core = (row.get("Core Type") or row.get("core_type") or "")
                    op_name = (row.get("Kernel Name") or row.get("op_name")
                               or row.get("Node Type") or "unknown")
                    dur_str = (row.get("Duration(us)") or row.get("total_time_us")
                               or row.get("Total Cycle") or "0")
                    try:
                        dur_us = float(dur_str)
                    except (ValueError, TypeError):
                        continue
                    if dur_us <= 0:
                        continue

                    b = by_core[core]
                    b["total_time_us"] += dur_us
                    b["kernel_count"] += 1
                    b["ops"][op_name]["count"] += 1
                    b["ops"][op_name]["total_us"] += dur_us
                    total_op_time_us += dur_us

                    for rkey in ["aic_mac_fp16_ratio", "aic_cube_fops",
                                 "aiv_vec_fp16_ratio", "aiv_vec_fp32_ratio",
                                 "aiv_vec_misc_ratio"]:
                        val = row.get(rkey)
                        if val is not None and val != "":
                            try:
                                b["pmu_fields"][rkey] += float(val)
                            except (ValueError, TypeError):
                                pass
                    b["ratio_count"] += 1
        except Exception:
            continue

    if total_op_time_us == 0:
        return {"error": "no_valid_kernel_data", "files_scanned": len(csv_files)}

    def _avg(b, key):
        s = b["pmu_fields"].get(key, 0)
        n = max(b.get("ratio_count", 1), 1)
        return round(s / n * 100, 2)

    core_summary = {}
    for ct, data in sorted(by_core.items()):
        oss = sorted(data["ops"].items(), key=lambda kv: kv[1]["total_us"], reverse=True)
        core_summary[ct] = {
            "total_time_us": round(data["total_time_us"], 2),
            "pct_of_total": round(data["total_time_us"] / total_op_time_us * 100, 2),
            "kernel_count": data["kernel_count"],
            "top_ops": [
                {"op": op, "count": s["count"],
                 "total_us": round(s["total_us"], 2),
                 "ratio_pct": round(s["total_us"] / total_op_time_us * 100, 3)}
                for op, s in oss[:15]
            ],
        }

    aic = by_core.get("AI_CORE", {})
    aiv = by_core.get("AI_VECTOR_CORE", {})
    aic_time = aic.get("total_time_us", 0)
    aiv_time = aiv.get("total_time_us", 0)

    aic_mac = _avg(aic, "aic_mac_fp16_ratio")
    aiv_fp16 = _avg(aiv, "aiv_vec_fp16_ratio")
    aiv_fp32 = _avg(aiv, "aiv_vec_fp32_ratio")
    aiv_misc = _avg(aiv, "aiv_vec_misc_ratio")

    vec_eff = aiv_fp16 + aiv_fp32
    cube_eff = aic_mac
    vec_idle_ms = (aiv_time / 1000) * (1 - vec_eff / 100) if vec_eff < 100 else 0

    synthesis = {
        "total_op_time_us": round(total_op_time_us, 2),
        "aic_time_pct": round(aic_time / total_op_time_us * 100, 2) if total_op_time_us else 0,
        "aiv_time_pct": round(aiv_time / total_op_time_us * 100, 2) if total_op_time_us else 0,
        "cube_eff_util_pct": cube_eff,
        "vec_eff_util_pct": vec_eff,
        "vec_idle_pct": round(100 - vec_eff, 2),
        "vec_idle_ms_est": round(vec_idle_ms, 2),
        "aic_mac_fp16_ratio_pct": aic_mac,
        "aiv_vec_fp16_ratio_pct": aiv_fp16,
        "aiv_vec_fp32_ratio_pct": aiv_fp32,
        "aiv_vec_misc_ratio_pct": aiv_misc,
    }

    return {
        "by_core_type": core_summary,
        "synthesis": synthesis,
        "files_parsed": len(csv_files),
    }


def run_one(exp_label, inject_params):
    """
    1) Start msprof in background
    2) Train in-process (wall-time timing via Python callback)
    3) Stop msprof, collect CSVs, parse PMU
    """
    print(f"\n{'='*60}")
    print(f"  {exp_label}: inject={inject_params} params")
    print(f"{'='*60}", flush=True)

    prof_dir = os.path.join(PROF_DIR_BASE, exp_label)
    if os.path.exists(prof_dir):
        shutil.rmtree(prof_dir)
    os.makedirs(prof_dir, exist_ok=True)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)
    print(f"  [{exp_label}] Model built OK", flush=True)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    all_params = list(model.trainable_params())
    n_total = len(all_params)

    # Injection setup
    covered = all_params[:inject_params] if inject_params > 0 else []
    n_inject = len(covered)
    num_groups = max(1, min(math.ceil(n_inject / 100), 10)) if n_inject > 0 else 0
    param_groups = []; fp16_needed = []
    if n_inject > 0:
        gs = max(1, math.ceil(n_inject / num_groups))
        for g in range(num_groups):
            s = g * gs; e = min(s + gs, n_inject)
            if s < n_inject:
                pg = covered[s:e]
                param_groups.append(pg)
                fp16_needed.append([
                    hasattr(p, 'dtype') and p.dtype != ms.float16 for p in pg
                ])

    total_elems = sum(int(np.prod(p.shape)) for p in covered) if inject_params else 0
    print(f"  [{exp_label}] Total params={n_total}, Inject={n_inject}, "
          f"{total_elems/1e9:.2f}B elems, {num_groups} groups", flush=True)

    class ProfiledCell(nn.Cell):
        def __init__(self, network, optimizer, param_groups, fp16_needed, inject):
            super().__init__(auto_prefix=False)
            self.network = network; self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.param_groups = param_groups; self.fp16_needed = fp16_needed
            self.inject = inject

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)
            if self.inject:
                acc = Tensor([0.0], dtype=ms.float16)
                for gi, group in enumerate(self.param_groups):
                    flags = self.fp16_needed[gi]
                    flat_parts = []
                    for pi, p in enumerate(group):
                        pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                        flat_parts.append(ops.Reshape()(pv, (-1,)))
                    flat = flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
                    delta = ops.Sub()(flat, ops.ZerosLike()(flat))
                    red   = ops.ReduceSum()(delta)
                    c32   = ops.Cast()(red, ms.float32)
                    c16   = ops.Cast()(c32, ms.float16)
                    acc   = ops.Add()(acc, c16)
                loss = self.depend(loss, acc)
            opt_res = self.optimizer(grads)
            return self.depend(loss, opt_res)

    # ── Build + compile ──
    print(f"  [{exp_label}] Building cell...", flush=True)
    t_build = time.perf_counter()
    cell = ProfiledCell(model, opt, param_groups, fp16_needed, inject_params > 0)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print(f"  [{exp_label}] Build={build_s:.1f}s", flush=True)

    # ── Train (msprof launched externally in driver) ──
    epoch_times_ms = []
    step_times_ms = []

    class StepCB(ms.Callback):
        def __init__(self):
            self.last_epoch_start = 0
            self.last_step = 0
        def on_train_epoch_begin(self, rc):
            self.last_step = time.perf_counter()
            self.last_epoch_start = time.perf_counter()
        def on_train_step_end(self, rc):
            now = time.perf_counter()
            step_times_ms.append((now - self.last_step) * 1000)
            self.last_step = now
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.last_epoch_start) * 1000)

    print(f"  [{exp_label}] Starting {TOTAL_STEPS} steps...", flush=True)
    compiled_ok = True; error_msg = None
    cb = StepCB()
    t_total = time.perf_counter()

    try:
        ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[cb],
                       dataset_sink_mode=True, sink_size=SINK_SIZE)
    except Exception as e:
        compiled_ok = False; error_msg = str(e)[:300]
        print(f"  [{exp_label}] FAILED: {error_msg}", flush=True)

    total_s = time.perf_counter() - t_total

    # ── Timing stats ──
    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms)>1 else []
    avg_step = sum(warm_epochs)/len(warm_epochs)/SINK_SIZE if warm_epochs else 0

    # Per-step stats from StepCB (sink=TRUE → only fires at epoch boundary)
    # So we use epoch-based step calculation for wall time
    warm_steps = step_times_ms[TOTAL_STEPS:] if len(step_times_ms)>TOTAL_STEPS else []
    avg_step_cb = sum(warm_steps)/len(warm_steps) if warm_steps else avg_step

    print(f"  [{exp_label}] CompileEpoch={compile_epoch:.0f}ms  "
          f"AvgStep={avg_step:.0f}ms  (epoch)  "
          f"WarmEpochs={[f'{e:.0f}ms' for e in warm_epochs]}", flush=True)

    # ── Search for msprof output ──
    # msprof --output=<prof_dir> may create subdir or use different naming
    pmu_csvs = glob.glob(os.path.join(prof_dir, "**", "*.csv"), recursive=True)
    print(f"  [{exp_label}] PMU CSVs: {len(pmu_csvs)}", flush=True)
    for f in sorted(pmu_csvs)[:5]:
        print(f"    {os.path.relpath(f, PROF_DIR_BASE)}", flush=True)

    pmu = parse_pmu_csvs(pmu_csvs) if pmu_csvs else {}

    result = {
        "test": exp_label, "model": "GPT-2 XL 48L",
        "total_params": n_total, "inject_params": inject_params,
        "inject_elems_B": round(total_elems/1e9, 3), "num_groups": num_groups,
        "sink_size": SINK_SIZE, "total_steps": TOTAL_STEPS, "epochs": EPOCHS,
        "compiled_ok": compiled_ok, "error": error_msg,
        "build_s": round(build_s, 1), "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(et,0) for et in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
        "avg_step_cb_ms": round(avg_step_cb, 1),
        "per_step_times_ms": [round(s,1) for s in step_times_ms[:20]],  # first 20
        "pmu_data": pmu, "pmu_csv_count": len(pmu_csvs),
        "prof_dir": prof_dir,
    }

    if pmu and "synthesis" in pmu:
        s = pmu["synthesis"]
        print(f"  [{exp_label}] Cube={s['aic_time_pct']:.1f}% @{s['cube_eff_util_pct']:.1f}%  "
              f"Vec={s['aiv_time_pct']:.1f}% @{s['vec_eff_util_pct']:.1f}%  "
              f"VecIdle={s['vec_idle_ms_est']:.0f}ms", flush=True)
    else:
        print(f"  [{exp_label}] ⚠ No PMU data — msprof may need separate launch", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"phase1a_{exp_label.lower()}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{exp_label}] -> {os.path.basename(out_json)}", flush=True)

    # Clean up GE graph cache for next run
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    ms.reset_auto_parallel_context()
    import gc; gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="A1,A2")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    exps = args.exp.split(",")

    if "A1" in exps:
        results["A1"] = run_one("A1", inject_params=0)
    if "A2" in exps:
        results["A2"] = run_one("A2", inject_params=200)

    if "A1" in results and "A2" in results:
        r1, r2 = results["A1"], results["A2"]

        print(f"\n{'='*80}")
        print(f"{'Phase 1a — GE Scheduling Verification':^80}")
        print(f"{'='*80}")
        print(f"{'Metric':<40} {'A1 (baseline)':>15} {'A2 (inject)':>15} {'Delta':>12}")
        print("-"*80)

        def row(label, v1, v2, unit=""):
            d = v2-v1 if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) else 0
            p = d/v1*100 if v1 else 0
            print(f"{label:<40} {v1:>12.1f}{unit} {v2:>12.1f}{unit} {d:>+9.1f}{unit} ({p:+.1f}%)")

        row("Per-step wall time (epoch)", r1["avg_step_ms"], r2["avg_step_ms"], "ms")

        s1 = (r1.get("pmu_data") or {}).get("synthesis", {})
        s2 = (r2.get("pmu_data") or {}).get("synthesis", {})
        if s1 and s2:
            print("-"*80)
            row("Cube eff util", s1.get("cube_eff_util_pct",0), s2.get("cube_eff_util_pct",0), "%")
            row("Vector eff util", s1.get("vec_eff_util_pct",0), s2.get("vec_eff_util_pct",0), "%")
            row("Vector idle %", s1.get("vec_idle_pct",0), s2.get("vec_idle_pct",0), "%")
            print("-"*80)
            row("Cube time %", s1.get("aic_time_pct",0), s2.get("aic_time_pct",0), "%")
            row("Vector time %", s1.get("aiv_time_pct",0), s2.get("aiv_time_pct",0), "%")
            row("Vector idle est", s1.get("vec_idle_ms_est",0), s2.get("vec_idle_ms_est",0), "ms")
        print("="*80)

        # Verdict
        print(f"\n{' VERDICT ':=^60}")
        w1, w2 = r1["avg_step_ms"], r2["avg_step_ms"]
        step_ok = abs(w2-w1) < max(0.05*w1, 25)
        print(f"  {'✅' if step_ok else '❌'} Step time: A2={w2:.0f}ms vs A1={w1:.0f}ms (Δ={w2-w1:+.0f}ms)")

        if s1 and s2:
            cube_ok = abs(s2.get("cube_eff_util_pct",0)-s1.get("cube_eff_util_pct",0)) < 2.0
            vec_up = s2.get("vec_eff_util_pct",0) > s1.get("vec_eff_util_pct",0)+0.3
            print(f"  {'✅' if cube_ok else '❌'} Cube util unchanged: A2={s2.get('cube_eff_util_pct',0):.1f}% vs A1={s1.get('cube_eff_util_pct',0):.1f}%")
            print(f"  {'✅' if vec_up else '❌'} Vector util INCREASED: A2={s2.get('vec_eff_util_pct',0):.1f}% vs A1={s1.get('vec_eff_util_pct',0):.1f}%")
            if step_ok and cube_ok and vec_up:
                print(f"\n  ★★★ ALL CHECKS PASSED ★★★")
        else:
            print(f"  ⚠ PMU data missing — need msprof wrapper")

        # Save comparison
        cmp = {
            "test": "Phase 1a", "model": "GPT-2 XL",
            "a1_step_ms": w1, "a2_step_ms": w2, "step_ok": step_ok,
        }
        if s1 and s2:
            cmp.update({
                "a1_cube_util": s1.get("cube_eff_util_pct"),
                "a2_cube_util": s2.get("cube_eff_util_pct"),
                "a1_vec_util": s1.get("vec_eff_util_pct"),
                "a2_vec_util": s2.get("vec_eff_util_pct"),
            })
        with open(os.path.join(OUTPUT_DIR, "phase1a_comparison.json"), "w") as f:
            json.dump(cmp, f, indent=2)

    print("\n[Phase1a] DONE.", flush=True)


if __name__ == "__main__":
    main()
