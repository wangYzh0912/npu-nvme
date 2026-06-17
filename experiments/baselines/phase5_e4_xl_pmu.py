#!/usr/bin/env python3
"""
Phase 5 E4 v3: PMU Profiling — GPT-2 XL Core Utilization (msprof)
=====================================================================
Correctly measures Vector/Cube core utilization for GPT-2 XL with
and without batched delta detection using msprof CSV parsing.

Configurations:
  A: BASELINE — Pure GPT-2 XL training, no delta detection
  B: I3_BATCHED — GPT-2 XL + per-layer batched delta (24 layers)

Each configuration is profiled with msprof, then the op_summary CSV
is parsed to extract:
  - Core type counts (AI_CORE=Cuber, AI_VECTOR_CORE)
  - Vector ALU utilization (aiv_vec_ratio)
  - Cube MAC utilization (aic_mac_ratio)
  - Step time

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    msprof --output=/home/user7/npu-nvme/output/profiling_vec/E4 -- \
    python phase5_e4_xl_pmu.py --mode baseline'

  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    msprof --output=/home/user7/npu-nvme/output/profiling_vec/E4_I3 -- \
    python phase5_e4_xl_pmu.py --mode i3_batched'
"""
import os, sys, time, json, math, re, csv, argparse, glob as gb
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288; SMALL_THRESHOLD = 10000
PROFILER_BASE = os.path.join(REPO, "output", "profiling_vec", "E4")


def analyze_model(model):
    params = list(model.trainable_params())
    layer_map = {}
    for pi, p in enumerate(params):
        name = p.name
        m = re.search(r'backbone\.blocks\.(\d+)\.', name)
        if m:    lid = int(m.group(1))
        elif 'backbone.embedding' in name: lid = -2
        elif 'backbone.layernorm' in name:  lid = -1
        else:    lid = -3
        ne = int(p.size)
        layer_map.setdefault(lid, {})[pi] = (p, name, ne)
    return params, layer_map


# ═══════════════════════════════════════════════
# CSV Parser
# ═══════════════════════════════════════════════

def parse_profiler_csv(csv_path):
    """Parse op_summary CSV, return core utilization stats."""
    stats = {
        "total_ops": 0, "total_dur_us": 0.0,
        "aic_count": 0, "aic_dur_us": 0.0,    # Cube / AI_CORE
        "aiv_count": 0, "aiv_dur_us": 0.0,    # Vector / AI_VECTOR_CORE
        "aicpu_count": 0, "aicpu_dur_us": 0.0,
        "aic_mac_weighted_sum": 0.0,           # sum(mac_ratio * dur) for weighted avg
        "aiv_vec_weighted_sum": 0.0,
        "aic_scalar_weighted_sum": 0.0,
        "aiv_scalar_weighted_sum": 0.0,
        "aic_mac_active_us": 0.0,              # sum(mac_ratio * dur) / 100
        "aiv_vec_active_us": 0.0,
        "delta_ops": {},                       # per-type stats for Sub/Mul/ReduceSum/...
    }

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Build column index map
        col = {h.strip(): i for i, h in enumerate(header)}

        for row in reader:
            if len(row) < max(col.values()) + 1:
                continue
            try:
                task_type = row[col.get("Task Type", 7)].strip()
                op_type = row[col.get("OP Type", 5)].strip()
                op_name = row[col.get("Op Name", 4)].strip()
                dur_us = float(row[col.get("Task Duration(us)", 9)].strip())
            except (IndexError, ValueError, KeyError):
                continue

            stats["total_ops"] += 1
            stats["total_dur_us"] += dur_us

            # Core type counting
            if "AI_CORE" in task_type and "AI_VECTOR" not in task_type:
                stats["aic_count"] += 1; stats["aic_dur_us"] += dur_us
                try:
                    mac_r = float(row[col.get("aic_mac_ratio", 24)].strip())
                    scalar_r = float(row[col.get("aic_scalar_ratio", 26)].strip())
                    stats["aic_mac_weighted_sum"] += mac_r * dur_us
                    stats["aic_scalar_weighted_sum"] += scalar_r * dur_us
                    stats["aic_mac_active_us"] += mac_r / 100.0 * dur_us
                except: pass
            elif "AI_VECTOR" in task_type:
                stats["aiv_count"] += 1; stats["aiv_dur_us"] += dur_us
                try:
                    vec_r = float(row[col.get("aiv_vec_ratio", 37)].strip())
                    scalar_r = float(row[col.get("aiv_scalar_ratio", 39)].strip())
                    stats["aiv_vec_weighted_sum"] += vec_r * dur_us
                    stats["aiv_scalar_weighted_sum"] += scalar_r * dur_us
                    stats["aiv_vec_active_us"] += vec_r / 100.0 * dur_us
                except: pass
            elif "AICPU" in task_type:
                stats["aicpu_count"] += 1; stats["aicpu_dur_us"] += dur_us

            # Track delta-related ops specifically
            delta_types = {"Sub", "Mul", "ReduceSum", "Cast", "Reshape", "Concat",
                           "ZerosLike", "OnesLike", "Add", "Abs", "ReduceMax", "Div",
                           "Round", "ClipByValue"}
            if op_type in delta_types:
                if op_type not in stats["delta_ops"]:
                    stats["delta_ops"][op_type] = {"count": 0, "dur_us": 0.0, "core_type": task_type,
                                                    "names": set()}
                stats["delta_ops"][op_type]["count"] += 1
                stats["delta_ops"][op_type]["dur_us"] += dur_us
                stats["delta_ops"][op_type]["core_type"] = task_type
                stats["delta_ops"][op_type]["names"].add(op_name[:60])

    # Compute weighted averages
    if stats["aic_dur_us"] > 0:
        stats["aic_mac_util_pct"] = stats["aic_mac_weighted_sum"] / stats["aic_dur_us"]
        stats["aic_scalar_util_pct"] = stats["aic_scalar_weighted_sum"] / stats["aic_dur_us"]
    if stats["aiv_dur_us"] > 0:
        stats["aiv_vec_util_pct"] = stats["aiv_vec_weighted_sum"] / stats["aiv_dur_us"]
        stats["aiv_scalar_util_pct"] = stats["aiv_scalar_weighted_sum"] / stats["aiv_dur_us"]

    # Convert names set to list for JSON
    for ot in stats["delta_ops"]:
        stats["delta_ops"][ot]["names"] = list(stats["delta_ops"][ot]["names"])

    return stats


# ═══════════════════════════════════════════════
# Main: Profile one mode
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "i3_batched"], required=True,
                       help="baseline = pure training, i3_batched = + per-layer batched delta")
    parser.add_argument("--n-layers", type=int, default=24,
                       help="Number of XL layers for I3 delta (default 24)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Phase 5 E4: GPT-2 XL PMU Profiling — mode={args.mode}")
    print(f"{'='*70}")

    # ── 1. Build model ──
    print("\n[1] Building GPT-2 XL...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    params, layer_map = analyze_model(model)
    layer_ids = sorted([l for l in layer_map.keys() if l >= 0])
    total_elems = sum(sum(n for _, _, n in layer_map[l].values()) for l in layer_ids)
    print(f"  {len(layer_ids)} transformer layers, {total_elems/1e6:.0f}M elems ({total_elems*2/1e9:.2f}GB FP16)")

    # ── 2. Build GE cell ──
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    model2 = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(model2.trainable_params(), learning_rate=1e-5)
    _, lm2 = analyze_model(model2)

    if args.mode == "baseline":
        print("\n[2] BASELINE: pure training, no delta...")

        class BaselineCell(nn.Cell):
            def __init__(self):
                super().__init__(auto_prefix=False)
                self.net = model2; self.net.set_grad(); self.opt = opt
                self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            def construct(self, *inp):
                loss, grads = self.gf(*inp)
                return ops.Depend()(loss, self.opt(grads))

        CellClass = BaselineCell
        label = "BASELINE"

    else:
        print(f"\n[2] I3_BATCHED: per-layer batched delta, {args.n_layers} layers...")
        use_layers = layer_ids[:args.n_layers]

        # Build per-layer delta sub-cells
        delta_cells = []
        for lid in use_layers:
            plist, flist, nelem = [], [], 0
            for pi in sorted(lm2[lid].keys()):
                p, name, n = lm2[lid][pi]
                if n >= SMALL_THRESHOLD:
                    plist.append(p); flist.append(p.dtype != ms.float16); nelem += n
            if not plist: continue
            nb = math.ceil(nelem / BLOCK_SIZE); pl = nb * BLOCK_SIZE; pa = pl - nelem

            class LayerD(nn.Cell):
                def __init__(self):
                    super().__init__(auto_prefix=False)
                    self.bs = BLOCK_SIZE; self.nb = nb; self.pl = pl; self.pa = pa
                    self.dp = tuple(plist); self.df = tuple(flist); self.np = len(plist)
                def construct(self):
                    parts = []
                    for i in range(self.np):
                        p = self.dp[i]
                        pv = ops.Cast()(p, ms.float16) if self.df[i] else p
                        parts.append(ops.Reshape()(pv, (-1,)))
                    fd = parts[0] if len(parts) == 1 else ops.Concat()(tuple(parts))
                    if self.pa > 0:
                        padded = ops.pad(fd, (0, self.pa), mode='constant', value=0.0)
                    else:
                        padded = fd
                    blocks = ops.Reshape()(padded, (self.nb, self.bs))
                    zeros = ops.ZerosLike()(blocks)
                    deltas = ops.Sub()(blocks, zeros)
                    sq = ops.Mul()(deltas, deltas)
                    norms = ops.ReduceSum()(sq, 1)
                    return ops.ReduceSum()(ops.Cast()(norms, ms.float32))
            delta_cells.append(LayerD())

        print(f"    {len(delta_cells)} delta sub-cells, {sum(c.nb for c in delta_cells)} total blocks")

        class I3Cell(nn.Cell):
            def __init__(self):
                super().__init__(auto_prefix=False)
                self.net = model2; self.net.set_grad(); self.opt = opt
                self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
                self.dc = nn.CellList(delta_cells); self.nd = len(delta_cells)
            def construct(self, *inp):
                loss, grads = self.gf(*inp)
                acc = Tensor([0.0], dtype=ms.float32)
                for i in range(self.nd):
                    acc = ops.Add()(acc, self.dc[i]())
                loss = ops.Depend()(loss, acc)
                return ops.Depend()(loss, self.opt(grads))

        CellClass = I3Cell
        label = "I3_BATCHED"

    # ── 3. Measure (with profiler running) ──
    print(f"\n[3] Measuring {label} (sink_size=4, 12 steps)...")
    cell = CellClass()
    ms_model = ms.Model(cell)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(12)

    epoch_times = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append((time.perf_counter() - self.t0) * 1000)

    t0 = time.perf_counter()
    error = None
    try:
        ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=4)
    except Exception as e:
        error = str(e)[:500]
        print(f"    FAILED: {error}")

    dt = time.perf_counter() - t0
    compile_ms = epoch_times[0] if epoch_times else 0
    warm_epochs = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    avg_step = sum(warm_epochs) / (len(warm_epochs) * 4) if warm_epochs else 0

    print(f"    compile={compile_ms:.0f}ms  avg_step={avg_step:.1f}ms  total={dt:.1f}s")

    # ── 4. Parse profiler CSV if available ──
    pmu_data = None
    prof_dirs = gb.glob(os.path.join(PROFILER_BASE, "PROF_*")) + gb.glob(
        os.path.join(REPO, "output", "profiling_vec", "E4_I3", "PROF_*"))
    for pd in prof_dirs:
        csv_dir = os.path.join(pd, "mindstudio_profiler_output")
        if os.path.isdir(csv_dir):
            for csv_file in gb.glob(os.path.join(csv_dir, "op_summary_*.csv")):
                try:
                    pmu_data = parse_profiler_csv(csv_file)
                    print(f"\n[4] Parsed PMU from: {csv_file}")
                    break
                except Exception as e:
                    print(f"    Parse error: {e}")
            if pmu_data: break

    # ── 5. Summary report ──
    result = {
        "experiment": f"Phase 5 E4: {label} PMU (GPT-2 XL)",
        "mode": args.mode, "n_layers": args.n_layers if args.mode == "i3_batched" else 0,
        "model": "GPT-2 XL 48L/1600d",
        "timing": {"compile_ms": compile_ms, "avg_step_ms": avg_step,
                   "total_s": dt, "error": error},
        "pmu": pmu_data,
    }

    print(f"\n{'='*70}")
    print(f"E4 RESULT: {label}")
    print(f"{'='*70}")
    print(f"  Step time: {avg_step:.1f}ms")

    if pmu_data:
        print(f"\n  Core Distribution:")
        total_dur = pmu_data["total_dur_us"]
        aic_pct = pmu_data["aic_dur_us"] / total_dur * 100 if total_dur > 0 else 0
        aiv_pct = pmu_data["aiv_dur_us"] / total_dur * 100 if total_dur > 0 else 0
        aicpu_pct = pmu_data["aicpu_dur_us"] / total_dur * 100 if total_dur > 0 else 0
        print(f"    AI_CORE (Cube):  {pmu_data['aic_count']:5d} ops  {pmu_data['aic_dur_us']/1e6:.2f}s  ({aic_pct:.1f}%)")
        print(f"    AI_VECTOR:       {pmu_data['aiv_count']:5d} ops  {pmu_data['aiv_dur_us']/1e6:.2f}s  ({aiv_pct:.1f}%)")
        print(f"    AICPU:           {pmu_data['aicpu_count']:5d} ops  {pmu_data['aicpu_dur_us']/1e6:.2f}s  ({aicpu_pct:.1f}%)")

        print(f"\n  Core Utilization:")
        if "aic_mac_util_pct" in pmu_data:
            print(f"    Cube MAC util:    {pmu_data['aic_mac_util_pct']:.1f}%")
            print(f"    Cube Scalar util: {pmu_data['aic_scalar_util_pct']:.1f}%")
        if "aiv_vec_util_pct" in pmu_data:
            print(f"    Vector ALU util:  {pmu_data['aiv_vec_util_pct']:.1f}%")
            print(f"    Vector Scalar:    {pmu_data['aiv_scalar_util_pct']:.1f}%")

        print(f"\n  Delta-related Ops Core Attribution:")
        for ot in sorted(pmu_data["delta_ops"].keys()):
            d = pmu_data["delta_ops"][ot]
            print(f"    {ot:15s}: {d['count']:5d} ops  dur={d['dur_us']/1e3:.2f}ms  core={d['core_type']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"phase5_e4_{args.mode}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  → Saved: {out_file}")
    print(f"[E4 {label}] DONE.")


if __name__ == "__main__":
    main()
