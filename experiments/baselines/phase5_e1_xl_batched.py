#!/usr/bin/env python3
"""
Phase 5 E1 v2: GPT-2 XL Batched GE Compile + Step Time
==========================================================
Fixed version: uses per-layer sub-Cells (no Python loops inside construct())
and incremental layer testing (12/24/48) to isolate scaling behavior.

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_e1_xl_batched.py'
"""
import os, sys, time, json, math, re
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288
SMALL_THRESHOLD = 10000


# ═══════════════════════════════════════════════
# Per-layer batched delta sub-cell
# Each instance handles ONE layer's large params.
# This keeps construct() clean: no loops, no conditionals.
# ═══════════════════════════════════════════════

class LayerDeltaCell(nn.Cell):
    """Batched delta detection for ONE layer's large params."""
    def __init__(self, plist, fp16_flags, total_elems, block_size):
        super().__init__(auto_prefix=False)
        self.layer_params = tuple(plist)  # ParameterTuple
        self.fp16_flags = tuple(fp16_flags)
        self.total_elems = total_elems
        self.block_size = block_size
        self.num_blocks = math.ceil(total_elems / block_size)
        self.padded_len = self.num_blocks * block_size
        self.pad_amt = self.padded_len - total_elems

        # Pre-register all params as attributes so GE sees them correctly
        for i, p in enumerate(plist):
            setattr(self, f'p{i}', p)

    def construct(self):
        """Returns scalar norm sum. No inputs — reads from self.layer_params."""
        acc = Tensor([0.0], dtype=ms.float32)

        # Concat all params in this layer
        parts = []
        for i in range(len(self.layer_params)):
            p = self.layer_params[i]
            pv = ops.Cast()(p, ms.float16) if self.fp16_flags[i] else p
            parts.append(ops.Reshape()(pv, (-1,)))
        fd = parts[0] if len(parts) == 1 else ops.Concat()(tuple(parts))

        # Pad + Reshape to [N_blocks, block_size]
        if self.pad_amt > 0:
            padded = ops.pad(fd, (0, self.pad_amt), mode='constant', value=0.0)
        else:
            padded = fd

        blocks = ops.Reshape()(padded, (self.num_blocks, self.block_size))
        zeros = ops.ZerosLike()(blocks)
        deltas = ops.Sub()(blocks, zeros)
        norms = ops.ReduceSum()(ops.Mul()(deltas, deltas), 1)
        return ops.ReduceSum()(ops.Cast()(norms, ms.float32))


# ═══════════════════════════════════════════════
# Model Analysis
# ═══════════════════════════════════════════════

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


def build_delta_cells(layer_map, block_size, max_layers=None):
    """Build LayerDeltaCell instances for each transformer layer."""
    layer_ids = sorted([l for l in layer_map.keys() if l >= 0])
    if max_layers: layer_ids = layer_ids[:max_layers]

    cells_info = []
    all_cells = []
    for lid in layer_ids:
        plist, flist, nelem = [], [], 0
        for pi in sorted(layer_map[lid].keys()):
            p, name, n = layer_map[lid][pi]
            if n >= SMALL_THRESHOLD:
                plist.append(p); flist.append(p.dtype != ms.float16); nelem += n
        if plist:
            cell = LayerDeltaCell(plist, flist, nelem, block_size)
            all_cells.append(cell)
            cells_info.append({"lid": lid, "n_params": len(plist),
                               "n_elems": nelem, "n_blocks": cell.num_blocks})
    return all_cells, cells_info


# ═══════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════

def measure(label, build_fn, n_steps=16, n_warm=4, sink_size=4):
    """Build and measure. Returns avg_step_ms or error."""
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    print(f"  [{label}] Building...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        cell = build_fn()
        ms_model = ms.Model(cell)
        ds = ms.dataset.MindDataset(
            REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
        ds = ds.batch(1, drop_remainder=True).take(n_steps)

        epoch_times_ms = []
        class CB(ms.Callback):
            def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
            def on_train_epoch_end(self, rc):
                epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

        epochs = max(1, n_steps // sink_size)
        ms_model.train(epoch=epochs, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=sink_size)

        compile_ms = epoch_times_ms[0] if epoch_times_ms else 0
        warm_epochs = epoch_times_ms[1:]
        if warm_epochs:
            avg_step = sum(warm_epochs) / (len(warm_epochs) * sink_size)
        else:
            avg_step = compile_ms / sink_size

        dt = time.perf_counter() - t0
        print(f"compile={compile_ms:.0f}ms  avg_step={avg_step:.1f}ms  total={dt:.1f}s")
        return {"ok": True, "compile_ms": compile_ms, "avg_step_ms": avg_step,
                "epoch_times": epoch_times_ms, "total_s": dt}
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"FAIL ({dt:.1f}s): {str(e)[:300]}")
        return {"ok": False, "error": str(e)[:500], "total_s": dt}


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 5 E1 v2: GPT-2 XL Batched Delta — Layer Scaling")
    print("=" * 70)

    # ── 1. PYNATIVE: analyze model ──
    print("\n[1] Analyzing GPT-2 XL...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    # Don't load checkpoint — use random init (much faster)
    params, layer_map = analyze_model(model)

    all_layer_ids = sorted([l for l in layer_map.keys() if l >= 0])
    total_elems = sum(sum(n for _, _, n in layer_map[l].values()) for l in all_layer_ids)
    total_large = sum(sum(1 for _, _, n in layer_map[l].values() if n >= SMALL_THRESHOLD) for l in all_layer_ids)

    print(f"  {len(all_layer_ids)} layers, {total_elems/1e6:.0f}M elems ({total_elems*2/1e9:.2f}GB FP16)")
    print(f"  {total_large} large params (≥10K), {len(params)-total_large} small params")

    # ── 2. Build Baseline ──
    print("\n[2] Building BASELINE...")

    def build_baseline():
        ms.common.set_seed(42)
        from mindformers import AutoModel as AM2, AutoConfig as AC2
        cfg2 = AC2.from_pretrained("gpt2_xl"); cfg2.seq_length=SEQ_LEN; cfg2.max_position_embeddings=SEQ_LEN
        m2 = AM2.from_config(cfg2)
        opt2 = nn.AdamWeightDecay(m2.trainable_params(), learning_rate=1e-5)
        class BC(nn.Cell):
            def __init__(self):
                super().__init__(auto_prefix=False)
                self.net = m2; self.net.set_grad(); self.opt = opt2
                self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            def construct(self, *inp):
                loss, grads = self.gf(*inp)
                return ops.Depend()(loss, self.opt(grads))
        return BC()

    r_base = measure("BASELINE", build_baseline, n_steps=16, sink_size=4)

    if not r_base["ok"]:
        print("  ❌ Baseline failed!")
        return

    baseline_ms = r_base["avg_step_ms"]
    print(f"\n  Baseline: {baseline_ms:.1f}ms/step")

    # ── 3. Test I3 with 12, 24, 48 layers ──
    results = {"experiment": "Phase 5 E1 v2: XL Layer Scaling",
               "model": "GPT-2 XL 48L",
               "block_size": BLOCK_SIZE,
               "baseline_ms": baseline_ms,
               "layer_tests": []}

    for n_layers in [12, 24, 48]:
        print(f"\n[3.{n_layers}] I3 with {n_layers} layers...")

        delta_cells, cells_info = build_delta_cells(layer_map, BLOCK_SIZE, max_layers=n_layers)
        print(f"    {len(delta_cells)} delta sub-cells, {sum(c['n_blocks'] for c in cells_info)} total blocks")

        def build_i3(n_layers=n_layers):
            # Force reuse of the module-level `model` reference via capturing
            # Build fresh model
            ms.common.set_seed(42)
            from mindformers import AutoModel as AM3, AutoConfig as AC3
            cfg3 = AC3.from_pretrained("gpt2_xl"); cfg3.seq_length=SEQ_LEN; cfg3.max_position_embeddings=SEQ_LEN
            m3 = AM3.from_config(cfg3)
            opt3 = nn.AdamWeightDecay(m3.trainable_params(), learning_rate=1e-5)

            # Build delta cells for this model's layers
            # Need to re-analyze because params are new objects after from_config
            _, lm3 = analyze_model(m3)
            delta_cells3, _ = build_delta_cells(lm3, BLOCK_SIZE, max_layers=n_layers)

            class I3XL(nn.Cell):
                def __init__(self):
                    super().__init__(auto_prefix=False)
                    self.net = m3; self.net.set_grad(); self.opt = opt3
                    self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
                    self.dcells = nn.CellList(delta_cells3)
                    self.n_d = len(delta_cells3)
                def construct(self, *inp):
                    loss, grads = self.gf(*inp)
                    # Call each layer's delta detection
                    acc = Tensor([0.0], dtype=ms.float32)
                    for i in range(self.n_d):
                        acc = ops.Add()(acc, self.dcells[i]())
                    loss = ops.Depend()(loss, acc)
                    return ops.Depend()(loss, self.opt(grads))
            return I3XL()

        n_steps = 8 if n_layers <= 24 else 8
        r_i3 = measure(f"I3-{n_layers}L", build_i3, n_steps=n_steps, sink_size=4)

        layer_result = {
            "n_layers": n_layers,
            "delta_cells": len(delta_cells),
            "total_blocks": sum(c['n_blocks'] for c in cells_info),
        }
        if r_i3["ok"]:
            layer_result["ok"] = True
            layer_result["avg_step_ms"] = r_i3["avg_step_ms"]
            layer_result["compile_ms"] = r_i3["compile_ms"]
            layer_result["overhead_ms"] = r_i3["avg_step_ms"] - baseline_ms
            layer_result["overhead_pct"] = (r_i3["avg_step_ms"] - baseline_ms) / baseline_ms * 100
            print(f"    Step: {r_i3['avg_step_ms']:.1f}ms  Overhead: {layer_result['overhead_ms']:+.1f}ms ({layer_result['overhead_pct']:+.1f}%)")
        else:
            layer_result["ok"] = False
            layer_result["error"] = r_i3.get("error", "unknown")[:300]
            print(f"    FAILED")

        results["layer_tests"].append(layer_result)

    # ── 4. Final Report ──
    print("\n" + "=" * 70)
    print("E1 FINAL RESULTS: GPT-2 XL Layer Scaling")
    print("=" * 70)
    print(f"  Baseline: {baseline_ms:.1f}ms/step")
    print(f"  {'Layers':<10} {'Step(ms)':<12} {'Overhead(ms)':<15} {'Overhead%'}")
    print(f"  {'-'*50}")
    for lt in results["layer_tests"]:
        if lt["ok"]:
            print(f"  {lt['n_layers']:<10} {lt['avg_step_ms']:<12.1f} {lt['overhead_ms']:<+15.1f} {lt['overhead_pct']:+.1f}%")
        else:
            print(f"  {lt['n_layers']:<10} FAILED")
    print(f"  {'-'*50}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_e1_xl_scaling.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Saved: {out}")
    print("[E1 DONE]")


if __name__ == "__main__":
    main()
