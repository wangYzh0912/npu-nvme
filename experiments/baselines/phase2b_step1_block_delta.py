#!/usr/bin/env python3
"""
Phase 2b Step 1: Block Aggregation + Delta Detection Prototype
==============================================================

Tests fixed-size block delta detection on GPT-2 Small.
Validates the core architecture from PHASE2B_DESIGN.md:

  1. Group params by layer
  2. For selected layer: Cast→FP16, flatten, concat into one flat buffer
  3. Split flat buffer into fixed-size blocks (512K elements each)
  4. For each block: ||block - P_old_block||^2 (P_old = zeros for stub)
  5. Verify GE graph compiles (node count << 1000 limit)

GPT-2 Small per-layer: ~7.1M elements → ~14 blocks of 512K → ~70 GE ops
→ Very safe under ~1000 node limit ✅

Usage:
  # PYNATIVE correctness test
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    /home/user7/miniconda3/envs/ms_2.5/bin/python phase2b_step1_block_delta.py --mode pynative'

  # GRAPH_MODE compilation + timing test
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    /home/user7/miniconda3/envs/ms_2.5/bin/python phase2b_step1_block_delta.py --mode graph'
"""
import os, sys, time, json, math, re, argparse

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter

ms.set_recursion_limit(10000)

DEVICE_ID = 1
SEQ_LEN = 1024
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


def inspect_model_layers(model):
    """Extract per-layer parameter info from model."""
    params = list(model.trainable_params())

    layer_map = {}  # layer_id → [(param_idx, param, name, num_elems), ...]
    layer_elems = {}

    for pi, p in enumerate(params):
        name = p.name
        m = re.search(r'backbone\.blocks\.(\d+)\.', name)
        if m:
            layer_id = int(m.group(1))
        elif 'backbone.embedding' in name:
            layer_id = -2  # embedding (word + position)
        elif 'backbone.layernorm' in name:
            layer_id = -1  # final layernorm
        else:
            layer_id = -3  # unknown

        ne = int(p.size)
        if layer_id not in layer_map:
            layer_map[layer_id] = []
            layer_elems[layer_id] = 0
        layer_map[layer_id].append((pi, p, name, ne))
        layer_elems[layer_id] += ne

    return params, layer_map, layer_elems


def print_layer_summary(layer_map, layer_elems):
    """Print per-layer parameter summary."""
    print(f"\n  {'Layer':>6}  {'Params':>8}  {'Elements':>14}  {'MB FP16':>9}  {'Blocks@512K':>12}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*9}  {'-'*12}")
    for lid in sorted(layer_map.keys()):
        n_params = len(layer_map[lid])
        ne = layer_elems[lid]
        n_blocks = math.ceil(ne / 524288)
        print(f"  {lid:>6}  {n_params:>8}  {ne:>14,}  {ne*2/1e6:>8.1f}  {n_blocks:>12}")


def run_pynative_verify(model, params, layer_map, layer_elems, block_size, selected_layer):
    """PYNATIVE mode: verify delta norm computation correctness."""
    print(f"\n{'='*60}")
    print(f"  PYNATIVE Verification — Layer {selected_layer}")
    print(f"{'='*60}")

    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)

    # Get params for selected layer
    layer_info = layer_map[selected_layer]
    layer_params = [info[1] for info in layer_info]
    total_elems = layer_elems[selected_layer]
    num_blocks = math.ceil(total_elems / block_size)

    print(f"  Layer {selected_layer}: {len(layer_params)} params, {total_elems:,} elems, {num_blocks} blocks")

    # Step 1: Flatten + concat all params in this layer
    flat_parts = []
    for p in layer_params:
        p_fp16 = ops.Cast()(p, ms.float16) if p.dtype != ms.float16 else p
        flat_parts.append(ops.Reshape()(p_fp16, (-1,)))

    if len(flat_parts) == 1:
        full_flat = flat_parts[0]
    else:
        full_flat = ops.Concat()(tuple(flat_parts))

    flat_np = full_flat.asnumpy().astype(np.float32)  # Use FP32 to avoid overflow
    print(f"  Full flat tensor: {flat_np.shape}, dtype=float16→float32")
    print(f"  Range: [{flat_np.min():.4f}, {flat_np.max():.4f}], L2 norm²={np.sum(flat_np**2):.4f}")

    # Step 2: Split into blocks and compute per-block delta norms
    block_norms = []
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, total_elems)
        block = full_flat[start:end]

        # P_old = zeros (stub — in real impl, this is FP8 quantized previous value)
        p_old = ops.ZerosLike()(block)
        delta = ops.Sub()(block, p_old)
        delta_sq = ops.Mul()(delta, delta)
        norm = ops.ReduceSum()(delta_sq)
        # Cast to float32 before numpy conversion for precision
        norm_f32 = ops.Cast()(norm, ms.float32)
        norm_val = float(norm_f32.asnumpy())
        block_norms.append(norm_val)

    print(f"\n  Per-block delta norms (P_old=zeros):")
    for b, n in enumerate(block_norms):
        b_start = b * block_size
        b_end = min(b_start + block_size, total_elems)
        b_elems = b_end - b_start
        print(f"    Block {b:3d}: [{b_start:>10,}:{b_end:>10,}] ({b_elems:>9,} elems) → norm²={n:.4f}")

    total_norm_sq = sum(float(n) for n in block_norms)
    expected_norm_sq = float(np.sum(flat_np.astype(np.float64) ** 2))
    rel_err = abs(total_norm_sq - expected_norm_sq) / max(expected_norm_sq, 1e-10)

    print(f"\n  Verification:")
    print(f"    Sum of block norms:           {total_norm_sq:.6f}")
    print(f"    Direct ||W||² on flat tensor: {expected_norm_sq:.6f}")
    print(f"    Relative error:                {rel_err:.2e}")

    ok = rel_err < 1e-3  # FP16 ReduceSum on 512K elems has ~0.01% error
    print(f"    {'✅ PASS' if ok else '❌ FAIL'} (threshold: relative error < 1e-3)")

    # Also compute per-param norms for comparison (float64 to avoid overflow)
    print(f"\n  Per-param L2 norms (for reference):")
    offset = 0
    param_norms = []
    for info in layer_info:
        pi, p, name, ne = info
        val = flat_np[offset:offset+ne]
        p_norm = float(np.sum(val.astype(np.float64)**2))
        param_norms.append((name, ne, p_norm))
        offset += ne
    # Sort by norm descending
    param_norms.sort(key=lambda x: -x[2])
    for name, ne, p_norm in param_norms[:5]:
        print(f"    {name:55s} {ne:>10,} elems → norm²={p_norm:.4f}")
    if len(param_norms) > 5:
        print(f"    ... and {len(param_norms)-5} more")

    return {
        "layer": selected_layer,
        "num_params": len(layer_params),
        "total_elems": total_elems,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "block_norms": block_norms,
        "total_norm_sq": total_norm_sq,
        "expected_norm_sq": expected_norm_sq,
        "rel_error": rel_err,
        "verified": ok,
    }


def run_graph_test(block_size, selected_layer, sink_size, total_steps):
    """GRAPH_MODE: build model + Cell in GRAPH_MODE context. Test GE compilation and measure step time."""
    print(f"\n{'='*60}")
    print(f"  GRAPH_MODE Test — Layer {selected_layer}")
    print(f"{'='*60}")

    # Build EVERYTHING inside GRAPH_MODE context (Phase 1a pattern)
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    # Inspect layers
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_info = layer_map[selected_layer]
    layer_params = [info[1] for info in layer_info]
    total_elems = layer_elems[selected_layer]
    num_blocks = math.ceil(total_elems / block_size)
    n_params_in_layer = len(layer_params)

    print(f"  Layer {selected_layer}: {n_params_in_layer} params, {total_elems:,} elems, {num_blocks} blocks")

    # Create optimizer
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    # Pre-compute per-param dtype info
    fp16_cast_needed = [p.dtype != ms.float16 for p in layer_params]
    n_cast = sum(fp16_cast_needed)

    # GE ops estimate
    # Cast: n_cast ops
    # Reshape: n_params_in_layer ops
    # Concat: 1 op
    # Per block: ZerosLike + Sub + Mul + ReduceSum + Add(to acc)
    n_agg_ops = n_cast + n_params_in_layer + 1
    n_delta_ops = num_blocks * 5  # ZerosLike + Sub + Mul + ReduceSum + Add
    n_cast_acc = 1  # Cast ReduceSum result to float32 for accumulation
    ge_ops_est = n_agg_ops + n_delta_ops + n_cast_acc
    print(f"  Estimated GE ops: {n_agg_ops} aggreg + {n_delta_ops} delta + {n_cast_acc} cast = {ge_ops_est}")
    print(f"  Safe margin: {1000 - ge_ops_est} nodes under 1000 limit {'✅' if ge_ops_est < 800 else '⚠️'}")

    # Mirror Phase 1a pattern: params wrapped in list-of-lists (self.pg)
    # Use plain list (NOT ParameterTuple) to avoid GE duplicate registration
    param_groups = [layer_params]
    fp16_needed = [fp16_cast_needed]

    class BlockDeltaCell(nn.Cell):
        def __init__(self, network, opt, pg, fn, n_blocks, bs):
            super().__init__(auto_prefix=False)
            self.net = network
            self.net.set_grad()
            self.opt = opt
            self.grad_fn = ops.value_and_grad(self.net, grad_position=None,
                                               weights=self.opt.parameters)
            self.depend = ops.Depend()
            self.pg = pg        # list of lists of Parameters (Phase 1a pattern)
            self.fn = fn        # list of lists of bools
            self.n_blocks = n_blocks
            self.bs = bs

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)

            # ── Block aggregation (Phase 1a per-group pattern) ──
            acc_norm = Tensor([0.0], dtype=ms.float32)
            for gi, group in enumerate(self.pg):
                flat_parts = []
                flags = self.fn[gi]
                for pi, p in enumerate(group):
                    pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                    flat_parts.append(ops.Reshape()(pv, (-1,)))

                if len(flat_parts) == 1:
                    full_flat = flat_parts[0]
                else:
                    full_flat = ops.Concat()(tuple(flat_parts))

                # ── Per-block delta detection ──
                for b in range(self.n_blocks):
                    start_idx = b * self.bs
                    block = full_flat[start_idx:start_idx + self.bs]
                    p_old = ops.ZerosLike()(block)
                    delta = ops.Sub()(block, p_old)
                    delta_sq = ops.Mul()(delta, delta)
                    norm_val = ops.ReduceSum()(delta_sq)
                    acc_norm = ops.Add()(acc_norm, ops.Cast()(norm_val, ms.float32))

            loss = self.depend(loss, acc_norm)
            opt_res = self.opt(grads)
            return self.depend(loss, opt_res)

    t_build = time.perf_counter()
    cell = BlockDeltaCell(model, optimizer, param_groups, fp16_needed, num_blocks, block_size)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print(f"  Build time: {build_s:.1f}s")

    # Dataset
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    epochs = total_steps // sink_size

    epoch_times_ms = []
    class EpochCB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  Training {total_steps} steps (sink={sink_size})...", flush=True)
    compiled_ok = True
    error_msg = None
    t_total = time.perf_counter()

    try:
        ms_model.train(epoch=epochs, train_dataset=ds, callbacks=[EpochCB()],
                       dataset_sink_mode=True, sink_size=sink_size)
    except Exception as e:
        compiled_ok = False
        error_msg = str(e)[:400]
        print(f"  ❌ FAILED: {error_msg}", flush=True)

    total_s = time.perf_counter() - t_total

    if compiled_ok and epoch_times_ms:
        compile_epoch = epoch_times_ms[0]
        warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
        avg_step = sum(warm_epochs) / len(warm_epochs) / sink_size if warm_epochs else 0

        print(f"  compile_epoch={compile_epoch:.0f}ms  "
              f"warm_epochs={[f'{e:.0f}ms' for e in warm_epochs]}  "
              f"avg_step={avg_step:.0f}ms", flush=True)
    else:
        compile_epoch = 0
        warm_epochs = []
        avg_step = 0

    return {
        "layer": selected_layer,
        "num_params_in_layer": n_params_in_layer,
        "total_elems": total_elems,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "ge_ops_estimated": ge_ops_est,
        "compiled_ok": compiled_ok,
        "error": error_msg,
        "build_s": round(build_s, 1),
        "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(e, 0) for e in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2b Step 1: Block Delta Detection")
    parser.add_argument("--mode", default="pynative", choices=["pynative", "graph", "both"],
                       help="Execution mode")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer to test (-2=emb, -1=final_ln, 0-11=transformer)")
    parser.add_argument("--block_size", type=int, default=524288,
                       help="Elements per block (default 512K = 1MB FP16)")
    parser.add_argument("--steps", type=int, default=8, help="Training steps")
    parser.add_argument("--sink", type=int, default=4, help="Sink size")
    parser.add_argument("--label", default="phase2b_s1")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2b Step 1: Fixed-Size Block Delta Detection")
    print(f"  Mode={args.mode}  Layer={args.layer}  BlockSize={args.block_size:,}")
    print("=" * 70, flush=True)

    # ── Build model ──
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)
    print("  Model: GPT-2 Small built OK")

    # ── Inspect layers ──
    params, layer_map, layer_elems = inspect_model_layers(model)
    print(f"  Total trainable params: {len(params)}")
    print_layer_summary(layer_map, layer_elems)

    # Validate layer selection
    if args.layer not in layer_map:
        print(f"\n  ⚠ Layer {args.layer} not found! Available: {sorted(layer_map.keys())}")
        sys.exit(1)

    results = {}

    # ── PYNATIVE verification ──
    if args.mode in ("pynative", "both"):
        results["pynative"] = run_pynative_verify(
            model, params, layer_map, layer_elems, args.block_size, args.layer)

    # ── GRAPH_MODE compilation test ──
    if args.mode in ("graph", "both"):
        results["graph"] = run_graph_test(
            args.block_size, args.layer, args.sink, args.steps)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  Phase 2b Step 1 — Results Summary")
    print(f"{'='*70}")

    if "pynative" in results:
        r = results["pynative"]
        status = "✅ PASS" if r["verified"] else "❌ FAIL"
        print(f"  PYNATIVE correctness:  {status} (rel_err={r['rel_error']:.2e})")
        print(f"    Layer {r['layer']}: {r['num_params']} params → {r['total_elems']:,} elems → {r['num_blocks']} blocks")

    if "graph" in results:
        r = results["graph"]
        if r["compiled_ok"]:
            print(f"  GRAPH_MODE:            ✅ Compiled OK ({r['build_s']:.1f}s)")
            print(f"    GE ops est: {r['ge_ops_estimated']} (limit ~1000)")
            print(f"    avg_step: {r['avg_step_ms']}ms")
        else:
            print(f"  GRAPH_MODE:            ❌ FAILED")
            print(f"    Error: {r['error'][:200] if r['error'] else 'N/A'}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"phase2b_s1_{args.label}.json")
    with open(out_json, "w") as f:
        json.dump({
            "test": "Phase 2b Step 1: Block Delta Detection",
            "layer": args.layer,
            "block_size": args.block_size,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results → {os.path.basename(out_json)}")
    print("[Phase2b_Step1] DONE.", flush=True)


if __name__ == "__main__":
    main()
