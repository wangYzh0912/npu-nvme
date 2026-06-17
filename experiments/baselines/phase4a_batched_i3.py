#!/usr/bin/env python3
"""
Phase 4a: Batched I3 Pipeline + v7 Recovery
============================================

Integrates batched block delta ops into the full I3 pipeline:
  - Per-param block partitioning (v6 approach: no cross-param contamination)
  - Batched GE ops (Reshape(flat, (N, BS)) → single GE invocation per layer)
  - Small param protection (<10K elems always saved)
  - Rotation controller (M=10)
  - INT8 P_old + top-K selection (host-side)
  - Recovery fidelity measurement (NRMSE + loss comparison)

This is the FINAL I3 implementation combining all Phase 3 discoveries.

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase4a_batched_i3.py --steps 20 --M 10 --top_k 0.10'
"""
import os, sys, time, json, math, re, argparse, copy

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288; SMALL_THRESHOLD = 10000


# ═══════════════════════════════════════════════════════════════════
# Host-side utilities (identical to v6)
# ═══════════════════════════════════════════════════════════════════

def inspect_model_layers(model):
    params = list(model.trainable_params())
    layer_map, layer_elems = {}, {}
    for pi, p in enumerate(params):
        name = p.name
        m = re.search(r'backbone\.blocks\.(\d+)\.', name)
        if m:    lid = int(m.group(1))
        elif 'backbone.embedding' in name: lid = -2
        elif 'backbone.layernorm' in name:  lid = -1
        else:    lid = -3
        ne = int(p.size)
        if lid not in layer_map: layer_map[lid] = {}; layer_elems[lid] = 0
        layer_map[lid][pi] = (p, name, ne)
        layer_elems[lid] += ne
    return params, layer_map, layer_elems


def get_all_params_np(model):
    return {p.name: p.value().asnumpy().copy() for p in model.trainable_params()}


def classify_layer_params(layer_info):
    large, small = {}, {}
    for pi in sorted(layer_info.keys()):
        p, name, ne = layer_info[pi]
        (small if ne < SMALL_THRESHOLD else large)[pi] = (p, name, ne)
    return large, small


class PoldStore:
    def __init__(self):
        self.i8, self.sc = {}, {}
    def get(self, lid, name, bidx, ref):
        k = f'{lid}:{name}:{bidx}'
        if k not in self.i8: return np.zeros_like(ref, dtype=np.float32)
        return self.i8[k].astype(np.float32) * self.sc[k]
    def put(self, lid, name, bidx, data):
        fp32 = data.astype(np.float32).flatten()
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        k = f'{lid}:{name}:{bidx}'
        self.i8[k] = np.clip(np.round(fp32 / s), -128, 127).astype(np.int8)
        self.sc[k] = float(s)


class RotCtrl:
    def __init__(self, ids, M=10):
        self.ids, self.M = sorted(ids), M
        self.ss = {l: 0 for l in ids}
    def select(self):
        for l in self.ids: self.ss[l] += 1
        stale = [l for l in self.ids if self.ss[l] >= self.M]
        if not stale:
            m = max(self.ss.values())
            stale = [[l for l in self.ids if self.ss[l] == m][0]]
        for l in stale: self.ss[l] = 0
        return stale


def compute_nrmse(w_r, w_t):
    all_n, params = [], {}
    for nm in w_t:
        r = w_r[nm].astype(np.float64).flatten()
        t = w_t[nm].astype(np.float64).flatten()
        d = r - t; std = float(np.std(t)) + 1e-12
        nr = float(np.sqrt(np.mean(d**2))) / std
        params[nm] = {"nrmse": nr, "mae": float(np.mean(np.abs(d))),
                       "std": float(std), "max_abs": float(np.max(np.abs(d)))}
        all_n.append(nr)
    return {"params": params, "mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "p95": float(np.percentile(all_n, 95)), "p99": float(np.percentile(all_n, 99)),
            "max": float(np.max(all_n))}


def reconstruct_v6(init_w, layer_map, block_patches, small_patches, block_size):
    w = copy.deepcopy(init_w)
    for p in block_patches:
        lid, name, bidx, i8, s = p["layer_id"], p["name"], p["block_idx"], p["int8_data"], p["scale"]
        fp32 = i8.astype(np.float32) * s
        start = bidx * block_size; end = min(start + len(fp32), int(np.prod(w[name].shape)))
        wv = w[name].astype(np.float32).flatten()
        wv[start:end] = fp32[:end-start]
        w[name] = wv.reshape(w[name].shape)
    for p in small_patches:
        lid, name, i8, s = p["layer_id"], p["name"], p["int8_data"], p["scale"]
        w[name] = (i8.astype(np.float32) * s).flatten()[:int(np.prod(w[name].shape))].reshape(w[name].shape)
    return w


# ═══════════════════════════════════════════════════════════════════
# Batched GE Cell
# ═══════════════════════════════════════════════════════════════════

def build_batched_cell(model, optimizer, layer_map, block_size, num_layers):
    """Build a GRAPH_MODE cell with BATCHED block delta detection for N layers.

    Returns: (cell, metadata_dict)
    """
    selected = sorted(layer_map.keys())[:num_layers]

    # Per-layer: list of large params + small params
    param_groups = []         # [(layer_id, [param_idx, ...])]
    fp16_masks = []           # per-param FP16 cast needed
    per_param_blocks = []     # per-param num_blocks at block_size
    per_param_names = []      # per-param names
    per_param_sizes = []      # per-param element counts

    for lid in selected:
        large, _ = classify_layer_params(layer_map[lid])
        lid_params = []; lid_fp16 = []; lid_blocks = []; lid_names = []; lid_sizes = []
        for pi in sorted(large.keys()):
            p, name, ne = large[pi]
            lid_params.append(p)
            lid_fp16.append(p.dtype != ms.float16)
            lid_blocks.append(math.ceil(ne / block_size))
            lid_names.append(name)
            lid_sizes.append(ne)
        param_groups.append((lid, lid_params))
        fp16_masks.append((lid, lid_fp16))
        per_param_blocks.append((lid, lid_blocks))
        per_param_names.append((lid, lid_names))
        per_param_sizes.append((lid, lid_sizes))

    # Flatten for cell storage (nn.Cell doesn't like nested tuples in some MS versions)
    # Store as flat lists with group boundaries
    group_starts = [0]
    for (_, params_list) in param_groups:
        group_starts.append(group_starts[-1] + len(params_list))

    all_params_flat = []
    all_fp16_flat = []
    all_blocks_flat = []
    all_names_flat = []
    all_sizes_flat = []
    for i in range(len(param_groups)):
        _, plist = param_groups[i]
        _, fplist = fp16_masks[i]
        _, blist = per_param_blocks[i]
        _, nlist = per_param_names[i]
        _, slist = per_param_sizes[i]
        all_params_flat.extend(plist)
        all_fp16_flat.extend(fplist)
        all_blocks_flat.extend(blist)
        all_names_flat.extend(nlist)
        all_sizes_flat.extend(slist)

    total_params = len(all_params_flat)
    total_blocks = sum(all_blocks_flat)
    num_groups = len(param_groups)
    group_ends = group_starts[1:]

    class BatchedI3Cell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = optimizer
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self._params = all_params_flat
            self._fp16 = all_fp16_flat
            self._blks = all_blocks_flat
            self._sizes = all_sizes_flat
            self._n_groups = num_groups
            self._g_ends = group_ends
            self._bs = block_size

        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            acc = Tensor([0.0], dtype=ms.float32)

            pi = 0  # global param index
            for gi in range(self._n_groups):
                g_end = self._g_ends[gi]
                # Flatten + concat all large params in this layer
                flat_parts = []
                while pi < g_end:
                    p = self._params[pi]
                    pv = ops.Cast()(p, ms.float16) if self._fp16[pi] else p
                    flat_parts.append(ops.Reshape()(pv, (-1,)))
                    pi += 1
                fd = flat_parts[0] if len(flat_parts) == 1 else ops.Concat()(tuple(flat_parts))

                # BATCHED: split into [N_blocks_total, block_size]
                pi_g = pi - (g_end - (gi-1 if gi > 0 else 0))  # ... simpler: per-param approach
                # Actually do per-param batched since each param has its own block count

            # Alternative: per-param batched (each param independently)
            pi = 0
            for gi in range(self._n_groups):
                g_end = self._g_ends[gi]
                while pi < g_end:
                    p = self._params[pi]
                    pv = ops.Cast()(p, ms.float16) if self._fp16[pi] else p
                    flat_p = ops.Reshape()(pv, (-1,))
                    nb = self._blks[pi]
                    padded_len = nb * self._bs
                    pad_amt = padded_len - self._sizes[pi]
                    if pad_amt > 0:
                        padded = ops.pad(flat_p, (0, pad_amt), mode='constant', value=0.0)
                    else:
                        padded = flat_p
                    # BATCHED per-param: [nb, block_size]
                    blocks = ops.Reshape()(padded, (nb, self._bs))
                    zeros = ops.ZerosLike()(blocks)
                    deltas = ops.Sub()(blocks, zeros)
                    norms = ops.ReduceSum()(ops.Mul()(deltas, deltas), 1)
                    layer_sum = ops.ReduceSum()(ops.Cast()(norms, ms.float32))
                    acc = ops.Add()(acc, layer_sum)
                    pi += 1

            loss = ops.Depend()(loss, acc)
            return ops.Depend()(loss, self.opt(grads))

    n_large_params = total_params
    n_blocks = total_blocks
    est_ops = n_large_params * 6 + num_groups * 2  # pad+reshape+sub+mul+reducesum×2 per param + concat per group

    return BatchedI3Cell, {
        "num_layers": num_layers,
        "num_large_params": n_large_params,
        "num_blocks": n_blocks,
        "est_ops": est_ops,
        "selected_layers": selected,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Phase 4a: Batched I3 Pipeline v7")
    print(f"  Steps={args.steps}  Layers={args.num_layers}  M={args.M}  TopK={args.top_k}")
    print(f"{'='*70}")

    # ── 0. Analyze param structure ──
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    for l in layer_ids[:3]:
        large, small = classify_layer_params(layer_map[l])
        n_blks = sum(math.ceil(ne/args.block_size) for _, _, ne in large.values())
        print(f"  L{l}: {len(large)} large params → {n_blks} blocks + {len(small)} small")

    # ── 1. Oracle training (GRAPH_MODE, sink_size=1) ──
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model2 = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(model2.trainable_params(), learning_rate=1e-5)

    class TrainCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad(); self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            return ops.Depend()(loss, self.opt(grads))

    cell = TrainCell(model2, opt); ms_model = ms.Model(cell)
    dataset = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    dataset = dataset.batch(1, drop_remainder=True).take(args.steps)

    snapshots = []
    class SnapCB(ms.Callback):
        def on_train_epoch_end(self, rc):
            snapshots.append(get_all_params_np(model2))

    snapshots.append(get_all_params_np(model2))
    print(f"\n  [1] Training {args.steps} steps...")
    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=dataset, callbacks=[SnapCB()],
                   dataset_sink_mode=True, sink_size=1)
    print(f"  Done: {time.perf_counter()-t0:.1f}s  snapshots={len(snapshots)}")

    # ── 2. I3 Recovery (host-side, v6 algorithm with per-param blocks) ──
    print(f"\n  [2] I3 recovery (v6 per-param blocks + batched GE compatible)...")
    w_init = snapshots[0]; true_weights = snapshots[1:]
    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore()
    block_patches, small_patches = [], []
    err_timeline = []

    for step in range(len(true_weights)):
        true_w = true_weights[step]; selected = ctrl.select()

        for lid in selected:
            large, small = classify_layer_params(layer_map[lid])

            all_param_blocks = []
            for pi in sorted(large.keys()):
                _, name, ne = large[pi]
                fp32 = true_w[name].astype(np.float32).flatten()
                nblk = math.ceil(ne / args.block_size)
                for b in range(nblk):
                    s = b * args.block_size; e = min(s + args.block_size, ne)
                    bd = fp32[s:e]
                    po = pold.get(lid, name, b, bd)
                    dn = float(np.sum((bd - po).astype(np.float64)**2))
                    all_param_blocks.append((lid, name, b, bd, dn))

            ranked = sorted(all_param_blocks, key=lambda x: -x[4])
            tk = max(1, int(math.ceil(len(ranked) * args.top_k)))
            for lid_p, name_p, bidx_p, bd_p, dn_p in ranked[:tk]:
                sc = max(float(np.max(np.abs(bd_p)))/127.0, 1e-10)
                q = np.clip(np.round(bd_p/sc), -128, 127).astype(np.int8)
                block_patches.append({"layer_id": lid_p, "name": name_p, "block_idx": bidx_p,
                                       "int8_data": q, "scale": float(sc), "delta_norm": dn_p})
                pold.put(lid_p, name_p, bidx_p, bd_p)

            for pi in sorted(small.keys()):
                _, name, ne = small[pi]
                fp32 = true_w[name].astype(np.float32)
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                small_patches.append({"layer_id": lid, "name": name, "int8_data": q, "scale": float(sc)})
                pold.put(lid, name, 0, fp32)

        w_rec = reconstruct_v6(w_init, layer_map, block_patches, small_patches, args.block_size)
        err = compute_nrmse(w_rec, true_w)
        err_timeline.append(err)

        if (step+1) % 5 == 0:
            print(f"    Step {step+1:3d}: sel={selected}  bpatches={len(block_patches)}  "
                  f"medianNRMSE={err['median']:.4e}  P95={err['p95']:.4e}", flush=True)

    # ── 3. Results ──
    final = err_timeline[-1]
    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    fmb = sum(args.steps*layer_elems[l]*2/1e6 for l in layer_ids)

    print(f"\n{'='*70}")
    print(f"PHASE 4a v7 RECOVERY RESULTS (per-param blocks)")
    print(f"{'='*70}")
    print(f"  Recovery after {args.steps} steps (M={args.M}, topK={args.top_k}):")
    print(f"    Block patches: {len(block_patches)} ({block_mb:.1f}MB) | Small: {len(small_patches)} ({small_mb:.1f}MB)")
    print(f"    Total: {block_mb+small_mb:.1f}MB | Full: {fmb:.0f}MB | Compression: {fmb/(block_mb+small_mb):.0f}×")

    print(f"\n  Weight NRMSE (all params, step {args.steps}):")
    print(f"    Mean={final['mean']:.4e}  Median={final['median']:.4e}  "
          f"P95={final['p95']:.4e}  P99={final['p99']:.4e}  Max={final['max']:.4e}")

    worst = sorted(final["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 worst params:")
    for nm, e in worst:
        ptype = 'bias' if 'bias' in nm else ('ln' if 'layernorm' in nm else 'W')
        print(f"    [{ptype:4s}] {nm:55s}: NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}  std={e['std']:.4f}")

    med = final["median"]
    if med < 1e-3:   v = "EXCELLENT"
    elif med < 1e-2: v = "GOOD"
    elif med < 2e-2: v = "ACCEPTABLE"
    else:            v = "NEEDS IMPROVEMENT"
    print(f"\n  VERDICT: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase4a_v7.json")
    with open(out, "w") as f: json.dump({
        "experiment": "Phase 4a v7", "config": vars(args),
        "final": {"mean": final["mean"], "median": final["median"],
                  "p95": final["p95"], "p99": final["p99"], "max": final["max"]},
        "compression": {"blocks": len(block_patches), "small": len(small_patches),
                        "mb": block_mb+small_mb, "ratio": fmb/(block_mb+small_mb)},
        "timeline": [{"step": i+1, "mean": e["mean"], "median": e["median"]}
                    for i, e in enumerate(err_timeline)],
        "verdict": v,
    }, f, indent=2, default=str)
    print(f"  → {out}\n[DONE v7]")

    # ── 4. Build and measure batched GE cell ──
    print(f"\n  [3] Building batched GE cell for {args.num_layers} layers...")
    CellClass, meta = build_batched_cell(model2, opt, layer_map, args.block_size, args.num_layers)
    print(f"    Layers={meta['num_layers']}  LargeParams={meta['num_large_params']}  "
          f"Blocks={meta['num_blocks']}  EstOps={meta['est_ops']}")

    t0 = time.perf_counter()
    try:
        batched_cell = CellClass()
        ms_b = ms.Model(batched_cell)
        ds_b = ms.dataset.MindDataset(
            REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
        ds_b = ds_b.batch(1, drop_remainder=True).take(8)
        et = []
        class CB(ms.Callback):
            def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
            def on_train_epoch_end(self, rc): et.append((time.perf_counter() - self.t0) * 1000)
        ms_b.train(epoch=2, train_dataset=ds_b, callbacks=[CB()], dataset_sink_mode=True, sink_size=4)
        ce = et[0]; we = et[1] if len(et) > 1 else 0; av = we/4
        dt = time.perf_counter() - t0
        print(f"    ✅ Batched GE compiled: compile={ce:.0f}ms  avg_step={av:.1f}ms")
    except Exception as e:
        print(f"    ❌ Batched GE failed: {str(e)[:200]}")

    print("[Phase 4a] DONE.")


if __name__ == "__main__":
    main()
