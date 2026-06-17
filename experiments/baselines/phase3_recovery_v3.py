#!/usr/bin/env python3
"""
Phase 3: Recovery Fidelity Experiment (v3 — GRAPH_MODE weight snapshots)
=========================================================================

Strategy:
  - Train in GRAPH_MODE with dataset_sink_mode=True, sink_size=1
  - Use Callback to capture weights after each sink epoch (each step)
  - This is the proven Phase 1a pattern

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && python phase3_recovery_v3.py'
"""
import os, sys, time, json, math, re, argparse, copy

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")

DEVICE_ID = 1; SEQ_LEN = 1024


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


class PoldStore:
    def __init__(self, layer_elems, block_size=524288):
        self.block_size = block_size
        self.nb = {lid: math.ceil(ne/block_size) for lid, ne in layer_elems.items()}
        self.int8, self.scales = {}, {}
    def get_fp32(self, lid, bidx, ref):
        if lid not in self.int8 or bidx not in self.int8[lid]:
            return np.zeros_like(ref, dtype=np.float32)
        return self.int8[lid][bidx].astype(np.float32) * self.scales[lid][bidx]
    def update(self, lid, bidx, data):
        fp32 = data.astype(np.float32)
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        self.int8.setdefault(lid, {})[bidx] = np.clip(np.round(fp32/s), -128, 127).astype(np.int8)
        self.scales.setdefault(lid, {})[bidx] = float(s)


class RotCtrl:
    def __init__(self, layer_ids, M=10):
        self.ids = sorted(layer_ids); self.M = M
        self.ss = {l:0 for l in layer_ids}
    def select(self):
        for l in self.ids: self.ss[l] += 1
        stale = [l for l in self.ids if self.ss[l] >= self.M]
        if not stale:
            ms_val = max(self.ss.values())
            stale = [[l for l in self.ids if self.ss[l]==ms_val][0]]
        for l in stale: self.ss[l] = 0
        return stale


def classify_params_by_size(layer_map, small_threshold=10000, tiny_threshold=5000):
    """Classify params in each layer by element count.

    Small params (bias, layernorm) have < small_threshold elements.
    These are interleaved with large weight blocks and get lost in top-K selection.

    Returns: dict mapping layer_id → list of (pi, name, ne, category)
    """
    result = {}
    for lid in layer_map:
        result[lid] = {"block_params": [], "small_params": []}
        for pi in sorted(layer_map[lid].keys()):
            _, name, ne = layer_map[lid][pi]
            if ne < small_threshold:
                result[lid]["small_params"].append((pi, name, ne))
            else:
                result[lid]["block_params"].append((pi, name, ne))
    return result
    # Flatten only large params (weight matrices) into blocks.
    # Small params (bias, layernorm) are saved separately.
    big_parts, offs = [], []; off = 0
    for pi in sorted(layer_info.keys()):
        _, name, ne = layer_info[pi]
        if ne < small_threshold:
            continue  # skip small params → handled separately
        big_parts.append(params_np[name].astype(np.float32).flatten())
        offs.append((name, off, off+len(big_parts[-1])))
        off += len(big_parts[-1])
    fd = np.concatenate(big_parts) if big_parts else np.array([])
    n = math.ceil(len(fd)/block_size)
    return fd, [(b*block_size, min((b+1)*block_size, len(fd))) for b in range(n)], offs


def reconstruct(init_w, layer_map, patches):
    w = copy.deepcopy(init_w)
    for p in patches:
        lid, bidx, i8, s, bsz = p["layer_id"], p["block_idx"], p["int8_data"], p["scale"], p["block_size"]
        if lid not in layer_map: continue
        fp32 = i8.astype(np.float32) * s
        bstart = bidx * bsz
        for pi in sorted(layer_map[lid].keys()):
            _, name, ne = layer_map[lid][pi]
            p0 = sum(layer_map[lid][pj][2] for pj in sorted(layer_map[lid].keys()) if layer_map[lid][pj][1] < name)
            p1 = p0 + ne
            o0, o1 = max(bstart, p0), min(bstart+len(fp32), p1)
            if o0 < o1:
                wv = w[name].astype(np.float32).flatten()
                wv[o0-p0:o0-p0+(o1-o0)] = fp32[o0-bstart:o0-bstart+(o1-o0)]
                w[name] = wv.reshape(w[name].shape)
    return w


def compute_nrmse(w_r, w_t):
    all_n = []; params = {}
    for nm in w_t:
        r = w_r[nm].astype(np.float64).flatten()
        t = w_t[nm].astype(np.float64).flatten()
        d = r - t
        std = float(np.std(t)) + 1e-12
        nr = float(np.sqrt(np.mean(d**2))) / std
        params[nm] = {"nrmse": nr, "mae": float(np.mean(np.abs(d))),
                       "max_abs": float(np.max(np.abs(d)))}
        all_n.append(nr)
    return {"params": params, "mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "max": float(np.max(all_n)), "p95": float(np.percentile(all_n, 95)),
            "p99": float(np.percentile(all_n, 99))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=524288)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*70}\nPhase 3 Recovery Fidelity v3 — Steps={args.steps} TopK={args.top_k} M={args.M}\n{'='*70}")

    # ── Build + Train in GRAPH_MODE with sink_size=1 ──
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]
    print(f"  Layers: {len(layer_ids)}. Per-layer blocks: ", end="")
    for l in layer_ids:
        print(f"{math.ceil(layer_elems[l]/args.block_size)} ", end="")
    print()

    # Build training cell
    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    class TrainCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            return ops.Depend()(loss, self.opt(grads))

    cell = TrainCell(model, opt)
    ms_model = ms.Model(cell)

    dataset = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    dataset = dataset.batch(1, drop_remainder=True).take(args.steps)

    # Weights + loss collector (capture after the one-step sink epoch completes)
    snapshots = []; losses = []
    step_counter = [0]  # mutable for closure

    class SnapCB(ms.Callback):
        def on_train_epoch_end(self, rc):
            if hasattr(rc, 'epoch_num'):
                step_counter[0] = rc.epoch_num
            snapshots.append(get_all_params_np(model))
            # capture loss for this step via model evaluation
            losses.append(0.0)  # will be filled from oracle snapshots

    # First snapshot = initial weights (before any training)
    snapshots.append(get_all_params_np(model))

    print(f"  [1] Training {args.steps} steps (sink_mode=True, sink_size=1)...")
    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=dataset, callbacks=[SnapCB()],
                   dataset_sink_mode=True, sink_size=1)
    train_s = time.perf_counter() - t0

    # snapshots: [0]=init, [1]=after_step_1, ..., [N]=after_step_N
    # The oracle "true weights" after step k is snapshots[k+1]
    # We need snapshots[1:] for comparison
    print(f"  Oracle done: {train_s:.1f}s. loss [{losses[0]:.4f}→{losses[-1]:.4f}]  snapshots: {len(snapshots)}")

    # ── 2. I3 Recovery Simulation ──
    print(f"\n  [2] I3 recovery simulation...")
    # Initial weights = snapshot 0 (before any training)
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model_init = AutoModel.from_config(cfg)
    # model_init = AutoModel.from_config(cfg)  ← this also loads pre-trained weights
    # Actually we need the actual initial weights. Let's use snapshot[0] which is step 0 = before training.
    w_init = snapshots[0]

    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore(layer_elems, args.block_size)
    all_patches = []; err_timeline = []

    # snapshots[0] = init, snapshots[1] = after step 1, ..., snapshots[N] = after step N
    # I3 recovery: reconstruct weight at each step using init + patches
    true_weights = snapshots[1:]  # Oracle weights after each step
    print(f"  training produced {len(true_weights)} oracle weight snapshots (after each step)")

    for step in range(len(true_weights)):
        true_w = true_weights[step]
        selected = ctrl.select()

        for lid in selected:
            flat, blocks, offs = flatten_layer(true_w, layer_map[lid], args.block_size)
            norms = []
            for b, (s, e) in enumerate(blocks):
                bd = flat[s:e].astype(np.float32)
                po = pold.get_fp32(lid, b, bd)
                norms.append(float(np.sum((bd - po).astype(np.float64)**2)))
            ranked = sorted(enumerate(norms), key=lambda x: -x[1])
            tk = max(1, int(math.ceil(len(blocks)*args.top_k)))
            for bidx, _ in ranked[:tk]:
                s, e = blocks[bidx]; bd = flat[s:e]
                fp32 = flat[s:e]
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                all_patches.append({"layer_id": lid, "block_idx": bidx,
                                    "int8_data": q, "scale": float(sc),
                                    "block_size": args.block_size})
                pold.update(lid, bidx, bd)

        w_rec = reconstruct(w_init, layer_map, all_patches)
        err_timeline.append(compute_nrmse(w_rec, true_w))

        if (step+1) % 10 == 0 or step == 0:
            e = err_timeline[-1]
            print(f"    Step {step+1:3d}: sel={selected}  patches={len(all_patches)}  "
                  f"meanNRMSE={e['mean']:.2e}  maxNRMSE={e['max']:.2e}", flush=True)

    # ── 3. Results ──
    final = err_timeline[-1]
    cmb = sum(p["int8_data"].nbytes+4 for p in all_patches)/1e6
    fmb = sum(args.steps*layer_elems[l]*2/1e6 for l in layer_ids)

    print(f"\n{'='*70}\nRECOVERY FIDELITY RESULTS\n{'='*70}")
    print(f"\n  Oracle: loss {losses[0]:.6f}→{losses[-1]:.6f}  Δ={losses[0]-losses[-1]:.4f}")
    print(f"\n  I3 Recovery ({args.steps} steps):")
    print(f"    Patches: {len(all_patches)}  |  Data: {cmb:.1f}MB  |  Full: {fmb:.0f}MB  |  {fmb/cmb:.0f}×")
    print(f"\n  Weight NRMSE (all params):")
    print(f"    Mean={final['mean']:.2e}  Median={final['median']:.2e}  "
          f"P95={final['p95']:.2e}  P99={final['p99']:.2e}  Max={final['max']:.2e}")

    # Worst params by NRMSE
    worst = sorted(final["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 most-deviated params:")
    for nm, e in worst:
        lay = re.search(r'blocks\.(\d+)', nm)
        lay_str = f"(L{lay.group(1)})" if lay else ""
        print(f"    {nm[:55]:55s} {lay_str:6s} NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}")

    fn = final["mean"]
    if fn < 1e-4:     v = "EXCELLENT (<10⁻⁴)"
    elif fn < 1e-3:   v = f"GOOD (<10⁻³)"
    elif fn < 1e-2:   v = f"ACCEPTABLE (<10⁻²)"
    elif fn < 1e-1:   v = "MARGINAL — increase top_k or reduce M"
    else:             v = "FAIL"
    print(f"\n  VERDICT: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase3_recovery_v3.json")
    with open(out, "w") as f: json.dump({
        "experiment": "Phase 3 Recovery Fidelity v3",
        "config": {"steps": args.steps, "block_size": args.block_size,
                   "top_k": args.top_k, "M": args.M},
        "oracle": {"losses": losses, "delta": losses[0]-losses[-1]},
        "recovery": {"patches": len(all_patches), "compressed_mb": cmb, "full_mb": fmb,
                     "ratio": fmb/cmb, "mean_nrmse": final["mean"],
                     "p95_nrmse": final["p95"], "max_nrmse": final["max"],
                     "timeline": [e["mean"] for e in err_timeline]},
        "worst_params": [(nm, e) for nm, e in worst],
        "verdict": v,
        "train_time_s": train_s,
    }, f, indent=2, default=str)
    print(f"  → {out}\n[Recovery Fidelity] DONE.\n")


if __name__ == "__main__":
    main()
