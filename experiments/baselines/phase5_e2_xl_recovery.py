#!/usr/bin/env python3
"""
Phase 5 E2: GPT-2 XL Recovery Fidelity v7
==========================================
Validates that the per-param block + batched delta approach produces
acceptable recovery fidelity for GPT-2 XL (48 layers, 1.56B params).

IMPORTANT: GRAPH_MODE training with XL can be slow (~400ms/step).
For E2 we use PYNATIVE_MODE for training (more reliable on XL)
and host-side I3 recovery (no GE compilation needed for recovery).

Method:
  - Train GPT-2 XL in PYNATIVE for 10 steps, record weight snapshots
  - Run host-side I3 recovery (same as v6/v7 algorithm with per-param blocks)
  - Compute NRMSE at each step
  - Verify: Median NRMSE < 5%, Max NRMSE < 10%

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_e2_xl_recovery.py --steps 10'
"""
import os, sys, time, json, math, re, argparse, copy
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
# Utilities
# ═══════════════════════════════════════════════

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
            "p95": float(np.percentile(all_n, 95)), "max": float(np.max(all_n)),
            "p99": float(np.percentile(all_n, 99))}


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


# ═══════════════════════════════════════════════
# PYNATIVE Training
# ═══════════════════════════════════════════════

def train_pynative_model(model, opt, dataset, n_steps):
    """Train in PYNATIVE_MODE, return snapshot list [(step, weights_dict)]."""
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)

    gf = ops.value_and_grad(model, grad_position=None, weights=opt.parameters)

    snapshots = [(0, get_all_params_np(model))]
    ds_iter = dataset.create_tuple_iterator()
    step_times = []

    for step in range(n_steps):
        t0 = time.perf_counter()
        try:
            batch = next(ds_iter)
        except StopIteration:
            ds_iter = dataset.create_tuple_iterator()
            batch = next(ds_iter)

        input_ids = Tensor(batch[0], ms.int32)
        input_mask = Tensor(np.ones_like(batch[0].asnumpy(), dtype=np.int32), ms.int32)

        result = gf(input_ids, input_mask)
        # value_and_grad returns (model_output, grads_tuple)
        # model_output for GPT2LMHeadModel is (loss, logits)
        model_output = result[0]
        if isinstance(model_output, (tuple, list)):
            loss_val = float(model_output[0].asnumpy().flatten()[0])
        else:
            loss_val = float(model_output.asnumpy().flatten()[0])
        grads = result[1]
        opt(grads)
        opt(grads)

        # Record weights after optimizer step
        snapshots.append((step + 1, get_all_params_np(model)))
        dt = (time.perf_counter() - t0) * 1000
        step_times.append(dt)

        if (step + 1) % 5 == 0:
            avg = np.mean(step_times[-5:])
            print(f"  Step {step+1:3d}: {avg:.0f}ms  loss={loss_val:.4f}")

    avg_step = np.mean(step_times)
    print(f"  Done: avg_step={avg_step:.0f}ms")
    return snapshots, avg_step


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 E2: GPT-2 XL Recovery Fidelity v7")
    print(f"  Steps={args.steps}  M={args.M}  TopK={args.top_k}  BS={args.block_size}")
    print("=" * 70)

    # ── 0. Analyze model ──
    print("\n[0] Analyzing GPT-2 XL structure...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg_xl = AutoConfig.from_pretrained("gpt2_xl"); cfg_xl.seq_length=1025; cfg_xl.max_position_embeddings=1025
    cfg_xl.checkpoint_name_or_path = ""  # Don't auto-load ckpt for E2
    model = AutoModel.from_config(cfg_xl)
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    total_elems = sum(sum(n for _, _, n in layer_map[l].values()) for l in layer_ids)
    total_large = sum(sum(1 for _, _, n in layer_map[l].values() if n >= SMALL_THRESHOLD) for l in layer_ids)
    total_small = sum(sum(1 for _, _, n in layer_map[l].values() if n < SMALL_THRESHOLD) for l in layer_ids)

    print(f"  {len(layer_ids)} layers, {total_elems/1e6:.0f}M elems ({total_elems*2/1e9:.2f}GB FP16)")
    print(f"  {total_large} large params, {total_small} small params")

    # Per-layer block count
    total_blocks = 0
    for l in layer_ids[:3]:
        large, small = classify_layer_params(layer_map[l])
        n_blks = sum(math.ceil(ne/args.block_size) for _, _, ne in large.values())
        total_blocks += n_blks
        print(f"  L{l}: {len(large)} large → {n_blks} blocks + {len(small)} small  ({layer_elems[l]/1e6:.1f}M elems)")
    print(f"  ... (48 layers total)")

    # Total block count across all layers
    all_blocks = 0
    for l in layer_ids:
        large, _ = classify_layer_params(layer_map[l])
        all_blocks += sum(math.ceil(ne/args.block_size) for _, _, ne in large.values())
    print(f"  Total: ~{all_blocks} blocks across {len(layer_ids)} layers")

    # ── 1. Train in PYNATIVE ──
    print(f"\n[1] Training GPT-2 XL {args.steps} steps (PYNATIVE)...")
    ms.common.set_seed(42)
    model2 = AutoModel.from_config(cfg_xl)
    opt = nn.AdamWeightDecay(model2.trainable_params(), learning_rate=1e-5)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True)

    t0 = time.perf_counter()
    snapshots, avg_step = train_pynative_model(model2, opt, ds, args.steps)
    dt = time.perf_counter() - t0
    print(f"  Training total: {dt:.1f}s ({args.steps} steps)")

    # ── 2. I3 Recovery ──
    print(f"\n[2] I3 recovery (host-side, per-param blocks)...")
    w_init = snapshots[0][1]  # step 0
    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore()
    block_patches, small_patches = [], []
    nrmse_timeline = []

    for step_idx in range(1, len(snapshots)):
        step, true_w = snapshots[step_idx]

        # Selection
        selected = ctrl.select()

        step_block_count = 0
        step_small_count = 0

        # Per-selected-layer
        for lid in selected:
            large, small = classify_layer_params(layer_map[lid])

            # All param-blocks in this layer → compute delta
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

            # Top-K
            ranked = sorted(all_param_blocks, key=lambda x: -x[4])
            tk = max(1, int(math.ceil(len(ranked) * args.top_k)))
            for lid_p, name_p, bidx_p, bd_p, dn_p in ranked[:tk]:
                sc = max(float(np.max(np.abs(bd_p)))/127.0, 1e-10)
                q = np.clip(np.round(bd_p/sc), -128, 127).astype(np.int8)
                block_patches.append({"layer_id": lid_p, "name": name_p, "block_idx": bidx_p,
                                       "int8_data": q, "scale": float(sc), "delta_norm": dn_p})
                pold.put(lid_p, name_p, bidx_p, bd_p)
                step_block_count += 1

            # Small params
            for pi in sorted(small.keys()):
                _, name, ne = small[pi]
                fp32 = true_w[name].astype(np.float32)
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                small_patches.append({"layer_id": lid, "name": name, "int8_data": q, "scale": float(sc)})
                pold.put(lid, name, 0, fp32)
                step_small_count += 1

        # Reconstruct
        w_rec = reconstruct_v6(w_init, layer_map, block_patches, small_patches, args.block_size)

        # NRMSE
        err = compute_nrmse(w_rec, true_w)
        nrmse_timeline.append({"step": step, **{k: err[k] for k in ["mean","median","p95","max","p99"]}})

        if step % 3 == 0 or step == args.steps:
            print(f"  Step {step:3d}: sel={selected}  blocks={step_block_count}  small={step_small_count}  "
                  f"MedNRMSE={err['median']:.4e}  P95={err['p95']:.4e}  Max={err['max']:.4e}")

    # ── 3. Report ──
    final = nrmse_timeline[-1]
    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    total_saved_mb = block_mb + small_mb
    full_mb_per_step = sum(layer_elems[l]*2/1e6 for l in layer_ids) * args.steps

    print(f"\n{'='*70}")
    print(f"E2 RESULTS: GPT-2 XL Recovery Fidelity ({args.steps} steps)")
    print(f"{'='*70}")

    print(f"\n  Training: {args.steps} steps @ {avg_step:.0f}ms/step")
    print(f"  Total snapshots: {len(snapshots)}")

    print(f"\n  Recovery (step {args.steps}):")
    print(f"    Mean NRMSE:   {final['mean']:.4e}")
    print(f"    Median NRMSE: {final['median']:.4e}")
    print(f"    P95 NRMSE:    {final['p95']:.4e}")
    print(f"    P99 NRMSE:    {final['p99']:.4e}")
    print(f"    Max NRMSE:    {final['max']:.4e}")

    if final['median'] < 1e-3:
        verdict = "EXCELLENT"
    elif final['median'] < 0.01:
        verdict = "GOOD"
    elif final['median'] < 0.05:
        verdict = "ACCEPTABLE"
    elif final['median'] < 0.10:
        verdict = "ADEQUATE"
    else:
        verdict = "FAIL"

    # Show worst params
    worst = sorted(final["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 worst params:")
    for nm, e in worst:
        ptype = 'bias' if 'bias' in nm else ('ln' if 'layernorm' in nm else 'W')
        print(f"    [{ptype:4s}] {nm:55s}: NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}")

    print(f"\n  Compression:")
    print(f"    Total saved: {total_saved_mb:.1f}MB (block={block_mb:.1f}MB + small={small_mb:.1f}MB)")
    print(f"    Full model:  {full_mb_per_step:.0f}MB")
    if total_saved_mb > 0:
        print(f"    Ratio: {full_mb_per_step/total_saved_mb:.0f}×")

    print(f"\n  VERDICT: {verdict}")

    # Paper target check
    if final['median'] < 0.05 and final['max'] < 0.10:
        print(f"  ✅ MEETS TARGET: Median NRMSE < 5% AND Max NRMSE < 10%")
    elif final['median'] < 0.05:
        print(f"  ⚠️  MEDIAN OK but MAX exceeds target (need to check outliers)")
    else:
        print(f"  ❌ Median NRMSE exceeds 5% target")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_e2_xl_recovery.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 E2: XL Recovery",
            "config": vars(args),
            "model": {"name": "GPT-2 XL", "layers": len(layer_ids), "elems_m": total_elems/1e6,
                      "fp16_gb": total_elems*2/1e9, "large_params": total_large, "small_params": total_small,
                      "total_blocks": all_blocks},
            "training": {"mode": "PYNATIVE", "n_steps": args.steps, "avg_step_ms": avg_step, "total_s": dt},
            "recovery": {
                "final": final,
                "target_check": {"median_OK": final['median'] < 0.05, "max_OK": final['max'] < 0.10},
                "timeline": nrmse_timeline,
            },
            "compression": {"total_saved_mb": total_saved_mb, "block_mb": block_mb,
                           "small_mb": small_mb, "full_step_mb": full_mb_per_step/args.steps},
            "verdict": verdict,
        }, f, indent=2, default=str)
    print(f"  → Saved: {out}")
    print("[E2 DONE]")


if __name__ == "__main__":
    main()
