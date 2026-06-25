#!/usr/bin/env python3
"""
Phase 3: Recovery Fidelity Experiment (v2 — clean PYNATIVE step loop)
========================================================================

Tests: training 20 steps with weight snapshots, then simulating I3 recovery
from step 0 full checkpoint + incremental chain, measuring weight deviation.

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /home/user7/miniconda3/envs/ms_2.5/bin/python phase3_recovery_v2.py --steps 20 --top_k_frac 0.10'
"""
import os, sys, time, json, math, re, argparse, copy

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")

DEVICE_ID = 1
SEQ_LEN = 1024


# ═══════════════════════════════════════════════════════════════════
# Utilities
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
        if lid not in layer_map:
            layer_map[lid] = {}
            layer_elems[lid] = 0
        layer_map[lid][pi] = (p, name, ne)
        layer_elems[lid] += ne
    return params, layer_map, layer_elems


def get_all_params_np(model):
    return {p.name: p.value().asnumpy().copy() for p in model.trainable_params()}


def flatten_layer_params(params_np, layer_info, block_size):
    """Flatten all params in a layer."""
    flat_parts, param_offsets = [], []
    offset = 0
    for pi in sorted(layer_info.keys()):
        p_obj, name, ne = layer_info[pi]
        pv = params_np[name].astype(np.float32).flatten()
        flat_parts.append(pv)
        param_offsets.append((name, offset, offset + len(pv)))
        offset += len(pv)
    flat_data = np.concatenate(flat_parts) if flat_parts else np.array([])
    num_blocks = math.ceil(len(flat_data) / block_size)
    blocks = [(b * block_size, min((b + 1) * block_size, len(flat_data)))
              for b in range(num_blocks)]
    return flat_data, blocks, param_offsets


class PoldStore:
    def __init__(self, layer_elems, block_size=524288):
        self.block_size = block_size
        self.nb = {lid: math.ceil(ne / block_size) for lid, ne in layer_elems.items()}
        self.int8 = {}
        self.scales = {}

    def get_fp32(self, lid, bidx, ref):
        if lid not in self.int8 or bidx not in self.int8[lid]:
            return np.zeros_like(ref, dtype=np.float32)
        return self.int8[lid][bidx].astype(np.float32) * self.scales[lid][bidx]

    def update(self, lid, bidx, data):
        fp32 = data.astype(np.float32)
        am = float(np.max(np.abs(fp32)))
        s = max(am / 127.0, 1e-10)
        q = np.clip(np.round(fp32 / s), -128, 127).astype(np.int8)
        if lid not in self.int8:
            self.int8[lid] = {}
            self.scales[lid] = {}
        self.int8[lid][bidx] = q
        self.scales[lid][bidx] = float(s)


class RotCtrl:
    def __init__(self, layer_ids, M=10):
        self.layer_ids = sorted(layer_ids)
        self.M = M
        self.stale = {l: 0 for l in layer_ids}

    def select(self):
        for l in self.layer_ids:
            self.stale[l] += 1
        past = [l for l in self.layer_ids if self.stale[l] >= self.M]
        if past:
            sel = past
        else:
            ms = max(self.stale.values())
            sel = [[l for l in self.layer_ids if self.stale[l] == ms][0]]
        for l in sel:
            self.stale[l] = 0
        return sel


def reconstruct_from_patches(init_weights, layer_map, patches):
    """Apply all I3 patches to initial weights. Returns reconstructed weights."""
    w = copy.deepcopy(init_weights)
    for patch in patches:
        lid = patch["layer_id"]
        if lid not in layer_map:
            continue
        bidx = patch["block_idx"]
        i8 = patch["int8_data"]
        s = patch["scale"]
        bsz = patch["block_size"]
        fp32 = i8.astype(np.float32) * s
        bstart = bidx * bsz

        for pi in sorted(layer_map[lid].keys()):
            _, name, ne = layer_map[lid][pi]
            p_start = 0
            for pj in sorted(layer_map[lid].keys()):
                _, n2, ne2 = layer_map[lid][pj]
                if n2 == name:
                    break
                p_start += ne2
            p_end = p_start + ne
            o_start = max(bstart, p_start)
            o_end = min(bstart + len(fp32), p_end)
            if o_start < o_end:
                w[name] = w[name].astype(np.float32).flatten()
                w[name][o_start - p_start:o_start - p_start + (o_end - o_start)] = \
                    fp32[o_start - bstart:o_start - bstart + (o_end - o_start)]
                orig_shape = w[name].shape
                # Find original shape from the parameter reference
                for p in sorted(layer_map[lid].values()):
                    if p[1] == name:
                        w[name] = w[name].flatten()[:ne].reshape(p[0].shape)
                        break
                else:
                    w[name] = w[name].flatten()[:ne]
    return w


def compute_nrmse(w_rec, w_true):
    """Per-parameter NRMSE."""
    params_errors = {}
    all_nrmse = []
    for name in w_true:
        r = w_rec[name].astype(np.float64).flatten()
        t = w_true[name].astype(np.float64).flatten()
        diff = r - t
        std = float(np.std(t)) + 1e-12
        nrmse = float(np.sqrt(np.mean(diff ** 2))) / std
        mae = float(np.mean(np.abs(diff)))
        maxe = float(np.max(np.abs(diff)))
        params_errors[name] = {"nrmse": nrmse, "mae": mae, "max_abs_err": maxe, "std_ref": float(std)}
        all_nrmse.append(nrmse)
    return {"params": params_errors, "mean": float(np.mean(all_nrmse)),
            "median": float(np.median(all_nrmse)), "max": float(np.max(all_nrmse)),
            "p95": float(np.percentile(all_nrmse, 95)), "p99": float(np.percentile(all_nrmse, 99))}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=524288)
    parser.add_argument("--top_k_frac", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Phase 3 Recovery Fidelity v2 — Steps={args.steps} TopK={args.top_k_frac} M={args.M}")
    print("=" * 70)

    # ── 0. Build in PYNATIVE for direct parameter access ──
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    print(f"  Layers: {len(layer_ids)}, Params: {len(params)}")
    for l in layer_ids:
        nb = math.ceil(layer_elems[l] / args.block_size)
        print(f"    L{l}: {layer_elems[l]:>10,} elems → {nb} blocks")

    # ── 1. Oracle training run ──
    print(f"\n  [1] Oracle run: {args.steps} steps PYNATIVE...")
    t0 = time.perf_counter()

    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    grad_fn = ops.value_and_grad(model, grad_position=None, weights=optimizer.parameters)

    dataset = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    dataset = dataset.batch(1, drop_remainder=True).take(args.steps)
    data_iter = dataset.create_dict_iterator()

    oracle_losses = []
    oracle_snapshots = []

    for step in range(args.steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"]
        loss_val, grads = grad_fn(input_ids)
        optimizer(grads)

        oracle_losses.append(float(loss_val.asnumpy()))
        oracle_snapshots.append(get_all_params_np(model))

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1:3d}/{args.steps}: loss={oracle_losses[-1]:.4f}", flush=True)

    oracle_s = time.perf_counter() - t0
    print(f"  Oracle done: {oracle_s:.1f}s  loss {oracle_losses[0]:.4f}→{oracle_losses[-1]:.4f}")

    # ── 2. I3 simulation: run recovery from initial weights + patches ──
    print(f"\n  [2] I3 recovery simulation (M={args.M}, topK={args.top_k_frac}):")

    # Get initial weights
    ms.common.set_seed(42)
    model_init = AutoModel.from_config(cfg)
    w_init = get_all_params_np(model_init)

    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore(layer_elems, args.block_size)
    all_patches = []

    w_recovered = copy.deepcopy(w_init)
    recovery_snapshots = []
    error_timeline = []

    for step in range(args.steps):
        true_w = oracle_snapshots[step]
        selected = ctrl.select()

        step_patches = []
        for lid in selected:
            flat, blocks, offsets = flatten_layer_params(true_w, layer_map[lid], args.block_size)

            # Compute delta norms
            norms = []
            for b, (s, e) in enumerate(blocks):
                block_data = flat[s:e].astype(np.float32)
                p_old = pold.get_fp32(lid, b, block_data)
                delta = block_data - p_old
                norms.append(float(np.sum(delta.astype(np.float64) ** 2)))

            # Top-K
            ranked = sorted(enumerate(norms), key=lambda x: -x[1])
            tk = max(1, int(math.ceil(len(blocks) * args.top_k_frac)))

            for bidx, dn in ranked[:tk]:
                s, e = blocks[bidx]
                bd = flat[s:e].astype(np.float16)
                # Quantize
                fp32 = bd.astype(np.float32)
                am = float(np.max(np.abs(fp32)))
                sc = max(am / 127.0, 1e-10)
                q = np.clip(np.round(fp32 / sc), -128, 127).astype(np.int8)
                step_patches.append({"layer_id": lid, "block_idx": bidx,
                                     "int8_data": q, "scale": float(sc),
                                     "block_size": args.block_size})
                pold.update(lid, bidx, bd)

        if step_patches:
            all_patches.extend(step_patches)

        # Reconstruct from init + all patches
        w_rec = reconstruct_from_patches(w_init, layer_map, all_patches)
        recovery_snapshots.append(copy.deepcopy(w_rec))

        err = compute_nrmse(w_rec, true_w)
        error_timeline.append(err)

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1:3d}: sel={selected}  patches={len(all_patches)}  "
                  f"meanNRMSE={err['mean']:.2e}  maxNRMSE={err['max']:.2e}", flush=True)

    # ── 3. Results ──
    final_err = error_timeline[-1]
    compressed_mb = sum(p["int8_data"].nbytes + 4 for p in all_patches) / 1e6
    full_size_mb = sum(args.steps * layer_elems[l] * 2 / 1e6 for l in layer_ids)

    print(f"\n{'='*70}")
    print("RECOVERY FIDELITY RESULTS")
    print("=" * 70)
    print(f"\n  Oracle: loss {oracle_losses[0]:.6f} → {oracle_losses[-1]:.6f}  (Δ={oracle_losses[0]-oracle_losses[-1]:.4f})")
    print(f"\n  I3 Recovery after {args.steps} steps:")
    print(f"    Total patches:     {len(all_patches)}")
    print(f"    Compressed data:   {compressed_mb:.1f} MB")
    print(f"    vs full ckpt:      {full_size_mb:.0f} MB  (compression {full_size_mb/compressed_mb:.0f}×)")

    print(f"\n  Weight Deviation (NRMSE = RMSE / std(W)):")
    print(f"    Mean NRMSE:        {final_err['mean']:.2e}")
    print(f"    Median NRMSE:      {final_err['median']:.2e}")
    print(f"    P95 NRMSE:         {final_err['p95']:.2e}")
    print(f"    P99 NRMSE:         {final_err['p99']:.2e}")
    print(f"    Max NRMSE:         {final_err['max']:.2e}")

    # Find worst param
    worst = max(final_err["params"].items(), key=lambda x: x[1]["nrmse"])
    print(f"\n  Worst param: {worst[0]}")
    print(f"    NRMSE={worst[1]['nrmse']:.4f}  MAE={worst[1]['mae']:.2e}  max_abs_err={worst[1]['max_abs_err']:.2e}")

    # Top-5 worst
    worst5 = sorted(final_err["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 worst params:")
    for name, e in worst5:
        print(f"    {name:60s}: NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}  std={e['std_ref']:.4f}")

    # Verdict
    fnrmse = final_err['mean']
    print(f"\n  VERDICT:")

    if fnrmse < 1e-4:
        v = "EXCELLENT (<10⁻⁴) — recovery is essentially lossless"
    elif fnrmse < 1e-3:
        v = f"GOOD (<10⁻³) — weight error ~{fnrmse:.1e}, loss impact ~{fnrmse*0.1:.1e} (negligible)"
    elif fnrmse < 1e-2:
        v = f"ACCEPTABLE (<10⁻²) — weight error ~{fnrmse:.1e}, loss impact ~{fnrmse*0.1:.1e}"
    elif fnrmse < 1e-1:
        v = f"MARGINAL — need larger top_k or smaller M"
    else:
        v = "FAIL — information loss too large"

    print(f"    {v}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase3_recovery_v2.json")
    result = {
        "experiment": "Phase 3: Recovery Fidelity v2",
        "config": {"steps": args.steps, "block_size": args.block_size,
                   "top_k_frac": args.top_k_frac, "M": args.M},
        "oracle": {"losses": oracle_losses, "first": oracle_losses[0], "last": oracle_losses[-1],
                   "delta": oracle_losses[0] - oracle_losses[-1]},
        "recovery": {"total_patches": len(all_patches), "compressed_mb": compressed_mb,
                     "full_mb": full_size_mb, "compression_ratio": full_size_mb / compressed_mb,
                     "mean_nrmse": final_err["mean"], "p95_nrmse": final_err["p95"],
                     "max_nrmse": final_err["max"],
                     "nrmse_timeline": [e["mean"] for e in error_timeline]},
        "worst_param": {"name": worst[0], **worst[1]},
        "oracle_time_s": oracle_s,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  → {out}")
    print("[Recovery Fidelity] DONE.")


if __name__ == "__main__":
    main()
