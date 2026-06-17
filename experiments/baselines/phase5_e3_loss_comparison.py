#!/usr/bin/env python3
"""
Phase 5 E3 v2: Direct Loss Comparison — I3 Recovery vs Oracle
===============================================================
Fixed version: handles sink_size=1 loss tracking, uses subset sampling
for oracle loss, and correctly handles I3 recovery timeline alignment.

Method:
  - Oracle: train 50 steps, snapshot weights every step
  - I3 Recovery: host-side per-param blocks + top-K + INT8 quantization
  - Compare: NRMSE + loss on eval batch at select checkpoints

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_e3_loss_comparison.py --steps 50'
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
CHECKPOINT_DIR = os.path.join(REPO, "checkpoint_download", "gpt2")


# ═══════════════ Utilities ═══════════════

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
    all_n = []
    for nm in w_t:
        r = w_r[nm].astype(np.float64).flatten()
        t = w_t[nm].astype(np.float64).flatten()
        d = r - t; std = float(np.std(t)) + 1e-12
        nr = float(np.sqrt(np.mean(d**2))) / std
        all_n.append(nr)
    return {"mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "p95": float(np.percentile(all_n, 95)), "max": float(np.max(all_n))}


def reconstruct_v6(init_w, block_patches, small_patches, block_size):
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


def compute_loss_pynative(model, batch_data):
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    input_ids = Tensor(batch_data.astype(np.int32), ms.int32)
    input_mask = Tensor(np.ones(batch_data.shape, dtype=np.int32), ms.int32)
    output = model(input_ids, input_mask)
    loss = output[0] if isinstance(output, tuple) else output
    return float(loss.asnumpy().flatten()[0])


# ═══════════════ Training ═══════════════

def train_oracle_pynative(model, opt, ds, n_steps):
    """PYNATIVE training with per-step weight snapshots. Much more reliable than sink."""
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    gf = ops.value_and_grad(model, grad_position=None, weights=opt.parameters)
    ds_iter = ds.create_tuple_iterator()

    snapshots = [(0, get_all_params_np(model))]
    step_times = []

    for s in range(n_steps):
        t0 = time.perf_counter()
        try:
            batch = next(ds_iter)
        except StopIteration:
            ds_iter = ds.create_tuple_iterator()
            batch = next(ds_iter)

        input_ids = Tensor(batch[0], ms.int32)
        input_mask = Tensor(np.ones(batch[0].shape, dtype=np.int32), ms.int32)
        result = gf(input_ids, input_mask)
        model_output = result[0]
        if isinstance(model_output, (tuple, list)):
            loss_val = float(model_output[0].asnumpy().flatten()[0])
        else:
            loss_val = float(model_output.asnumpy().flatten()[0])
        grads = result[1]
        opt(grads)
        dt = (time.perf_counter() - t0) * 1000
        step_times.append(dt)
        snapshots.append((s + 1, get_all_params_np(model)))

        if (s + 1) % 10 == 0:
            avg = np.mean(step_times[-10:])
            print(f"  Step {s+1:3d}: {avg:.0f}ms  loss={loss_val:.4f}")

    avg_step = np.mean(step_times)
    return snapshots, avg_step


# ═══════════════ Main ═══════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 E3 v2: Direct Loss Comparison — Oracle vs I3 Recovery")
    print(f"  Steps={args.steps}  M={args.M}  TopK={args.top_k}")
    print("=" * 70)

    # ── 0. Analyze ──
    print("\n[0] Model analysis...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""  # Use random init, no ckpt load
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    for l in layer_ids[:3]:
        large, small = classify_layer_params(layer_map[l])
        n_blks = sum(math.ceil(ne/args.block_size) for _, _, ne in large.values())
        print(f"  L{l}: {len(large)} large → {n_blks} blocks + {len(small)} small")

    # ── 1. Prepare eval batch ──
    print("\n[1] Preparing eval data...")
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=False)
    ds = ds.batch(1, drop_remainder=True)
    ds_iter = ds.create_tuple_iterator()
    eval_batch = next(ds_iter)[0].asnumpy()  # fixed batch for loss measurement

    # ── 2. Oracle training (PYNATIVE) ──
    print(f"\n[2] Oracle training {args.steps} steps (PYNATIVE)...")
    ms.set_seed(42)
    ms.common.set_seed(42)
    model2 = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(model2.trainable_params(), learning_rate=1e-5)

    t0 = time.perf_counter()
    ds_train = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds_train = ds_train.batch(1, drop_remainder=True)
    snapshots, avg_step = train_oracle_pynative(model2, opt, ds_train, args.steps)
    dt = time.perf_counter() - t0
    print(f"  Oracle done: {dt:.1f}s ({avg_step:.0f}ms/step)")

    # ── 3. Compute oracle loss for key checkpoints ──
    print(f"\n[3] Oracle loss at key checkpoints...")
    loss_model = AutoModel.from_config(cfg)
    oracle_losses = {}
    for step, w in snapshots:
        if step % 5 == 0 or step == 0 or step == args.steps:
            for p in loss_model.trainable_params():
                if p.name in w:
                    p.set_data(Tensor(w[p.name], ms.float32))
            loss_val = compute_loss_pynative(loss_model, eval_batch)
            oracle_losses[step] = loss_val
    for s in sorted(oracle_losses.keys())[:6]:
        print(f"  Step {s:3d}: oracle_loss={oracle_losses[s]:.6f}")
    print(f"  ...")
    for s in sorted(oracle_losses.keys())[-3:]:
        print(f"  Step {s:3d}: oracle_loss={oracle_losses[s]:.6f}")

    # ── 4. I3 Recovery ──
    print(f"\n[4] I3 recovery (host-side, per-param blocks + top-K)...")
    w_init = snapshots[0][1]
    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore()
    block_patches, small_patches = [], []
    nrmse_timeline = []

    # Actually use all layer selection (Phase 5 simplified: no rotation!
    # Just select ALL layers every step — batched GE makes this zero overhead)
    for step_idx in range(1, len(snapshots)):
        step, true_w = snapshots[step_idx]

        # Phase 5 simplified: process ALL layers every step
        selected = list(layer_ids)  # ALL layers every step

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

        # Reconstruct
        w_rec = reconstruct_v6(w_init, block_patches, small_patches, args.block_size)
        err = compute_nrmse(w_rec, true_w)
        nrmse_timeline.append({"step": step, **err})

        if step % 10 == 0:
            # Compute I3 loss for this step
            for p in loss_model.trainable_params():
                if p.name in w_rec:
                    p.set_data(Tensor(w_rec[p.name], ms.float32))
            i3_l = compute_loss_pynative(loss_model, eval_batch)
            o_l = oracle_losses.get(step, oracle_losses.get(max(k for k in oracle_losses if k <= step), 0))
            rel_d = abs(i3_l - o_l) / (abs(o_l) + 1e-12) * 100
            print(f"  Step {step:3d}: oracle={o_l:.6f}  i3={i3_l:.6f}  "
                  f"Δ={i3_l-o_l:+.6f} ({rel_d:.2f}%)  MedNRMSE={err['median']:.4e}")

    # ── 5. Final i3 vs oracle comparison ──
    final_err = nrmse_timeline[-1]
    final_step, final_true = snapshots[-1]
    w_final = reconstruct_v6(w_init, block_patches, small_patches, args.block_size)
    for p in loss_model.trainable_params():
        if p.name in w_final:
            p.set_data(Tensor(w_final[p.name], ms.float32))
    final_i3_loss = compute_loss_pynative(loss_model, eval_batch)
    final_oracle_loss = oracle_losses.get(args.steps, list(oracle_losses.values())[-1])
    final_rel_diff = abs(final_i3_loss - final_oracle_loss) / (abs(final_oracle_loss) + 1e-12) * 100

    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    fmb = sum(args.steps*layer_elems[l]*2/1e6 for l in layer_ids)

    print(f"\n{'='*70}")
    print(f"E3 RESULTS: Loss Comparison ({args.steps} steps)")
    print(f"{'='*70}")
    print(f"\n  Weight Fidelity (step {args.steps}):")
    print(f"    Median NRMSE: {final_err['median']:.4e}  P95: {final_err['p95']:.4e}  Max: {final_err['max']:.4e}")
    print(f"\n  Loss Fidelity (step {args.steps}):")
    print(f"    Oracle loss: {final_oracle_loss:.6f}")
    print(f"    I3 loss:     {final_i3_loss:.6f}")
    print(f"    Rel diff:    {final_rel_diff:.2f}%")
    print(f"\n  Compression: {block_mb+small_mb:.1f}MB saved / {fmb:.0f}MB full → {fmb/(block_mb+small_mb):.0f}×")

    if final_rel_diff < 1.0:
        verdict = "EXCELLENT: loss diff < 1%"
    elif final_rel_diff < 5.0:
        verdict = "GOOD: loss diff < 5%"
    else:
        verdict = "NEEDS ANALYSIS"
    print(f"  VERDICT: {verdict}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_e3_loss_comparison.json")
    with open(out, "w") as f:
        json.dump({"experiment": "Phase 5 E3: Loss Comparison",
                   "config": vars(args),
                   "training": {"n_steps": args.steps, "avg_step_ms": avg_step, "total_s": dt},
                   "oracle_losses": {str(k): v for k, v in oracle_losses.items()},
                   "final_loss": {"oracle": final_oracle_loss, "i3": final_i3_loss,
                                  "rel_diff_pct": final_rel_diff},
                   "nrmse": {"final": final_err, "timeline": nrmse_timeline},
                   "compression": {"ratio": fmb/(block_mb+small_mb) if (block_mb+small_mb) > 0 else 0,
                                   "saved_mb": block_mb+small_mb},
                   "verdict": verdict}, f, indent=2, default=str)
    print(f"  → Saved: {out}\n[E3 DONE]")


if __name__ == "__main__":
    main()
