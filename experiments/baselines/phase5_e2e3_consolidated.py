#!/usr/bin/env python3
"""
Phase 5 E2+E3 v8: Loss Comparison + NRMSE (GPT-2 Small)
=========================================================
Fixes from v7:
  1. Rotation covers embed/LN layers (-2, -1) not just transformer blocks
  2. Loss evaluation reuses the same model instance (no re-init per step)
  3. Includes both M=10 rotation and all-layers-every-step modes

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_e2e3_consolidated.py --steps 50'
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
        self.ids = sorted(ids); self.M = M
        self.ss = {l: 0 for l in self.ids}
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
        all_n.append(float(np.sqrt(np.mean(d**2))) / std)
    return {"mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "p95": float(np.percentile(all_n, 95)), "max": float(np.max(all_n)),
            "raw_counts": len(all_n)}


def reconstruct_v6(init_w, block_patches, small_patches, block_size):
    w = copy.deepcopy(init_w)
    for p in block_patches:
        _, name, bidx, i8, s = p["layer_id"], p["name"], p["block_idx"], p["int8_data"], p["scale"]
        fp32 = i8.astype(np.float32) * s
        start = bidx * block_size; end = min(start + len(fp32), int(np.prod(w[name].shape)))
        wv = w[name].astype(np.float32).flatten()
        wv[start:end] = fp32[:end-start]
        w[name] = wv.reshape(w[name].shape)
    for p in small_patches:
        _, name, i8, s = p["layer_id"], p["name"], p["int8_data"], p["scale"]
        w[name] = (i8.astype(np.float32) * s).flatten()[:int(np.prod(w[name].shape))].reshape(w[name].shape)
    return w


def compute_loss_pynative(model, eval_batch):
    """Compute loss for the model on a fixed batch. Reuses model instance."""
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    sl = SEQ_LEN
    inp = eval_batch[:, :sl].astype(np.int32)
    input_ids = Tensor(inp, ms.int32)
    input_mask = Tensor(np.ones(inp.shape, dtype=np.int32), ms.int32)
    try:
        output = model(input_ids, input_mask)
    except Exception:
        input_ids = Tensor(eval_batch[:, :sl-1].astype(np.int32), ms.int32)
        input_mask = Tensor(np.ones(input_ids.shape, dtype=np.int32), ms.int32)
        output = model(input_ids, input_mask)
    loss_tensor = output[0] if isinstance(output, tuple) else output
    return float(loss_tensor.asnumpy().flatten()[0])


# ═══════════════ Main ═══════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=524288)
    parser.add_argument("--all-layers", action="store_true",
                       help="Process ALL layers every step (Phase 5 simplified)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Phase 5 E2+E3 v8: Loss + NRMSE  (Steps={args.steps} M={args.M} TopK={args.top_k})")
    print("=" * 70)

    # ── 0. Model analysis ──
    print("\n[0] Model analysis...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)

    # Rotation covers: embed(-2) + final_ln(-1) + 12 transformer blocks = 14 layers total
    rotation_ids = sorted([l for l in layer_map.keys() if l >= -2])
    transformer_ids = [l for l in rotation_ids if l >= 0]
    total_elems = sum(layer_elems[l] for l in rotation_ids)
    print(f"  Rotation: {len(rotation_ids)} layers (embed, final_ln, {len(transformer_ids)} blocks)")
    print(f"  {total_elems/1e6:.0f}M elements ({total_elems*2/1e9:.2f}GB FP16)")

    # ── 1. Eval batch ──
    print("\n[1] Preparing eval batch...")
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=False)
    ds = ds.batch(1, drop_remainder=True)
    it = ds.create_tuple_iterator()
    eval_batch = next(it)[0].asnumpy()

    # ── 2. Oracle training (GRAPH_MODE, sink_size=1) ──
    print(f"\n[2] Oracle training {args.steps} steps (GRAPH_MODE)...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.set_seed(42); ms.common.set_seed(42)

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
    ds_train = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds_train = ds_train.batch(1, drop_remainder=True).take(args.steps)

    snapshots = [(0, get_all_params_np(model2))]
    class SnapCB(ms.Callback):
        def __init__(self): self.cnt = 0
        def on_train_epoch_end(self, rc): self.cnt += 1; snapshots.append((self.cnt, get_all_params_np(model2)))

    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=ds_train, callbacks=[SnapCB()],
                   dataset_sink_mode=True, sink_size=1)
    dt = time.perf_counter() - t0
    print(f"  Done: {dt:.1f}s ({len(snapshots)} snapshots)")

    # ── 3. Oracle loss (reuse SAME model instance for all evals) ──
    print(f"\n[3] Computing oracle loss at checkpoints...")
    loss_model = AutoModel.from_config(cfg)
    oracle_losses = {}
    for step, w in snapshots:
        if step % 5 == 0 or step == 0 or step == args.steps:
            for p in loss_model.trainable_params():
                if p.name in w:
                    p.set_data(Tensor(w[p.name].astype(np.float16), ms.float16))
            oracle_losses[step] = compute_loss_pynative(loss_model, eval_batch)

    for s in sorted(oracle_losses.keys())[:5]:
        print(f"  Step {s:3d}: oracle_loss={oracle_losses[s]:.6f}")
    print(f"  ...")
    for s in sorted(oracle_losses.keys())[-3:]:
        print(f"  Step {s:3d}: oracle_loss={oracle_losses[s]:.6f}")

    init_loss = oracle_losses.get(0, 0)

    # ── 4. I3 Recovery ──
    print(f"\n[4] I3 recovery...")
    w_init = snapshots[0][1]
    ctrl = RotCtrl(rotation_ids, M=args.M)
    pold = PoldStore()
    block_patches, small_patches = [], []
    nrmse_timeline = []
    all_i3_losses = {}

    for step_idx in range(1, len(snapshots)):
        step, true_w = snapshots[step_idx]
        selected = list(rotation_ids) if args.all_layers else ctrl.select()

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

        w_rec = reconstruct_v6(w_init, block_patches, small_patches, args.block_size)
        err = compute_nrmse(w_rec, true_w)
        nrmse_timeline.append({"step": step, **{k: err[k] for k in ["mean","median","p95","max"]}})

        if step % 5 == 0:
            for p in loss_model.trainable_params():
                if p.name in w_rec:
                    p.set_data(Tensor(w_rec[p.name].astype(np.float16), ms.float16))
            i3l = compute_loss_pynative(loss_model, eval_batch)
            all_i3_losses[step] = i3l
            ol = oracle_losses.get(step, 0)
            rel = abs(i3l - ol) / (abs(ol - init_loss) + 1e-10) * 100  # relative to loss change
            print(f"  Step {step:3d}: oracle={ol:+.4f}  i3={i3l:+.4f}  "
                  f"Δrel_vs_loss_change={rel:.1f}%  MedNRMSE={err['median']:.4e}  "
                  f"sel={selected}")

    # ── 5. Final report ──
    final_err = nrmse_timeline[-1]
    for p in loss_model.trainable_params():
        if p.name in w_rec:
            p.set_data(Tensor(w_rec[p.name].astype(np.float16), ms.float16))
    final_i3 = compute_loss_pynative(loss_model, eval_batch)
    final_ol = oracle_losses.get(args.steps, list(oracle_losses.values())[-1])
    rel = abs(final_i3 - final_ol) / (abs(final_ol - init_loss) + 1e-10) * 100

    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    fmb = args.steps * sum(layer_elems[l]*2/1e6 for l in rotation_ids)

    print(f"\n{'='*70}")
    print(f"E2+E3 v8 RESULTS ({args.steps} steps, M={args.M})")
    print(f"{'='*70}")
    print(f"\n  E2 — Recovery Fidelity:")
    print(f"    Median NRMSE: {final_err['median']:.4e}")
    print(f"    P95 NRMSE:    {final_err['p95']:.4e}")
    print(f"    Max NRMSE:    {final_err['max']:.4e}")
    print(f"    Params counted: {final_err.get('raw_counts', '?')}")

    print(f"\n  E3 — Loss Fidelity:")
    print(f"    Init loss:  {init_loss:.6f}")
    print(f"    Final oracle: {final_ol:.6f}")
    print(f"    Final I3:     {final_i3:.6f}")
    print(f"    Loss drop:    {final_ol - init_loss:.4f}")
    print(f"    Rel diff (% of loss drop): {rel:.1f}%")

    print(f"\n  Compression: {block_mb+small_mb:.1f}MB / {fmb:.0f}MB → {fmb/(block_mb+small_mb):.0f}×")

    if final_err["median"] < 0.05 and final_err["max"] < 0.10:
        v = "PASS — NRMSE targets met"
    elif final_err["median"] < 0.05:
        v = "WARN — median OK but max NRMSE exceeds 10%"
    else:
        v = "NEEDS DEBUG"
    print(f"\n  VERDICT: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_e2e3_consolidated.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 E2+E3 v8",
            "config": vars(args),
            "nrmse": {"final": final_err, "timeline": nrmse_timeline},
            "loss": {"init": init_loss, "final_oracle": final_ol, "final_i3": final_i3,
                     "oracle_checkpoints": {str(k): v for k, v in sorted(oracle_losses.items())},
                     "i3_checkpoints": {str(k): v for k, v in sorted(all_i3_losses.items())}},
            "compression": {"ratio": fmb/(block_mb+small_mb) if (block_mb+small_mb) > 0 else 0},
            "verdict": v,
        }, f, indent=2, default=str)
    print(f"  → {out}\n[DONE E2+E3 v8]")


if __name__ == "__main__":
    main()
