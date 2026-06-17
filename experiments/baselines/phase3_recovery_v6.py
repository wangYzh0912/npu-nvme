#!/usr/bin/env python3
"""
Phase 3 Recovery v6 — Per-Param Block Alignment
================================================

Root cause of outlier NRMSE: concatenating params within a layer causes
block boundaries to misalign with parameter boundaries.

Fix: Each param is independently padded to block_size boundary.
  - Small params (<10K elems): stored directly (always saved with layer)
  - Large params: split into N = ceil(nelems/block_size) blocks per param
  - Each block belongs to exactly ONE parameter
  - No cross-param contamination

Block mapping: (layer_id, param_name, block_idx_in_param) → INT8 data

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase3_recovery_v6.py --steps 20 --M 10 --top_k 0.10'
"""
import os, sys, time, json, math, re, argparse, copy

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; SMALL_THRESHOLD = 10000


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
    """Per-param P_old with block-level quantization."""
    def __init__(self):
        self.i8, self.sc = {}, {}
    def get(self, lid, name, bidx, ref):
        key = f'{lid}:{name}:{bidx}'
        if key not in self.i8:
            return np.zeros_like(ref, dtype=np.float32)
        return self.i8[key].astype(np.float32) * self.sc[key]
    def put(self, lid, name, bidx, data):
        fp32 = data.astype(np.float32).flatten()
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        key = f'{lid}:{name}:{bidx}'
        self.i8[key] = np.clip(np.round(fp32 / s), -128, 127).astype(np.int8)
        self.sc[key] = float(s)


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
        d = r - t
        std = float(np.std(t)) + 1e-12
        nr = float(np.sqrt(np.mean(d**2))) / std
        params[nm] = {"nrmse": nr, "mae": float(np.mean(np.abs(d))),
                       "max_abs": float(np.max(np.abs(d))), "std": float(std)}
        all_n.append(nr)
    return {"params": params, "mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "p95": float(np.percentile(all_n, 95)), "max": float(np.max(all_n)),
            "p99": float(np.percentile(all_n, 99))}


def reconstruct_v6(init_w, layer_map, block_patches, small_patches, block_size):
    w = copy.deepcopy(init_w)
    # Apply block patches (per-param, per-block)
    for p in block_patches:
        lid, name, bidx, i8, s = p["layer_id"], p["name"], p["block_idx"], p["int8_data"], p["scale"]
        fp32 = i8.astype(np.float32) * s
        start = bidx * block_size
        end = start + len(fp32)
        wv = w[name].astype(np.float32).flatten()
        if end > len(wv):
            end = len(wv)
        wv[start:end] = fp32[:end-start]
        w[name] = wv.reshape(w[name].shape)
    # Apply small param patches
    for p in small_patches:
        lid, name, i8, s = p["layer_id"], p["name"], p["int8_data"], p["scale"]
        w[name] = (i8.astype(np.float32) * s).flatten()[:int(np.prod(w[name].shape))].reshape(w[name].shape)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print(f"\n{'='*70}\nPhase 3 Recovery v6 — Per-Param Block Alignment")
    print(f"  Steps={args.steps}  M={args.M}  TopK={args.top_k}\n{'='*70}")

    # ── 0. Analyze ──
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    for l in layer_ids[:3]:
        large, small = classify_layer_params(layer_map[l])
        n_lb = sum(math.ceil(ne/args.block_size) for _, _, ne in large.values())
        print(f"  L{l}: {len(large)} large params → {n_lb} blocks + {len(small)} small")

    # ── 1. Oracle training ──
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
    print(f"  Done: {time.perf_counter()-t0:.1f}s")

    # ── 2. I3 recovery (per-param blocks) ──
    print(f"\n  [2] I3 recovery (per-param blocks)...")
    w_init = snapshots[0]; true_weights = snapshots[1:]
    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore()
    block_patches, small_patches = [], []
    err_timeline = []

    for step in range(len(true_weights)):
        true_w = true_weights[step]; selected = ctrl.select()

        for lid in selected:
            large, small = classify_layer_params(layer_map[lid])

            # Per-param block processing: compute delta per param-block
            all_param_blocks = []  # [(lid, name, bidx, block_data_fp32, start, end)]

            for pi in sorted(large.keys()):
                _, name, ne = large[pi]
                fp32 = true_w[name].astype(np.float32).flatten()
                nblk = math.ceil(ne / args.block_size)

                for b in range(nblk):
                    s = b * args.block_size
                    e = min(s + args.block_size, ne)
                    bd = fp32[s:e]
                    po = pold.get(lid, name, b, bd)
                    delta = bd - po
                    dn = float(np.sum(delta.astype(np.float64)**2))
                    all_param_blocks.append((lid, name, b, bd, dn))

            # Top-K across all param-blocks in this layer
            ranked = sorted(all_param_blocks, key=lambda x: -x[4])
            tk = max(1, int(math.ceil(len(ranked) * args.top_k)))
            for lid_p, name_p, bidx_p, bd_p, dn_p in ranked[:tk]:
                sc = max(float(np.max(np.abs(bd_p)))/127.0, 1e-10)
                q = np.clip(np.round(bd_p/sc), -128, 127).astype(np.int8)
                block_patches.append({"layer_id": lid_p, "name": name_p, "block_idx": bidx_p,
                                       "int8_data": q, "scale": float(sc),
                                       "delta_norm": dn_p})
                pold.put(lid_p, name_p, bidx_p, bd_p)

            # Small params: always save
            for pi in sorted(small.keys()):
                _, name, ne = small[pi]
                fp32 = true_w[name].astype(np.float32)
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                small_patches.append({"layer_id": lid, "name": name, "int8_data": q,
                                       "scale": float(sc)})
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

    print(f"\n{'='*70}\nRECOVERY FIDELITY v6 — Per-Param Blocks\n{'='*70}")
    print(f"\n  Final step (step {args.steps}):")
    print(f"    Mean NRMSE:   {final['mean']:.4e}")
    print(f"    Median NRMSE: {final['median']:.4e}")
    print(f"    P95 NRMSE:    {final['p95']:.4e}")
    print(f"    P99 NRMSE:    {final['p99']:.4e}")
    print(f"    Max NRMSE:    {final['max']:.4e}")

    worst = sorted(final["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 worst (per-param blocks, no cross-param contamination):")
    for nm, e in worst:
        ptype = 'bias' if 'bias' in nm else ('layernorm' if 'layernorm' in nm else 'weight')
        print(f"    [{ptype:10s}] {nm:55s}: NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}")

    print(f"\n  Compression:")
    print(f"    Block patches: {len(block_patches)} ({block_mb:.1f}MB) | Small: {len(small_patches)} ({small_mb:.1f}MB)")
    print(f"    Total: {block_mb+small_mb:.1f}MB vs Full: {fmb:.0f}MB → {fmb/(block_mb+small_mb):.0f}x")

    med = final["median"]
    if med < 1e-4:    v = "EXCELLENT"
    elif med < 1e-3:  v = "GOOD"
    elif med < 1e-2:  v = "ACCEPTABLE"
    elif med < 5e-2:  v = f"ADECUATE — median {med*100:.1f}%"
    else:             v = "FAIL"

    print(f"\n  VERDICT: {v}")
    print(f"  NRMSE growth: {(err_timeline[-1]['median']-err_timeline[0]['median'])/len(err_timeline):.6f}/step")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase3_recovery_v6.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 3 Recovery v6",
            "config": {"steps": args.steps, "top_k": args.top_k, "M": args.M,
                       "block_size": args.block_size, "approach": "per-param blocks"},
            "final": {"mean": final["mean"], "median": final["median"],
                      "p95": final["p95"], "p99": final["p99"], "max": final["max"]},
            "compression": {"block_patches": len(block_patches), "small_patches": len(small_patches),
                            "total_mb": block_mb+small_mb, "ratio": fmb/(block_mb+small_mb)},
            "timeline": [{"step": i+1, "mean": e["mean"], "median": e["median"], "p95": e["p95"]}
                        for i, e in enumerate(err_timeline)],
            "verdict": v,
        }, f, indent=2, default=str)
    print(f"  → {out}\n[Recovery v6] DONE.\n")


if __name__ == "__main__":
    main()
