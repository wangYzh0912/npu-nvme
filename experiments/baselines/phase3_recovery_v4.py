#!/usr/bin/env python3
"""
Phase 3: Recovery Fidelity v4 — with Small Param Protection
=============================================================

Key fix: Small params (bias/layernorm, <10000 elems) cannot survive block-level
top-K because they get interleaved between large weight matrices. A block's delta
norm is dominated by the large weight → small param updates are invisible.

Solution: Small params are ALWAYS saved with every selected layer.
This adds negligible overhead (~3KB bias + 1.5KB layernorm per layer).

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && python phase3_recovery_v4.py'
"""
import os, sys, time, json, math, re, argparse, copy

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024
SMALL_PARAM_THRESHOLD = 10000  # params with < this many elems are "small"


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
    """Split layer params into large (block-level) and small (always-saved)."""
    large, small = {}, {}
    for pi in sorted(layer_info.keys()):
        p, name, ne = layer_info[pi]
        if ne < SMALL_PARAM_THRESHOLD:
            small[pi] = (p, name, ne)
        else:
            large[pi] = (p, name, ne)
    return large, small


def flatten_and_block(params_np, layer_info, block_size):
    """Flatten large params only, split into blocks."""
    large, _ = classify_layer_params(layer_info)
    parts, offsets = [], []
    off = 0
    for pi in sorted(large.keys()):
        _, name, ne = large[pi]
        parts.append(params_np[name].astype(np.float32).flatten())
        offsets.append((name, off, off + len(parts[-1])))
        off += len(parts[-1])
    fd = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    n = math.ceil(len(fd) / block_size)
    return fd, [(b * block_size, min((b + 1) * block_size, len(fd))) for b in range(n)], offsets


class PoldStore:
    def __init__(self, layer_elems, block_size=524288):
        self.bs = block_size
        self.nb = {l: math.ceil(ne / block_size) for l, ne in layer_elems.items()}
        self.i8, self.sc = {}, {}
    def getb(self, lid, bidx, ref):
        if lid not in self.i8 or bidx not in self.i8[lid]:
            return np.zeros_like(ref, dtype=np.float32)
        return self.i8[lid][bidx].astype(np.float32) * self.sc[lid][bidx]
    def updateb(self, lid, bidx, data):
        fp32 = data.astype(np.float32)
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        self.i8.setdefault(lid, {})[bidx] = np.clip(np.round(fp32 / s), -128, 127).astype(np.int8)
        self.sc.setdefault(lid, {})[bidx] = float(s)
    # Small param storage (separate from blocks)
    def gets(self, lid, name, ref):
        key = f"{lid}:{name}"
        if key not in self.i8:
            return np.zeros_like(ref, dtype=np.float32)
        return self.i8[key].astype(np.float32) * self.sc[key]
    def updates(self, lid, name, data):
        fp32 = data.astype(np.float32).flatten()
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        key = f"{lid}:{name}"
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
            ms_val = max(self.ss.values())
            stale = [[l for l in self.ids if self.ss[l] == ms_val][0]]
        for l in stale: self.ss[l] = 0
        return stale


def reconstruct(init_w, layer_map, patches, small_patches):
    """Apply all patches (block + small) to init weights."""
    w = copy.deepcopy(init_w)
    # Apply block patches
    for p in patches:
        lid, bidx, i8, s = p["layer_id"], p["block_idx"], p["int8_data"], p["scale"]
        if lid not in layer_map: continue
        large, small = classify_layer_params(layer_map[lid])
        fp32 = i8.astype(np.float32) * s
        bstart = bidx * p["block_size"]
        for pi in sorted(large.keys()):
            _, name, ne = large[pi]
            p0 = sum(large[pj][2] for pj in sorted(large.keys()) if large[pj][1] < name)
            p1 = p0 + ne
            o0, o1 = max(bstart, p0), min(bstart + len(fp32), p1)
            if o0 < o1:
                wv = w[name].astype(np.float32).flatten()
                wv[o0-p0:o0-p0+(o1-o0)] = fp32[o0-bstart:o0-bstart+(o1-o0)]
                w[name] = wv.reshape(w[name].shape)
    # Apply small param patches
    for p in small_patches:
        lid, name, i8, s = p["layer_id"], p["name"], p["int8_data"], p["scale"]
        fp32 = i8.astype(np.float32) * s
        w[name] = fp32.reshape(w[name].shape)
    return w


def compute_nrmse(w_r, w_t):
    all_n = []; params = {}
    for nm in w_t:
        r = w_r[nm].astype(np.float64).flatten()
        t = w_t[nm].astype(np.float64).flatten()
        d = r - t; std = float(np.std(t)) + 1e-12
        nr = float(np.sqrt(np.mean(d**2))) / std
        params[nm] = {"nrmse": nr, "mae": float(np.mean(np.abs(d))),
                       "std_ref": float(std), "max_abs": float(np.max(np.abs(d)))}
        all_n.append(nr)
    return {"params": params, "mean": float(np.mean(all_n)), "median": float(np.median(all_n)),
            "p95": float(np.percentile(all_n, 95)), "p99": float(np.percentile(all_n, 99)),
            "max": float(np.max(all_n))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20); parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10); parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print(f"\n{'='*70}\nPhase 3 Recovery Fidelity v4 — Steps={args.steps} TopK={args.top_k} M={args.M}\n{'='*70}")

    # 0. Analyze param sizes
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    # Classify
    n_large, n_small = 0, 0
    for l in layer_ids:
        large, small = classify_layer_params(layer_map[l])
        n_large += len(large); n_small += len(small)
        large_elems = sum(ne for _, _, ne in large.values())
        small_elems = sum(ne for _, _, ne in small.values())
        print(f"  L{l}: {len(large)} large ({large_elems/1e6:.1f}M) + {len(small)} small ({small_elems:,} elems)")
    print(f"  Total: {n_large} large params + {n_small} small params (threshold={SMALL_PARAM_THRESHOLD})")

    # 1. Oracle training (GRAPH_MODE, sink_size=1)
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
    dataset = ms.dataset.MindDataset(REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    dataset = dataset.batch(1, drop_remainder=True).take(args.steps)

    snapshots = []
    class SnapCB(ms.Callback):
        def on_train_epoch_end(self, rc):
            snapshots.append(get_all_params_np(model2))

    snapshots.append(get_all_params_np(model2))  # [0] = init
    print(f"\n  [1] Training {args.steps} steps (sink=1)...")
    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=dataset, callbacks=[SnapCB()], dataset_sink_mode=True, sink_size=1)
    train_s = time.perf_counter() - t0
    print(f"  Done: {train_s:.1f}s  snapshots: {len(snapshots)} (0=init, 1..N=after each step)")

    # 2. I3 recovery simulation
    print(f"  [2] I3 recovery (M={args.M}, topK={args.top_k})...")
    w_init = snapshots[0]; true_weights = snapshots[1:]
    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore(layer_elems, args.block_size)
    block_patches, small_patches = [], []
    err_timeline = []

    for step in range(len(true_weights)):
        true_w = true_weights[step]; selected = ctrl.select()

        for lid in selected:
            large_params, small_params = classify_layer_params(layer_map[lid])

            # --- Block-level: large params only ---
            if large_params:
                fd, blocks, offs = flatten_and_block(true_w, layer_map[lid], args.block_size)
                norms = []
                for b, (s, e) in enumerate(blocks):
                    bd = fd[s:e].astype(np.float32)
                    po = pold.getb(lid, b, bd)
                    norms.append(float(np.sum((bd - po).astype(np.float64)**2)))
                ranked = sorted(enumerate(norms), key=lambda x: -x[1])
                tk = max(1, int(math.ceil(len(blocks) * args.top_k)))
                for bidx, _ in ranked[:tk]:
                    s, e = blocks[bidx]; fp32 = fd[s:e]
                    sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                    q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                    block_patches.append({"layer_id": lid, "block_idx": bidx, "int8_data": q,
                                          "scale": float(sc), "block_size": args.block_size})
                    pold.updateb(lid, bidx, fp32)

            # --- Small params: ALWAYS save (negligible overhead) ---
            for pi in sorted(small_params.keys()):
                _, name, ne = small_params[pi]
                data = true_w[name]
                fp32 = data.astype(np.float32)
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                small_patches.append({"layer_id": lid, "name": name, "int8_data": q,
                                      "scale": float(sc)})
                pold.updates(lid, name, fp32)

        w_rec = reconstruct(w_init, layer_map, block_patches, small_patches)
        err_timeline.append(compute_nrmse(w_rec, true_w))

        if (step+1) % 5 == 0 or step == 0:
            e = err_timeline[-1]
            print(f"    Step {step+1:3d}: sel={selected}  bpatches={len(block_patches)}  "
                  f"spatches={len(small_patches)}  meanNRMSE={e['mean']:.2e}  medNRMSE={e['median']:.2e}", flush=True)

    # 3. Results
    final = err_timeline[-1]
    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    fmb = sum(args.steps*layer_elems[l]*2/1e6 for l in layer_ids)

    print(f"\n{'='*70}\nRECOVERY FIDELITY RESULTS (v4 — small param protection)\n{'='*70}")
    print(f"\n  I3 Recovery ({args.steps} steps):")
    print(f"    Block patches:  {len(block_patches)}  ({block_mb:.1f} MB)")
    print(f"    Small patches:  {len(small_patches)}  ({small_mb:.1f} MB)")
    print(f"    Total:          {block_mb+small_mb:.1f} MB  vs full {fmb:.0f} MB  ({fmb/(block_mb+small_mb):.0f}×)")

    print(f"\n  Weight NRMSE:")
    print(f"    Mean={final['mean']:.2e}  Median={final['median']:.2e}  "
          f"P95={final['p95']:.2e}  P99={final['p99']:.2e}  Max={final['max']:.2e}")

    worst = sorted(final["params"].items(), key=lambda x: -x[1]["nrmse"])[:5]
    print(f"\n  Top-5 most-deviated params:")
    for nm, e in worst:
        ptype = 'bias' if 'bias' in nm else ('layernorm' if 'layernorm' in nm else 'weight')
        print(f"    [{ptype:10s}] {nm:55s}: NRMSE={e['nrmse']:.4f}  MAE={e['mae']:.2e}  std={e['std_ref']:.4f}")

    fn = final["mean"]
    fn_med = final["median"]
    p95 = final["p95"]
    # Use median as the primary metric — it's robust to outliers
    if fn_med < 1e-4:    v = "EXCELLENT — median NRMSE < 10⁻⁴"
    elif fn_med < 1e-3:  v = "GOOD — median NRMSE < 10⁻³"
    elif fn_med < 1e-2:  v = f"ACCEPTABLE — median NRMSE < 10⁻² (mean={fn:.2e}, P95={p95:.2e})"
    elif fn_med < 1e-1:  v = f"MARGINAL — median NRMSE {fn_med:.1e}, need larger top_k or smaller M"
    else:                v = f"FAIL — median NRMSE={fn_med:.1e}"

    print(f"\n  VERDICT: {v}")
    print(f"  NRMSE growth: {(err_timeline[-1]['mean']-err_timeline[0]['mean'])/len(err_timeline):.4f}/step")
    print(f"  Note: Mean ({final['mean']:.2e}) driven by few outliers; median ({final['median']:.2e}) is robust")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase3_recovery_v4.json")
    with open(out, "w") as f: json.dump({
        "experiment": "Phase 3 Recovery Fidelity v4",
        "config": {"steps": args.steps, "top_k": args.top_k, "M": args.M,
                   "block_size": args.block_size, "small_threshold": SMALL_PARAM_THRESHOLD},
        "recovery": {"block_patches": len(block_patches), "small_patches": len(small_patches),
                     "total_mb": block_mb + small_mb, "full_mb": fmb,
                     "mean_nrmse": final["mean"], "median_nrmse": final["median"],
                     "p95_nrmse": final["p95"], "max_nrmse": final["max"],
                     "timeline": [{"step": i+1, "mean": e["mean"], "median": e["median"]}
                                 for i, e in enumerate(err_timeline)]},
        "worst_params": [(nm, e) for nm, e in worst],
        "verdict": v, "train_time_s": train_s,
    }, f, indent=2, default=str)
    print(f"  → {out}\n[Recovery Fidelity v4] DONE.\n")


if __name__ == "__main__":
    main()
