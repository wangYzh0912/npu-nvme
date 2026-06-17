#!/usr/bin/env python3
"""
Phase 3: Recovery Fidelity v5 — Direct Loss Comparison
=======================================================

The definitive test: load recovered weights into model, run forward pass
on the SAME input_ids, compare loss against oracle.

Strategy:
  - Train N steps in GRAPH_MODE, storing input_ids + weights + loss per step
  - Simulate I3 recovery from init weights + incremental patches
  - For each step k, load recovered weights, run forward(input_ids[k])
  - Compare: oracle_loss[k] vs recovered_loss[k]

Small param protection: bias/layernorm (<10000 elems) always saved
with every selected layer (negligible overhead: ~1KB per param).

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase3_recovery_v5.py --steps 20 --M 10 --top_k 0.10'
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
SMALL_THRESHOLD = 10000


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


def flatten_large_and_block(params_np, layer_info, block_size):
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
    return fd, [(b*block_size, min((b+1)*block_size, len(fd))) for b in range(n)], offsets


class PoldStore:
    def __init__(self, bs=524288):
        self.bs = bs; self.i8, self.sc = {}, {}
    def get_block(self, lid, bidx, ref):
        if lid not in self.i8 or bidx not in self.i8[lid]:
            return np.zeros_like(ref, dtype=np.float32)
        return self.i8[lid][bidx].astype(np.float32) * self.sc[lid][bidx]
    def put_block(self, lid, bidx, data):
        fp32 = data.astype(np.float32)
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        self.i8.setdefault(lid, {})[bidx] = np.clip(np.round(fp32/s), -128, 127).astype(np.int8)
        self.sc.setdefault(lid, {})[bidx] = float(s)
    def get_small(self, lid, name, ref):
        k = f"{lid}:{name}"
        if k not in self.i8: return np.zeros_like(ref, dtype=np.float32)
        return self.i8[k].astype(np.float32) * self.sc[k]
    def put_small(self, lid, name, data):
        fp32 = data.astype(np.float32).flatten()
        s = max(float(np.max(np.abs(fp32))) / 127.0, 1e-10)
        k = f"{lid}:{name}"
        self.i8[k] = np.clip(np.round(fp32/s), -128, 127).astype(np.int8)
        self.sc[k] = float(s)


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


def reconstruct(init_w, layer_map, block_patches, small_patches):
    w = copy.deepcopy(init_w)
    for p in block_patches:
        lid, bidx, i8, s = p["layer_id"], p["block_idx"], p["int8_data"], p["scale"]
        if lid not in layer_map: continue
        large, _ = classify_layer_params(layer_map[lid])
        fp32 = i8.astype(np.float32) * s
        bstart = bidx * p["block_size"]
        for pi in sorted(large.keys()):
            _, name, ne = large[pi]
            p0 = sum(large[pj][2] for pj in sorted(large.keys()) if large[pj][1] < name)
            p1 = p0 + ne
            o0, o1 = max(bstart, p0), min(bstart+len(fp32), p1)
            if o0 < o1:
                wv = w[name].astype(np.float32).flatten()
                wv[o0-p0:o0-p0+(o1-o0)] = fp32[o0-bstart:o0-bstart+(o1-o0)]
                w[name] = wv.reshape(w[name].shape)
    for p in small_patches:
        lid, name, i8, s = p["layer_id"], p["name"], p["int8_data"], p["scale"]
        w[name] = (i8.astype(np.float32) * s).reshape(w[name].shape)
    return w


def compute_weight_errors(w_r, w_t):
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
            "p95": float(np.percentile(all_n, 95)), "max": float(np.max(all_n))}


def load_weights_into_model(model, weights_np):
    for p in model.trainable_params():
        if p.name in weights_np:
            p.set_data(Tensor(weights_np[p.name].astype(np.float32)))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Phase 3 Recovery v5 — Direct Loss Comparison")
    print(f"  Steps={args.steps}  M={args.M}  TopK={args.top_k}")
    print(f"  Small param threshold: {SMALL_THRESHOLD} elems")
    print("=" * 70)

    # ── 1. Oracle: train step-by-step in GRAPH_MODE (sink_size=1), store weights+loss ──
    # PYNATIVE has the seq_length mismatch bug. Use GRAPH_MODE instead.
    print("\n  [1/3] Oracle training (GRAPH_MODE, sink_size=1)...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    params, layer_map, layer_elems = inspect_model_layers(model)
    layer_ids = [l for l in sorted(layer_map.keys()) if l >= 0]

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    class TrainStepCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad(); self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            return ops.Depend()(loss, self.opt(grads))

    cell = TrainStepCell(model, opt)
    ms_model = ms.Model(cell)

    dataset = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    dataset = dataset.batch(1, drop_remainder=True).take(args.steps)

    # Collect: loss + weights after each sink epoch (step)
    oracle_losses = []
    oracle_snapshots = []
    class OracleCB(ms.Callback):
        def on_train_epoch_end(self, rc):
            oracle_snapshots.append(get_all_params_np(model))

    # Init snapshot
    oracle_snapshots.append(get_all_params_np(model))

    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=dataset, callbacks=[OracleCB()],
                   dataset_sink_mode=True, sink_size=1)
    oracle_s = time.perf_counter() - t0

    # oracle_snapshots: [0]=init, [1]=post_step1, ..., [N]=post_stepN
    # We need input_ids from the dataset. Since GRAPH doesn't expose them,
    # we re-create the dataset iterator to get them.
    dataset2 = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    dataset2 = dataset2.batch(1, drop_remainder=True).take(args.steps)
    data_iter = dataset2.create_dict_iterator()
    input_ids_list = [next(data_iter)["input_ids"].asnumpy().copy() for _ in range(args.steps)]

    print(f"  Oracle done: {oracle_s:.1f}s  snapshots={len(oracle_snapshots)}  inputs={len(input_ids_list)}")

    # ── 2. I3 recovery ──
    print(f"\n  [2/3] I3 recovery simulation (M={args.M}, topK={args.top_k})...")

    # Init weights = oracle_snapshots[0] (before training)
    w_init = oracle_snapshots[0]

    ctrl = RotCtrl(layer_ids, M=args.M)
    pold = PoldStore(args.block_size)
    block_patches, small_patches = [], []

    recovered_snapshots = []

    true_weights = oracle_snapshots[1:]  # [0]=post_step1, [1]=post_step2, ...

    for step in range(len(true_weights)):
        true_w = true_weights[step]
        selected = ctrl.select()

        for lid in selected:
            large, small = classify_layer_params(layer_map[lid])

            if large:
                fd, blocks, offs = flatten_large_and_block(true_w, layer_map[lid], args.block_size)
                norms = []
                for b, (s, e) in enumerate(blocks):
                    bd = fd[s:e].astype(np.float32)
                    po = pold.get_block(lid, b, bd)
                    norms.append(float(np.sum((bd - po).astype(np.float64)**2)))
                ranked = sorted(enumerate(norms), key=lambda x: -x[1])
                tk = max(1, int(math.ceil(len(blocks) * args.top_k)))
                for bidx, _ in ranked[:tk]:
                    s, e = blocks[bidx]; fp32 = fd[s:e]
                    sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                    q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                    block_patches.append({"layer_id": lid, "block_idx": bidx, "int8_data": q,
                                          "scale": float(sc), "block_size": args.block_size})
                    pold.put_block(lid, bidx, fp32)

            for pi in sorted(small.keys()):
                _, name, ne = small[pi]
                data = true_w[name]
                sc = max(float(np.max(np.abs(data.astype(np.float32))))/127.0, 1e-10)
                q = np.clip(np.round(data.astype(np.float32)/sc), -128, 127).astype(np.int8)
                small_patches.append({"layer_id": lid, "name": name, "int8_data": q,
                                      "scale": float(sc)})
                pold.put_small(lid, name, data)

        w_rec = reconstruct(w_init, layer_map, block_patches, small_patches)
        recovered_snapshots.append(w_rec)

        if (step + 1) % 5 == 0:
            print(f"    I3 step {step+1:3d}: sel={selected}  "
                  f"bpatches={len(block_patches)}  spatches={len(small_patches)}", flush=True)

    # ── 4. Direct loss comparison (GRAPH_MODE for forward, separate Cell) ──
    print(f"\n  [3/3] Direct loss comparison (GRAPH_MODE forward)...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model_eval = AutoModel.from_config(cfg)

    # Simple forward cell
    class ForwardCell(nn.Cell):
        def __init__(self, net):
            super().__init__(auto_prefix=False)
            self.net = net
        def construct(self, ids):
            return self.net(ids)

    fwd_cell = ForwardCell(model_eval)

    recovered_losses = []
    oracle_losses_computed = []

    for i in range(len(true_weights)):
        rec_w = recovered_snapshots[i]
        true_w = true_weights[i]
        inp = Tensor(input_ids_list[i])

        # Set recovered weights → forward
        load_weights_into_model(model_eval, rec_w)
        recovered_losses.append(float(fwd_cell(inp).asnumpy()))

        # Set true weights → forward
        load_weights_into_model(model_eval, true_w)
        oracle_losses_computed.append(float(fwd_cell(inp).asnumpy()))

    oracle_losses = oracle_losses_computed

    # ── 4. Analysis ──
    oracle_losses = [d["loss"] for d in oracle_data]
    loss_diffs = [abs(r - o) for r, o in zip(recovered_losses, oracle_losses)]
    loss_ratios = [d / max(o, 1e-10) for d, o in zip(loss_diffs, oracle_losses)]

    final_werr = compute_weight_errors(recovered_snapshots[-1], oracle_data[-1]["weights"])

    block_mb = sum(p["int8_data"].nbytes+4 for p in block_patches)/1e6
    small_mb = sum(p["int8_data"].nbytes+4 for p in small_patches)/1e6
    fmb = sum(args.steps*layer_elems[l]*2/1e6 for l in layer_ids)

    # ── Print ──
    print(f"\n{'='*70}")
    print("RECOVERY FIDELITY RESULTS (v5 — Direct Loss)")
    print("=" * 70)

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ Step │ Oracle Loss │ Recovered Loss │  Δ Loss  │ Δ/Loss │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    for i in range(0, len(oracle_losses), max(1, len(oracle_losses)//10)):
        print(f"  │ {i+1:4d} │  {oracle_losses[i]:9.5f}  │   {recovered_losses[i]:10.5f}   │ "
              f"{loss_diffs[i]:8.5f} │ {loss_ratios[i]:7.5f} │")
    # Last step
    i = len(oracle_losses) - 1
    print(f"  │ {i+1:4d} │  {oracle_losses[i]:9.5f}  │   {recovered_losses[i]:10.5f}   │ "
          f"{loss_diffs[i]:8.5f} │ {loss_ratios[i]:7.5f} │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    print(f"\n  Loss fidelity:")
    print(f"    Mean |Δloss|:            {np.mean(loss_diffs):.5f}")
    print(f"    Max |Δloss|:             {np.max(loss_diffs):.5f}")
    print(f"    Oracle loss Δ:           {oracle_losses[0]-oracle_losses[-1]:.5f}")
    print(f"    Recovery Δ / Oracle Δ:   {np.mean(loss_diffs)/(oracle_losses[0]-oracle_losses[-1]+1e-10):.4f}")

    print(f"\n  Compression:")
    print(f"    Block patches: {len(block_patches)} ({block_mb:.1f}MB)")
    print(f"    Small patches: {len(small_patches)} ({small_mb:.1f}MB)")
    print(f"    Total: {block_mb+small_mb:.1f}MB vs Full: {fmb:.0f}MB → {fmb/(block_mb+small_mb):.0f}×")

    print(f"\n  Weight NRMSE (final step):")
    print(f"    Mean={final_werr['mean']:.2e}  Median={final_werr['median']:.2e}  "
          f"P95={final_werr['p95']:.2e}  Max={final_werr['max']:.2e}")

    # Verdict based on DIRECT LOSS
    mean_ld = np.mean(loss_diffs)
    oracle_delta = oracle_losses[0] - oracle_losses[-1]
    rel_impact = mean_ld / (oracle_delta + 1e-10)

    print(f"\n  VERDICT:")
    if rel_impact < 0.01:
        v = f"EXCELLENT — loss deviation <1% of oracle Δloss"
    elif rel_impact < 0.05:
        v = f"GOOD — loss deviation {rel_impact:.1%} of oracle Δloss (<5%)"
    elif rel_impact < 0.20:
        v = f"ACCEPTABLE — loss deviation {rel_impact:.1%} of oracle Δloss (<20%)"
    elif rel_impact < 1.0:
        v = f"MARGINAL — loss deviation {rel_impact:.1%} of oracle Δloss"
    else:
        v = f"FAIL — recovery loss exceeds oracle improvement"

    # Also check: is recovered_loss decreasing?
    if recovered_losses[-1] < recovered_losses[0]:
        v += " | Loss trend CORRECT (decreasing)"
    else:
        v += " | Loss trend WRONG (not decreasing)"

    print(f"    {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase3_recovery_v5.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 3 Recovery v5",
            "config": {"steps": args.steps, "top_k": args.top_k, "M": args.M,
                       "block_size": args.block_size, "small_threshold": SMALL_THRESHOLD},
            "oracle": {"losses": oracle_losses, "first": oracle_losses[0], "last": oracle_losses[-1],
                       "delta": oracle_losses[0] - oracle_losses[-1]},
            "recovery": {"losses": recovered_losses,
                         "mean_abs_loss_diff": float(np.mean(loss_diffs)),
                         "max_abs_loss_diff": float(np.max(loss_diffs)),
                         "relative_impact": float(rel_impact),
                         "block_patches": len(block_patches),
                         "small_patches": len(small_patches),
                         "total_mb": block_mb + small_mb,
                         "full_mb": fmb,
                         "compression_ratio": fmb / (block_mb + small_mb),
                         "weight_nrmse_mean": final_werr["mean"],
                         "weight_nrmse_median": final_werr["median"]},
            "verdict": v,
            "train_time_s": oracle_s,
        }, f, indent=2, default=str)
    print(f"\n  → {out}")
    print("[Recovery Fidelity v5] DONE.\n")


if __name__ == "__main__":
    main()
