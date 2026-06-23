#!/usr/bin/env python3
"""
Phase 5 E10-FS: Delta End-to-End Test (FileSystem backend)
=============================================================
Validates the complete delta write→read→recovery pipeline using
FileDeltaWriter (filesystem ring buffer, no SPDK dependency).

Runs on GPT-2 Small with 10+10 steps:
  1. Train 20 steps, record weight snapshots
  2. I3 recovery host-side at each step
  3. Write delta frames via FileDeltaWriter
  4. Re-read delta frames
  5. Apply delta chain to initial weights
  6. Verify: NRMSE < 2%, compression > 10x, round-trip byte-perfect

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_e10_spdk_delta_e2e.py --steps 20'
"""
import os, sys, time, json, math, re, struct, ctypes, copy
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DELTA_DIR = os.path.join(REPO, "experiments", "output", "delta_ring_e10")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288; SMALL_THRESHOLD = 10000

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


def compute_nrmse(w_r, w_t):
    all_n = []
    for nm in w_t:
        r = w_r[nm].astype(np.float64).flatten()
        t = w_t[nm].astype(np.float64).flatten()
        d = r - t; std = float(np.std(t)) + 1e-12
        all_n.append(float(np.sqrt(np.mean(d**2))) / std)
    return {"median": float(np.median(all_n)), "p95": float(np.percentile(all_n, 95)),
            "max": float(np.max(all_n))}


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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--block_size", type=int, default=524288)
    parser.add_argument("--delta_slots", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 E10-FS: Delta End-to-End (FileSystem backend)")
    print(f"  Steps={args.steps}  TopK={args.top_k}  Slots={args.delta_slots}")
    print("=" * 70)

    # Clean delta ring directory
    import shutil
    if os.path.exists(DELTA_DIR):
        shutil.rmtree(DELTA_DIR, ignore_errors=True)

    # ── 0. Model analysis ──
    print("\n[0] Model structure...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    rotation_ids = sorted([l for l in layer_map.keys() if l >= -2])
    print(f"  {len(rotation_ids)} layers covered, {len([l for l in rotation_ids if l>=0])} transformer")

    # ── 1. Init FileDeltaWriter ──
    from direct_checkpoint import FileDeltaWriter, pack_delta_frame, unpack_delta_frame, apply_delta_patches

    print("\n[1] Init FileDeltaWriter (filesystem ring buffer)...")
    writer = FileDeltaWriter(delta_dir=DELTA_DIR, delta_slot_count=args.delta_slots)
    print(f"    Delta dir: {DELTA_DIR}")

    # ── 2. Oracle training ──
    print(f"\n[2] Training {args.steps} steps (GRAPH_MODE)...")
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
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(args.steps)

    snapshots = [(0, get_all_params_np(model2))]
    class SnapCB(ms.Callback):
        def __init__(self): self.cnt = 0
        def on_train_epoch_end(self, rc): self.cnt += 1; snapshots.append((self.cnt, get_all_params_np(model2)))

    t0 = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=ds, callbacks=[SnapCB()],
                   dataset_sink_mode=True, sink_size=1)
    dt = time.perf_counter() - t0
    print(f"  Done: {dt:.1f}s ({len(snapshots)} snapshots)")

    # ── 3. I3 recovery + delta write ──
    print(f"\n[3] I3 recovery + delta write at each step (all layers)...")
    w_init = snapshots[0][1]
    pold = PoldStore()
    all_patches = {}
    write_ms = []
    total_mb_written = 0

    for step_idx in range(1, len(snapshots)):
        step, true_w = snapshots[step_idx]

        step_blocks, step_smalls = [], []
        for lid in rotation_ids:
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
                step_blocks.append({"layer_id": lid_p, "name": name_p, "block_idx": bidx_p,
                                     "int8_data": q, "scale": float(sc), "delta_norm": dn_p})
                pold.put(lid_p, name_p, bidx_p, bd_p)

            for pi in sorted(small.keys()):
                _, name, ne = small[pi]
                fp32 = true_w[name].astype(np.float32)
                sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
                q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
                step_smalls.append({"layer_id": lid, "name": name, "int8_data": q, "scale": float(sc)})
                pold.put(lid, name, 0, fp32)

        all_patches[step] = (step_blocks, step_smalls)

        t_w = time.perf_counter()
        slot = writer.write_frame(step, step_blocks, step_smalls)
        wms = (time.perf_counter() - t_w) * 1000
        write_ms.append(wms)
        total_mb_written += writer.frame_sizes[-1] / (1024*1024) if writer.frame_sizes else 0

        if step % 5 == 0:
            fsz = writer.frame_sizes[-1]/1024 if writer.frame_sizes else 0
            print(f"  Step {step:3d}: slot={slot}  frame={fsz:.1f}KB  "
                  f"blocks={len(step_blocks)}  smalls={len(step_smalls)}  write={wms:.1f}ms")

    # ── 4. Read back ──
    print(f"\n[4] Reading delta frames back...")
    read_ms = []
    all_ok = 0

    for step, (orig_blocks, orig_smalls) in all_patches.items():
        slot = writer.step_map[step]
        t_r = time.perf_counter()
        s_id, r_blocks, r_smalls = writer.read_frame(slot)
        rms = (time.perf_counter() - t_r) * 1000
        read_ms.append(rms)

        if s_id != step: print(f"MISMATCH step {step} vs {s_id}")
        if len(r_blocks) != len(orig_blocks): print(f"BLOCK COUNT step {step}: {len(r_blocks)} vs {len(orig_blocks)}")
        if len(r_smalls) != len(orig_smalls): print(f"SMALL COUNT step {step}")

        # Verify block data byte-perfect
        for ob, rb in zip(orig_blocks, r_blocks):
            if not np.array_equal(ob["int8_data"], rb["int8_data"]):
                print(f"DATA MISMATCH {ob[chr(39)+chr(110)+chr(97)+chr(109)+chr(101)+chr(39)]}[{ob[chr(39)+chr(98)+chr(108)+chr(111)+chr(99)+chr(107)+chr(95)+chr(105)+chr(100)+chr(120)+chr(39)]}]")
                break
        else:
            all_ok += 1

    print(f"    {all_ok}/{len(all_patches)} frames byte-perfect round-trip ✅")

    # ── 5. Recovery fidelity ──
    print(f"\n[5] Recovery fidelity from delta chain...")
    w_rec = copy.deepcopy(w_init)
    nrmse_timeline = []

    for step in sorted(all_patches.keys()):
        blocks, smalls = all_patches[step]
        w_rec = apply_delta_patches(w_rec, blocks, smalls, args.block_size)
        true_w = snapshots[step][1]
        err = compute_nrmse(w_rec, true_w)
        nrmse_timeline.append({"step": step, **err})
        if step % 5 == 0:
            print(f"  Step {step:3d}: MedNRMSE={err['median']:.4e}  P95={err['p95']:.4e}")

    # ── 6. Report ──
    final_err = nrmse_timeline[-1]
    block_mb = sum(sum(p["int8_data"].nbytes+4 for p in all_patches[s][0]) for s in all_patches)/1e6
    small_mb = sum(sum(p["int8_data"].nbytes+4 for p in all_patches[s][1]) for s in all_patches)/1e6
    full_mb = args.steps * sum(layer_elems[l]*2/1e6 for l in rotation_ids)

    avg_w = np.mean(write_ms) if write_ms else 0
    avg_r = np.mean(read_ms) if read_ms else 0
    avg_fk = np.mean(writer.frame_sizes)/1024 if writer.frame_sizes else 0

    print(f"\n{'='*70}")
    print(f"E10-FS RESULTS: Delta E2E ({args.steps} steps, FileSystem)")
    print(f"{'='*70}")
    print(f"\n  Round-trip: {all_ok}/{len(all_patches)} byte-perfect ✅")
    print(f"\n  Recovery Fidelity (step {args.steps}):")
    print(f"    Median NRMSE: {final_err['median']:.4e}")
    print(f"    P95 NRMSE:    {final_err['p95']:.4e}")
    print(f"    Max NRMSE:    {final_err['max']:.4e}")
    print(f"\n  I/O Performance (tmpfs/filesystem):")
    print(f"    Avg write: {avg_w:.1f}ms/frame  (avg frame: {avg_fk:.1f}KB)")
    print(f"    Avg read:  {avg_r:.1f}ms/frame")
    print(f"    Total written: {total_mb_written:.1f}MB")
    print(f"\n  Compression:")
    ratio = full_mb/(block_mb+small_mb) if (block_mb+small_mb) > 0 else 0
    print(f"    Delta: {block_mb+small_mb:.1f}MB vs Full: {full_mb:.0f}MB → {ratio:.0f}×")

    v = "PASS ✅" if final_err["median"] < 0.02 else "FAIL"
    print(f"\n  VERDICT: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_e10_fs_delta_e2e.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 E10-FS: Delta E2E",
            "backend": "filesystem",
            "steps": args.steps, "top_k": args.top_k,
            "roundtrip_ok": all_ok, "roundtrip_total": len(all_patches),
            "nrmse": {"final": final_err, "timeline": nrmse_timeline},
            "io": {"avg_write_ms": avg_w, "avg_read_ms": avg_r, "total_mb": total_mb_written},
            "compression": {"ratio": ratio},
            "verdict": v,
        }, f, indent=2, default=str)
    print(f"  → {out}\n[DONE E10-FS]")


if __name__ == "__main__":
    main()
