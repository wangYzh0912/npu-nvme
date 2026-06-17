#!/usr/bin/env python3
"""
Phase 5 S6+S7: E2E Single-Card Delta Checkpoint (GRAPH_MODE, sink_size=1)
===========================================================================
Integrated FULL + delta checkpoint pipeline.
- GRAPH_MODE, dataset_sink_mode=True, sink_size=1 (=1 epoch per step)
- on_train_epoch_end callback: FULL ckpt at step 0, delta at every step
- SPDK write for both FULL and delta
- Recovery: read FULL from NVMe + delta chain from NVMe → verify

Key design:
  - epoch=steps, sink_size=1 → every step is an epoch boundary
  - callback.on_train_epoch_end fires after optimizer update (step complete)
  - FULL ckpt save is SYNCHRONOUS (blocking) via background_io_worker + join
  - Delta save is via sync_meta_io (small frames)

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_s6_s7_e2e.py --steps 20 --device-id 1'
"""
import os, sys, time, json, math, re, hashlib, copy, argparse, pickle, struct
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288
SMALL_THRESHOLD = 10000
PCI_ADDR = "0000:83:00.0"

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

def weights_hash(w):
    h = hashlib.md5()
    for nm in sorted(w.keys()):
        h.update(w[nm].astype(np.float32).tobytes())
    return h.hexdigest()

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

def apply_delta_patches(init_weights, block_patches, small_patches, block_size):
    w = copy.deepcopy(init_weights)
    for bp in block_patches:
        name = bp["name"]; bidx = bp["block_idx"]
        i8 = bp["int8_data"]; s = bp["scale"]
        fp32 = i8.astype(np.float32) * s if isinstance(i8, np.ndarray) else \
               np.frombuffer(i8, dtype=np.int8).astype(np.float32) * s
        start = bidx * block_size
        end = min(start + len(fp32), int(np.prod(w[name].shape)))
        wv = w[name].astype(np.float32).flatten()
        wv[start:end] = fp32[:end-start]
        w[name] = wv.reshape(w[name].shape)
    for sp in small_patches:
        name = sp["name"]; i8 = sp["int8_data"]; s = sp["scale"]
        fp32 = i8.astype(np.float32) * s if isinstance(i8, np.ndarray) else \
               np.frombuffer(i8, dtype=np.int8).astype(np.float32) * s
        w[name] = fp32[:int(np.prod(w[name].shape))].reshape(w[name].shape)
    return w

def build_block_delta(true_w, pold, lid, layer_map, block_size, top_k):
    """Compute block patches for one layer. Returns (blocks, smalls)."""
    large, small = classify_layer_params(layer_map[lid])
    all_param_blocks = []
    for pi in sorted(large.keys()):
        _, name, ne = large[pi]
        fp32 = true_w[name].astype(np.float32).flatten()
        nblk = math.ceil(ne / block_size)
        for b in range(nblk):
            s = b * block_size; e = min(s + block_size, ne)
            bd = fp32[s:e]
            po = pold.get(lid, name, b, bd)
            dn = float(np.sum((bd - po).astype(np.float64)**2))
            all_param_blocks.append((lid, name, b, bd, dn))

    ranked = sorted(all_param_blocks, key=lambda x: -x[4])
    tk = max(1, int(math.ceil(len(ranked) * top_k)))
    block_patches = []
    for lid_p, name_p, bidx_p, bd_p, dn_p in ranked[:tk]:
        sc = max(float(np.max(np.abs(bd_p)))/127.0, 1e-10)
        q = np.clip(np.round(bd_p/sc), -128, 127).astype(np.int8)
        block_patches.append({"layer_id": lid_p, "name": name_p, "block_idx": bidx_p,
                              "int8_data": q, "scale": float(sc), "delta_norm": dn_p})
        pold.put(lid_p, name_p, bidx_p, bd_p)

    small_patches = []
    for pi in sorted(small.keys()):
        _, name, ne = small[pi]
        fp32 = true_w[name].astype(np.float32)
        sc = max(float(np.max(np.abs(fp32)))/127.0, 1e-10)
        q = np.clip(np.round(fp32/sc), -128, 127).astype(np.int8)
        small_patches.append({"layer_id": lid, "name": name, "int8_data": q, "scale": float(sc)})
        pold.put(lid, name, 0, fp32)
    return block_patches, small_patches


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "78")
    os.environ["NPU_NVME_LISTENER_MODE"] = "off"

    print("=" * 70)
    print(f"Phase 5 S6+S7: E2E Delta Checkpoint  (Steps={args.steps}, GRAPH_MODE)")
    print("=" * 70)

    # ── 0. Model analysis ──
    print("\n[0] Model analysis (GPT-2 Small)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    rotation_ids = sorted([l for l in layer_map.keys() if l >= -2])
    total_elems_mb = sum(layer_elems[l] * 2 / 1e6 for l in rotation_ids)
    print(f"  {len(rotation_ids)} layers, {total_elems_mb:.0f}MB FP16")

    # ── 1. Init SPDK before training ──
    print("\n[1] Init SPDK (pipeline_depth=8)...")
    t0 = time.perf_counter()
    from direct_checkpoint import DirectCheckpoint, get_dev_ptr
    ckpt = DirectCheckpoint(
        nvme_addr=PCI_ADDR, npu_device_id=DEVICE_ID,
        pipeline_depth=8, requested_chunk_size=4*1024*1024,
        enable_profiling=False, spdk_shm_id=78,
        keep_last_n=100, slot_size_gb=5,
    )
    ckpt.delta_init(slot_size_mb=256, slot_count=128)
    dt_spdk = time.perf_counter() - t0
    print(f"  SPDK ready ({dt_spdk:.1f}s)")

    # ── 2. Training in GRAPH_MODE ──
    print(f"\n[2] Training {args.steps} steps (GRAPH_MODE, sink_size=1)...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.set_seed(42); ms.common.set_seed(42)

    train_model = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(train_model.trainable_params(), learning_rate=1e-5)

    class TrainCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad(); self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            return ops.Depend()(loss, self.opt(grads))

    cell = TrainCell(train_model, opt); ms_model = ms.Model(cell)
    ds_train = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds_train = ds_train.batch(1, drop_remainder=True).take(args.steps)

    snapshots = [(0, get_all_params_np(train_model))]
    pold = PoldStore()
    delta_stats = []
    ckpt_timings = []

    class CkptDeltaCallback(ms.Callback):
        def __init__(self, ckpt_mgr):
            self.ckpt = ckpt_mgr
            self.pold = pold
            self.stats = delta_stats
            self.timings = ckpt_timings
            self.cnt = 0
        def on_train_epoch_end(self, rc):
            cb_params = rc.original_args()
            # In sink_size=1, epoch_num = step number
            cur_step = cb_params.cur_epoch_num
            self.cnt += 1

            # Snapshot oracle weights
            t_snap = time.perf_counter()
            true_w = get_all_params_np(train_model)
            snapshots.append((self.cnt, true_w))
            t_snap_done = time.perf_counter()

            # Delta for every step > 0
            if self.cnt > 0:
                t_delta = time.perf_counter()
                block_patches, small_patches = [], []
                for lid in rotation_ids:
                    bp, sp = build_block_delta(true_w, self.pold, lid, layer_map,
                                               args.block_size, args.top_k)
                    block_patches.extend(bp); small_patches.extend(sp)

                # SPDK delta write (sync_meta_io — small frame, blocks until done)
                t_d_write = time.perf_counter()
                slot = self.ckpt.delta_save(self.cnt, block_patches, small_patches)
                t_d_done = time.perf_counter()

                self.stats.append({
                    "step": self.cnt, "slot": slot,
                    "snap_ms": (t_snap_done - t_snap) * 1000,
                    "delta_compute_ms": (t_d_write - t_delta) * 1000,
                    "delta_write_ms": (t_d_done - t_d_write) * 1000,
                    "n_blocks": len(block_patches), "n_small": len(small_patches),
                })

    cb = CkptDeltaCallback(ckpt)

    # Save initial FULL checkpoint synchronously
    print("  Writing FULL ckpt at step 0 (synchronous)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    save_model = AutoModel.from_config(cfg)
    init_w = snapshots[0][1]
    for p in save_model.trainable_params():
        if p.name in init_w:
            p.set_data(Tensor(init_w[p.name].astype(np.float16), ms.float16))

    t_full = time.perf_counter()
    ckpt.save(save_model, step=0, commit_meta=True)
    ckpt.wait_for_io_completion()
    ckpt.wait_async_io()
    dt_full = time.perf_counter() - t_full

    full_mb = sum(v["size"] for v in ckpt.meta_dict["checkpoints"]["step_0"]["params"].values()) / 1e6
    bw_full = full_mb / dt_full
    print(f"  FULL: {full_mb:.0f}MB in {dt_full*1000:.0f}ms (BW={bw_full:.0f} MB/s)")

    # Now train
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    t_train = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=ds_train, callbacks=[cb],
                   dataset_sink_mode=True, sink_size=1)
    dt_train = time.perf_counter() - t_train
    print(f"  Training: {dt_train:.1f}s ({len(snapshots)-1} steps)")

    # ── 3. Close SPDK ──
    ckpt.close()
    print("\n[3] SPDK closed. Delta stats:")

    for s in cb.stats:
        if s["step"] % 5 == 0:
            print(f"  Step {s['step']:3d}: snap={s['snap_ms']:.0f}ms  "
                  f"compute={s['delta_compute_ms']:.0f}ms  write={s['delta_write_ms']:.0f}ms  "
                  f"blocks={s['n_blocks']}  smalls={s['n_small']}")

    # ── 4. Recovery ──
    print(f"\n[4] Recovery: rebuild from init + delta chain (pure host-side)...")
    ms.set_seed(42); ms.common.set_seed(42)
    recover_model = AutoModel.from_config(cfg)
    w = {p.name: p.value().asnumpy().copy() for p in recover_model.trainable_params()}

    pkl_path = os.path.join(REPO, "experiments", "output", "checkpoint_meta.pkl")
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    r_ckpt = DirectCheckpoint(
        nvme_addr=PCI_ADDR, npu_device_id=DEVICE_ID,
        pipeline_depth=1, requested_chunk_size=4*1024*1024,
        enable_profiling=False, spdk_shm_id=77,
        keep_last_n=5, slot_size_gb=1,
    )
    r_ckpt.delta_init(slot_size_mb=256, slot_count=128)

    t_rec = time.perf_counter()
    for s in range(1, args.steps + 1):
        slot = meta["delta_chain"][f"step_{s}"]["slot"]
        sid, blocks, smalls = r_ckpt.delta_load_slot(slot)
        w = apply_delta_patches(w, blocks, smalls, args.block_size)
    dt_rec = time.perf_counter() - t_rec
    print(f"  {args.steps} deltas applied in {dt_rec*1000:.0f}ms")

    r_ckpt.close()

    # Write back to device
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    for p in recover_model.trainable_params():
        if p.name in w:
            ops.assign(p, Tensor(w[p.name].astype(np.float16), ms.float16))

    # ── 5. Verify ──
    print(f"\n[5] Verifying...")
    w_oracle = snapshots[args.steps][1]
    err = compute_nrmse(w, w_oracle)
    hash_rec = weights_hash(w)
    hash_ora = weights_hash(w_oracle)

    delta_mb = sum(v["frame_size"] for v in meta.get("delta_chain", {}).values()) / 1e6
    full_total = sum(s["n_blocks"] + s["n_small"] for s in cb.stats)
    ratio = (args.steps * total_elems_mb) / delta_mb if delta_mb > 0 else 0

    print(f"\n{'='*70}")
    print(f"S6+S7 RESULTS: E2E GRAPH_MODE Delta Checkpoint")
    print(f"{'='*70}")
    print(f"\n  NRMSE:  Median={err['median']:.4e}  P95={err['p95']:.4e}  Max={err['max']:.4e}")
    print(f"  Hash:   {'MATCH' if hash_rec == hash_ora else 'MISMATCH'}")
    print(f"\n  I/O:")
    print(f"    FULL write: {dt_full*1000:.0f}ms  ({full_mb:.0f}MB, BW={bw_full:.0f} MB/s)")
    avg_dw = np.mean([s['delta_write_ms'] for s in cb.stats]) if cb.stats else 0
    avg_dc = np.mean([s['delta_compute_ms'] for s in cb.stats]) if cb.stats else 0
    print(f"    Avg delta:  write={avg_dw:.0f}ms  compute(host)={avg_dc:.0f}ms")
    print(f"    Recovery:   {dt_rec*1000:.0f}ms")
    print(f"\n  Compression: {ratio:.1f}x  ({delta_mb:.1f}MB delta)")

    ok = err["median"] < 0.05 and err["max"] < 0.10
    hash_ok = hash_rec == hash_ora
    all_ok = ok and hash_ok
    verdict = "PASS" if all_ok else f"PARTIAL: NRMSE={'OK' if ok else 'FAIL'} Hash={'OK' if hash_ok else 'FAIL'}"
    print(f"\n  VERDICT: {verdict}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_s6_s7_e2e.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 S6+S7: E2E GRAPH_MODE",
            "config": {"steps": args.steps, "top_k": args.top_k, "block_size": args.block_size},
            "nrmse": err, "hash_match": hash_ok,
            "io": {"full_write_ms": dt_full*1000, "full_bw_mbs": bw_full,
                   "avg_delta_write_ms": float(avg_dw), "recovery_ms": dt_rec*1000,
                   "delta_stats": cb.stats},
            "compression": {"ratio": ratio, "delta_mb": delta_mb},
            "verdict": verdict, "all_pass": all_ok,
        }, f, indent=2, default=str)
    print(f"  → {out}")
    print("[DONE S6+S7]")


if __name__ == "__main__":
    main()
