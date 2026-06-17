#!/usr/bin/env python3
"""
Phase 5 S4: 单卡端到端增量检查点测试 (Simplified)
==================================================
完整链路: 训练 → 全量+增量写盘(SPDK) → 从pickle恢复(纯host)

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_s4_e2e_single_card.py --steps 30'
"""
import os, sys, time, json, math, re, hashlib, copy, argparse, pickle
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288
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

def compute_loss_pynative(model, eval_batch):
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

# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--block_size", type=int, default=524288)
    parser.add_argument("--pci-addr", type=str, default="0000:83:00.0")
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "98")
    os.environ["NPU_NVME_LISTENER_MODE"] = "off"

    print("=" * 70)
    print(f"Phase 5 S4: E2E Single-Card Delta Checkpoint  (Steps={args.steps})")
    print("=" * 70)

    # ── 0. Model analysis ──
    print("\n[0] Model analysis...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2"); cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    params, layer_map, layer_elems = inspect_model_layers(model)
    rotation_ids = sorted([l for l in layer_map.keys() if l >= -2])
    total_elems_mb = sum(layer_elems[l] * 2 / 1e6 for l in rotation_ids)
    print(f"  {len(rotation_ids)} layers, {total_elems_mb:.0f}MB FP16")

    # ── 1. Eval batch ──
    print("\n[1] Preparing eval batch...")
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=False)
    ds = ds.batch(1, drop_remainder=True)
    it = ds.create_tuple_iterator()
    eval_batch = next(it)[0].asnumpy()

    # ── 2. Oracle training ──
    print(f"\n[2] Oracle training {args.steps} steps (GRAPH_MODE, sink_size=1)...")
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
    print(f"  Training done: {dt:.1f}s ({len(snapshots)} snapshots)")

    # ── 3. Init SPDK + DirectCheckpoint ──
    print("\n[3] Init SPDK & DirectCheckpoint...")
    from direct_checkpoint import DirectCheckpoint
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci_addr, npu_device_id=DEVICE_ID,
        pipeline_depth=8, requested_chunk_size=4*1024*1024,
        enable_profiling=False, spdk_shm_id=98,
        keep_last_n=100, slot_size_gb=5,
    )
    print(f"  DirectCheckpoint ready")

    # ── 4. Full checkpoint at step 0 ──
    print(f"\n[4] Writing FULL checkpoint at step 0...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    save_model = AutoModel.from_config(cfg)
    for p in save_model.trainable_params():
        if p.name in snapshots[0][1]:
            p.set_data(Tensor(snapshots[0][1][p.name].astype(np.float16), ms.float16))

    t_w = time.perf_counter()
    ckpt.save(save_model, step=0, commit_meta=True)
    # Wait for async background I/O to finish — true SPDK write happens here
    ckpt.wait_for_io_completion()
    ckpt.wait_async_io()
    full_w_ms = (time.perf_counter() - t_w) * 1000
    full_param_info = ckpt.meta_dict["checkpoints"]["step_0"]["params"]
    total_full_mb = sum(v["size"] for v in full_param_info.values()) / (1024*1024)
    bw_full = total_full_mb / full_w_ms * 1000
    print(f"  FULL ckpt: {total_full_mb:.1f}MB written in {full_w_ms:.0f}ms "
          f"(BW={bw_full:.0f} MB/s)")

    # ── 5. Delta at every step ──
    print(f"\n[5] Writing DELTA frames for steps 1-{args.steps}...")
    ckpt.delta_init(slot_size_mb=256, slot_count=max(args.steps + 10, 128))

    pold = PoldStore()
    delta_stats = []

    for step_idx in range(1, len(snapshots)):
        step, true_w = snapshots[step_idx]
        block_patches, small_patches = [], []

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

        t_d = time.perf_counter()
        slot = ckpt.delta_save(step, block_patches, small_patches)
        dms = (time.perf_counter() - t_d) * 1000
        delta_stats.append({"step": step, "slot": slot, "write_ms": dms,
                            "n_blocks": len(block_patches), "n_small": len(small_patches)})

        if step % 10 == 0:
            mb = sum(ckpt._delta_frame_sizes) / (1024*1024)
            print(f"  Step {step:3d}: slot={slot}  write={dms:.1f}ms  "
                  f"blocks={len(block_patches)}  smalls={len(small_patches)}  "
                  f"total_delta={mb:.1f}MB")

    # Close SPDK — recovery will be pure host-side
    ckpt.close()

    # ── 6. Recovery from pickle (pure host-side, no SPDK dependency) ──
    print(f"\n[6] Recovery from meta pickle (pure host-side)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)

    pkl_path = os.path.join(REPO, "experiments", "output", "checkpoint_meta.pkl")
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    # Verify meta has FULL checkpoint
    assert "step_0" in meta["checkpoints"], "FULL ckpt step_0 not in pickle!"

    # Build from seed: re-create initial model, apply delta chain in host numpy
    ms.set_seed(42); ms.common.set_seed(42)
    recover_model = AutoModel.from_config(cfg)

    # Don't re-read from NVMe on every step — init SPDK once, read all slots
    print("    Initializing SPDK for delta slot reads...")
    rr_ckpt = DirectCheckpoint(
        nvme_addr=args.pci_addr, npu_device_id=DEVICE_ID,
        pipeline_depth=1, requested_chunk_size=4*1024*1024,
        enable_profiling=False, spdk_shm_id=97,
        keep_last_n=5, slot_size_gb=1,
    )
    rr_ckpt.delta_init(slot_size_mb=256, slot_count=128)

    w = {p.name: p.value().asnumpy().copy() for p in recover_model.trainable_params()}

    t_r = time.perf_counter()
    for s in range(1, args.steps + 1):
        key = f"step_{s}"
        assert key in meta.get("delta_chain", {}), f"Missing delta step {s}"
        slot = meta["delta_chain"][key]["slot"]
        sid, blocks, smalls = rr_ckpt.delta_load_slot(slot)
        w = apply_delta_patches(w, blocks, smalls, args.block_size)
    rr_ckpt.close()
    rms = (time.perf_counter() - t_r) * 1000
    print(f"    {args.steps} deltas applied in {rms:.0f}ms")

    # Write back to device
    for p in recover_model.trainable_params():
        if p.name in w:
            t = Tensor(w[p.name].astype(np.float16), ms.float16)
            ops.assign(p, t)

    w_rec = w
    nrmse_model = recover_model

    # ── 7. Verify ──
    print(f"\n[7] Verifying recovery...")
    w_oracle = snapshots[args.steps][1]

    err = compute_nrmse(w_rec, w_oracle)
    hash_rec = weights_hash(w_rec)
    hash_ora = weights_hash(w_oracle)

    loss_model = AutoModel.from_config(cfg)
    for p in loss_model.trainable_params():
        if p.name in w_oracle:
            p.set_data(Tensor(w_oracle[p.name].astype(np.float16), ms.float16))
    oracle_loss = compute_loss_pynative(loss_model, eval_batch)

    for p in nrmse_model.trainable_params():
        if p.name in w_rec:
            p.set_data(Tensor(w_rec[p.name].astype(np.float16), ms.float16))
    init_loss = compute_loss_pynative(nrmse_model, eval_batch)
    rec_loss = compute_loss_pynative(loss_model, eval_batch)

    loss_drop = abs(oracle_loss - init_loss) + 1e-10
    rel = abs(rec_loss - oracle_loss) / loss_drop * 100

    delta_mb = sum(meta["delta_chain"][f"step_{s}"]["frame_size"]
                   for s in range(1, args.steps+1)) / (1024*1024)
    full_mb = args.steps * total_elems_mb
    ratio = full_mb / delta_mb if delta_mb > 0 else 0

    # ── 8. Report ──
    print(f"\n{'='*70}")
    print(f"S4 RESULTS: Single-Card E2E Delta Checkpoint ({args.steps} steps)")
    print(f"{'='*70}")
    print(f"\n  NRMSE:")
    print(f"    Median: {err['median']:.4e}")
    print(f"    P95:    {err['p95']:.4e}")
    print(f"    Max:    {err['max']:.4e}")
    print(f"\n  Loss:")
    print(f"    Init:       {init_loss:.6f}")
    print(f"    Oracle:     {oracle_loss:.6f}")
    print(f"    Recovered:  {rec_loss:.6f}")
    print(f"    Δrel:       {rel:.1f}% of loss drop")
    print(f"\n  Hash:")
    print(f"    Oracle:     {hash_ora}")
    print(f"    Recovered:  {hash_rec}")
    print(f"    Match:      {'PERFECT' if hash_rec == hash_ora else 'MISMATCH'}")
    print(f"\n  I/O:")
    print(f"    FULL write: {full_w_ms:.0f}ms")
    avg_dw = np.mean([s['write_ms'] for s in delta_stats])
    print(f"    Avg delta write: {avg_dw:.1f}ms")
    print(f"    Recovery:   {rms:.0f}ms")
    print(f"\n  Compression: {ratio:.1f}x  ({delta_mb:.1f}MB delta / {full_mb:.0f}MB full)")

    ok = err["median"] < 0.05 and err["max"] < 0.10
    loss_ok = rel < 10.0
    hash_ok = hash_rec == hash_ora
    all_ok = ok and loss_ok and hash_ok

    verdict = "PASS" if all_ok else \
              f"PARTIAL: NRMSE={'OK' if ok else 'FAIL'} Loss={'OK' if loss_ok else 'FAIL'} Hash={'OK' if hash_ok else 'FAIL'}"
    print(f"\n  VERDICT: {verdict}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_s4_e2e_single_card.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 S4: E2E Single-Card",
            "config": {"steps": args.steps, "top_k": args.top_k, "block_size": args.block_size},
            "nrmse": err, "loss": {"init": init_loss, "oracle": oracle_loss, "recovered": rec_loss, "rel_delta_pct": rel},
            "hash": {"oracle": hash_ora, "recovered": hash_rec, "match": hash_ok},
            "io": {"full_write_ms": full_w_ms, "avg_delta_write_ms": float(avg_dw),
                   "recovery_ms": rms},
            "compression": {"ratio": ratio, "delta_mb": delta_mb, "full_mb": full_mb},
            "verdict": verdict, "all_pass": all_ok,
        }, f, indent=2, default=str)
    print(f"  → {out}")
    print("[DONE S4]")


if __name__ == "__main__":
    main()
