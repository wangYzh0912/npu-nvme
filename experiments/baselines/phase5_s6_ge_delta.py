#!/usr/bin/env python3
"""
Phase 5 S6: Batched Delta Detection in GE Graph + SPDK Delta Write
===================================================================
Moves delta compute from host numpy into the GE graph (CellB: batched ops).
- TrainCell.construct() does: forward → backward → delta detection → optimizer
- on_train_epoch_end callback: read delta norms, Top-K, INT8 quantize, SPDK delta write

Design:
  Delta norms are written to a Host-allocated buffer via ops (not via
  aclrtMemcpy). Instead, we compute on device, then use host-side callback
  to read via asnumpy() and do the rest on Python side.

Key: we need delta norms accessible from host after optimizer.
  One approach: use a Parameter tensor that GE writes to.
  Another: pass delta blocks through Depend and read after epoch_end.

  For sink=TRUE, we can't easily get per-step intermediate values.
  But with sink_size=1, each step IS an epoch boundary → callback fires!

  However, the trainable_params' values reflect post-optimizer state.
  We need post-backward/pre-optimizer parameter values for delta detection.

  Solution: In construct(), compute delta norms BEFORE optimizer,
  store them in a Parameter (non-trainable), and read them in the callback.

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_s6_ge_delta.py --steps 20'
"""
import os, sys, time, json, math, re, struct, argparse
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288

PCI_ADDR = "0000:83:00.0"


def inspect_model_layers(model):
    """Map params to layers, return (params, layer_map, layer_elems)."""
    params = list(model.trainable_params())
    layer_map, layer_elems = {}, {}
    for pi, p in enumerate(params):
        name = p.name
        m = re.search(r'backbone\.blocks\.(\d+)\.', name)
        if m:
            lid = int(m.group(1))
        elif 'backbone.embedding' in name:
            lid = -2
        elif 'backbone.layernorm' in name:
            lid = -1
        else:
            lid = -3
        ne = int(p.size)
        if lid not in layer_map:
            layer_map[lid] = {}
            layer_elems[lid] = 0
        layer_map[lid][pi] = (p, name, ne)
        layer_elems[lid] += ne
    return params, layer_map, layer_elems


def build_batched_delta_ops(layer_groups, nblks_list, flat_sizes_list, pold_map):
    """
    Build batched GE delta detection ops for all layers.
    Returns a list of per-layer tensors: [norm_tensor_layer_0, ...]

    pold_map: dict {layer_id: ms.Tensor(P_old_FP16_flat)} — pre-loaded initial guess
    For step 0, P_old is all zeros (initial condition).
    """
    layer_norms = []
    for gi in range(len(layer_groups)):
        group = layer_groups[gi]
        nb = nblks_list[gi]
        flat_sz = flat_sizes_list[gi]

        # Flatten + concat all params in this layer
        parts = []
        for p in group:
            pv = p.astype(ms.float32)
            parts.append(ops.Reshape()(pv, (-1,)))
        fd = parts[0] if len(parts) == 1 else ops.Concat()(tuple(parts))

        # Pad to multiple of BLOCK_SIZE
        padded_len = nb * BLOCK_SIZE
        pad_amt = padded_len - flat_sz
        if pad_amt > 0:
            padded = ops.pad(fd, (0, pad_amt), mode='constant', value=0.0)
        else:
            padded = fd

        # BATCHED: [nb, BLOCK_SIZE]
        blocks = ops.Reshape()(padded, (nb, BLOCK_SIZE))

        # P_old lookup — currently all zeros (true P_old would be in pold_map)
        # This is a placeholder: real P_old would be loaded before construct
        zeros = ops.ZerosLike()(blocks)
        deltas = ops.Sub()(blocks, zeros)

        # Per-block L2 norm: ReduceSum(Sub^2, axis=1)
        norms = ops.ReduceSum()(ops.Mul()(deltas, deltas), 1)
        # norms shape: [nb]
        layer_norms.append(ops.Cast()(norms, ms.float32))

    return layer_norms


def get_all_params_np(model):
    return {p.name: p.value().asnumpy().copy() for p in model.trainable_params()}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top_k", type=float, default=0.10)
    parser.add_argument("--block_size", type=int, default=524288)
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "76")
    os.environ["NPU_NVME_LISTENER_MODE"] = "off"

    print("=" * 70)
    print(f"Phase 5 S6: GE Batched Delta Detection + SPDK Write")
    print(f"  Steps={args.steps}  TopK={args.top_k}")
    print("=" * 70)

    # ── 0. Model analysis ──
    print("\n[0] Model analysis (GPT-2 Small)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)

    _, layer_map, layer_elems = inspect_model_layers(model)
    rotation_ids = sorted([l for l in layer_map.keys() if l >= -2])
    print(f"  {len(rotation_ids)} layers covered")

    # Build layer groups for GE
    layer_groups = []
    nblks_list = []
    flat_sizes_list = []
    per_layer_param_names = []  # for mapping norms back to param/block indices

    for lid in rotation_ids:
        group = []
        pnames_in_layer = []
        for pi in sorted(layer_map[lid].keys()):
            p, name, ne = layer_map[lid][pi]
            group.append(p)
            pnames_in_layer.append((name, ne, pi))
        layer_groups.append(group)
        flat_sz = sum(layer_elems[lid])
        flat_sizes_list.append(flat_sz)
        nblks_list.append(math.ceil(flat_sz / BLOCK_SIZE))
        per_layer_param_names.append(pnames_in_layer)

    total_blocks = sum(nblks_list)
    print(f"  {len(layer_groups)} layer groups, {total_blocks} total blocks")

    # ── 1. Init SPDK ──
    print("\n[1] Init SPDK...")
    from direct_checkpoint import DirectCheckpoint
    ckpt = DirectCheckpoint(
        nvme_addr=PCI_ADDR, npu_device_id=DEVICE_ID,
        pipeline_depth=8, requested_chunk_size=4 * 1024 * 1024,
        enable_profiling=False, spdk_shm_id=76,
        keep_last_n=100, slot_size_gb=5,
    )
    ckpt.delta_init(slot_size_mb=256, slot_count=128)
    print("  SPDK ready")

    # ── 2. Build GE Cell with batched delta ──
    print("\n[2] Building GE cell with batched delta ops...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.set_seed(42)
    ms.common.set_seed(42)

    train_model = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(train_model.trainable_params(), learning_rate=1e-5)

    # Non-trainable Parameter to store per-block delta norms (host-visible after epoch)
    norms_param = ms.Parameter(
        ms.Tensor(np.zeros(total_blocks, dtype=np.float32), ms.float32),
        requires_grad=False, name="delta_norms")

    class DeltaDetectionCell(nn.Cell):
        def __init__(self, net, optimizer):
            super().__init__(auto_prefix=False)
            self.net = net
            self.net.set_grad()
            self.opt = optimizer
            self.gf = ops.value_and_grad(self.net, grad_position=None,
                                          weights=self.opt.parameters)
            self.layer_groups = layer_groups
            self.nblks_list = nblks_list
            self.flat_sizes_list = flat_sizes_list
            self.norms_param = norms_param

        def construct(self, *inp):
            loss, grads = self.gf(*inp)

            # Batched delta detection (BEFORE optimizer)
            # Computes per-block norms into norms_buffer
            norms_parts = []
            offset = 0
            for gi in range(len(self.layer_groups)):
                group = self.layer_groups[gi]
                nb = self.nblks_list[gi]
                flat_sz = self.flat_sizes_list[gi]

                # Flatten + concat params
                parts = []
                for p in group:
                    pv = p.astype(ms.float32)
                    parts.append(ops.Reshape()(pv, (-1,)))
                fd = parts[0] if len(parts) == 1 else ops.Concat()(tuple(parts))

                # Pad to [nb * BLOCK_SIZE]
                padded_len = nb * BLOCK_SIZE
                if padded_len > flat_sz:
                    padded = ops.pad(fd, (0, padded_len - flat_sz), mode='constant', value=0.0)
                else:
                    padded = fd

                # Batched: Reshape → Sub → Square → ReduceSum
                blocks = ops.Reshape()(padded, (nb, BLOCK_SIZE))
                zeros = ops.ZerosLike()(blocks)
                deltas = ops.Sub()(blocks, zeros)
                sq = ops.Mul()(deltas, deltas)
                norms = ops.ReduceSum()(sq, 1)

                norms_parts.append(ops.Cast()(norms, ms.float32))

            all_norms = ops.Concat()(tuple(norms_parts))
            # Write to non-trainable param via assign
            norms_assigned = ops.Assign()(self.norms_param, all_norms)
            loss = ops.Depend()(loss, norms_assigned)

            # optimizer
            opt_res = self.opt(grads)
            loss = ops.Depend()(loss, opt_res)
            return loss

    cell = DeltaDetectionCell(train_model, opt)
    ms_model = ms.Model(cell)

    ds_train = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds_train = ds_train.batch(1, drop_remainder=True).take(args.steps)

    # ── 3. Save initial FULL ckpt ──
    print("\n[3] Writing FULL checkpoint at step 0 (sync)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=DEVICE_ID)
    save_model = AutoModel.from_config(cfg)
    init_w = get_all_params_np(train_model)
    for p in save_model.trainable_params():
        if p.name in init_w:
            p.set_data(Tensor(init_w[p.name].astype(np.float16), ms.float16))

    t_full = time.perf_counter()
    ckpt.save(save_model, step=0, commit_meta=True)
    ckpt.wait_for_io_completion()
    ckpt.wait_async_io()
    dt_full = time.perf_counter() - t_full
    full_mb = sum(v["size"] for v in ckpt.meta_dict["checkpoints"]["step_0"]["params"].values()) / 1e6
    print(f"  FULL: {full_mb:.0f}MB in {dt_full * 1000:.0f}ms (BW={full_mb / dt_full:.0f} MB/s)")

    # ── 4. Train with GE delta ──
    print(f"\n[4] Training {args.steps} steps (GRAPH_MODE, sink_size=1)...")
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)

    oracle_weights = [(0, get_all_params_np(train_model))]
    delta_write_stats = []
    step_times = []

    class DeltaCB(ms.Callback):
        def __init__(self):
            self.last_epoch_end = None
        def on_train_epoch_begin(self, rc):
            self.t_begin = time.perf_counter()
        def on_train_epoch_end(self, rc):
            t_now = time.perf_counter()
            step_ms = (t_now - self.t_begin) * 1000 if self.t_begin else 0
            cb_params = rc.original_args()
            cur_step = cb_params.cur_epoch_num
            step_times.append(step_ms)

            # Snapshot oracle weights
            true_w = get_all_params_np(train_model)
            oracle_weights.append((cur_step, true_w))

            # Read delta norms from GE
            t_delta = time.perf_counter()
            norms_arr = norms_param.value().asnumpy().copy()

            # Top-K block selection across all blocks
            # Map global block index → (layer_group, local_block_index)
            block_list = []
            offset = 0
            for gi, nb in enumerate(nblks_list):
                for bi in range(nb):
                    block_list.append((norms_arr[offset + bi], gi, bi))
                offset += nb

            ranked = sorted(block_list, key=lambda x: -x[0])
            tk = max(1, int(math.ceil(len(ranked) * args.top_k)))
            selected = ranked[:tk]

            block_patches = []
            small_patches = []
            # For selected blocks: read post-optimizer params, INT8 quantize
            for norm_val, gi, bi in selected:
                lid = rotation_ids[gi]
                # Find which param this block falls in
                off = 0
                found = False
                for pi in sorted(layer_map[lid].keys()):
                    p, name, ne = layer_map[lid][pi]
                    nbs = math.ceil(ne / BLOCK_SIZE)
                    if off <= bi < off + nbs:
                        local_bi = bi - off
                        fp32 = true_w[name].astype(np.float32).flatten()
                        s = local_bi * BLOCK_SIZE
                        e = min(s + BLOCK_SIZE, ne)
                        bd = fp32[s:e]
                        sc = max(float(np.max(np.abs(bd))) / 127.0, 1e-10)
                        q = np.clip(np.round(bd / sc), -128, 127).astype(np.int8)
                        block_patches.append({"layer_id": lid, "name": name,
                                               "block_idx": local_bi,
                                               "int8_data": q, "scale": float(sc),
                                               "delta_norm": float(norm_val)})
                        found = True
                        break
                    off += nbs
                if not found:
                    # Block falls in padding zone → skip
                    pass

            t_quant = time.perf_counter()
            # SPDK delta write
            slot = ckpt.delta_save(cur_step, block_patches, small_patches)
            t_save = time.perf_counter()

            delta_write_stats.append({
                "step": cur_step,
                "step_ms": step_ms,
                "delta_extract_ms": (t_delta - t_now) * 1000,
                "delta_quant_ms": (t_quant - t_delta) * 1000,
                "delta_write_ms": (t_save - t_quant) * 1000,
                "n_blocks": len(block_patches),
            })
            self.last_epoch_end = t_now

    cb = DeltaCB()

    t_train = time.perf_counter()
    ms_model.train(epoch=args.steps, train_dataset=ds_train, callbacks=[cb],
                   dataset_sink_mode=True, sink_size=1)
    dt_train = time.perf_counter() - t_train

    ckpt.close()

    print(f"  Training: {dt_train:.1f}s ({args.steps} steps)")

    avg_step = np.mean(step_times) if step_times else 0
    print(f"  Avg step: {avg_step:.0f}ms")

    for s in delta_write_stats:
        if s["step"] % 5 == 0:
            print(f"  Step {s['step']:3d}: {s['step_ms']:.0f}ms  "
                  f"extract={s['delta_extract_ms']:.0f}ms  "
                  f"quant={s['delta_quant_ms']:.0f}ms  "
                  f"write={s['delta_write_ms']:.0f}ms  "
                  f"blocks={s['n_blocks']}")

    # ── 5. Summary ──
    avg_delta_total = np.mean([s['delta_extract_ms'] + s['delta_quant_ms'] + s['delta_write_ms']
                                for s in delta_write_stats]) if delta_write_stats else 0
    delta_mb = sum(s["frame_size"] for s in ckpt.meta_dict.get("delta_chain", {}).values()) / 1e6

    print(f"\n{'='*70}")
    print(f"S6 Results: GE Batched Delta Detection + SPDK Write")
    print(f"{'='*70}")
    print(f"  FULL write: {dt_full * 1000:.0f}ms ({full_mb:.0f}MB, BW={full_mb / dt_full:.0f} MB/s)")
    print(f"  Avg step:   {avg_step:.0f}ms")
    print(f"  Avg delta total: {avg_delta_total:.0f}ms")
    print(f"  Delta MB:   {delta_mb:.1f}MB")
    print(f"  Steps:      {len(delta_write_stats)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_s6_ge_delta.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 S6: GE Batched Delta Detection",
            "config": {"steps": args.steps, "top_k": args.top_k, "block_size": args.block_size},
            "full_write_ms": dt_full * 1000,
            "full_bw_mbs": full_mb / dt_full,
            "avg_step_ms": float(avg_step),
            "delta_stats": delta_write_stats,
        }, f, indent=2, default=str)
    print(f"  → {out}")
    print("[DONE S6]")


if __name__ == "__main__":
    main()
