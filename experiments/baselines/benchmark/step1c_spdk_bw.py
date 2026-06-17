#!/usr/bin/env python3
"""
Step 1c: SPDK FULL Checkpoint Bandwidth Benchmark
==================================================
Measures SPDK DMA write bandwidth for a GPT-2 XL FULL checkpoint
under the standard config (pipeline_depth=8, chunk=4MB).

This is the authoritative baseline for I1 (SPDK user-space NVMe)
bandwidth — the theoretical upper bound for any checkpoint write.

Metrics:
  S1c.1: Total data written (GPT-2 XL full params, bytes)
  S1c.2: SPDK HW write time (blocking, from profiling CSV)
  S1c.3: Effective bandwidth (MB/s)
  S1c.4: Breakdown: Prep / Layout / SPDK / Meta time

Usage:
  bash _run_1c.sh

Output:
  experiments/output/benchmark/step1c_spdk_bw.json
"""

import os, sys, time, json, math, ctypes, glob as gb

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms

OUTPUT_DIR = os.path.join(REPO, "experiments", "output", "benchmark")
PCI_ADDR = "0000:83:00.0"
DEVICE_ID = 1
SEQ_LEN = 1024


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 1c: SPDK FULL Ckpt BW Benchmark")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=4)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "80")
    os.environ["NPU_NVME_LISTENER_MODE"] = "off"

    chunk_size = args.chunk_mb * 1024 * 1024

    print("=" * 70)
    print("Step 1c: SPDK FULL Checkpoint BW Benchmark")
    print(f"  Model: GPT-2 XL (48L/1600d)")
    print(f"  PCIe: {PCI_ADDR}  |  NPU device: {args.device_id}")
    print(f"  pipeline_depth={args.pipeline_depth}, chunk={args.chunk_mb}MB")
    print("=" * 70)

    # ── 1. Build GPT-2 XL model ──
    print("\n[1] Loading GPT-2 XL...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=args.device_id)
    ms.set_seed(42)
    ms.common.set_seed(42)

    t0 = time.perf_counter()
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)
    t_model = time.perf_counter() - t0

    params = list(model.trainable_params())
    total_elems = sum(int(p.size) for p in params)
    total_bytes = total_elems * 2  # FP16
    total_gb = total_bytes / 1e9
    print(f"  Model init: {t_model:.1f}s, {len(params)} params, {total_gb:.2f} GB FP16")

    # ── 2. Init DirectCheckpoint ──
    print(f"\n[2] Init SPDK (pipeline_depth={args.pipeline_depth}, chunk={args.chunk_mb}MB)...")
    from direct_checkpoint import DirectCheckpoint, lib

    prof_dir = os.path.join(REPO, "output", "profiling", "S1c_full_ckpt_bw")

    t_init = time.perf_counter()
    ckpt = DirectCheckpoint(
        nvme_addr=PCI_ADDR,
        npu_device_id=args.device_id,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=chunk_size,
        enable_profiling=True,
        profiling_dir=prof_dir,
        spdk_shm_id=80,
        keep_last_n=2,
        slot_size_gb=args.slot_size_gb,
    )
    dt_init = time.perf_counter() - t_init
    print(f"  SPDK init: {dt_init:.1f}s  |  NVMe capacity: {ckpt.total_bytes/1024**3:.1f} GB")
    print(f"  Slot bytes: {ckpt.slot_bytes/1024**3:.1f} GB  |  Chunk size: {ckpt.chunk_size/1024**2:.0f} MB")

    # ── 3. Allocate NPU buffer and write via SPDK DMA ──
    # DirectCheckpoint.save() relies on model param device pointers which
    # are not available in PYNATIVE_MODE (params are host-side). Instead,
    # allocate a contiguous HBM buffer and use write_batch directly (same
    # as Phase 5 S5 methodology).
    print(f"\n[3] Allocating NPU buffer ({total_gb:.2f} GB) and executing SPDK write_batch...")

    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=args.device_id)

    buf_np = np.random.bytes(total_bytes)
    chunk_size = args.chunk_mb * 1024 * 1024
    actual_size = int(math.ceil(total_bytes / chunk_size)) * chunk_size
    if actual_size > total_bytes:
        buf_np = buf_np + b'\x00' * (actual_size - total_bytes)

    actual_chunks = actual_size // chunk_size

    t_ms = ms.Parameter(
        ms.Tensor(np.frombuffer(buf_np, dtype=np.uint8).reshape(-1), ms.uint8),
        requires_grad=False, name="spdk_bw_buf")
    t_ms.init_data()
    time.sleep(0.5)

    from direct_checkpoint import get_dev_ptr
    ptr = get_dev_ptr(t_ms)
    if ptr == 0:
        d = t_ms.value() if hasattr(t_ms, "value") else (t_ms.data if hasattr(t_ms, "data") else t_ms)
        ptr = int(d._data_ptr()) if hasattr(d, "_data_ptr") else 0
    print(f"  NPU ptr: {hex(ptr)}, actual bytes: {actual_size:,} ({actual_size/1024**3:.2f} GB)")
    if ptr == 0:
        raise RuntimeError("Cannot get device pointer for NPU buffer")

    # Build write_batch arrays
    npu_ptrs = (ctypes.c_void_p * actual_chunks)()
    nvme_offsets = (ctypes.c_uint64 * actual_chunks)()
    sizes = (ctypes.c_size_t * actual_chunks)()

    base_offset = ckpt.stack_start_bytes
    for i in range(actual_chunks):
        npu_ptrs[i] = ctypes.c_void_p(ptr + i * chunk_size)
        nvme_offsets[i] = ctypes.c_uint64(base_offset + i * chunk_size)
        sizes[i] = ctypes.c_size_t(chunk_size)

    if hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
        ms.runtime.synchronize()

    t_write = time.perf_counter()
    rc = lib.npu_nvme_write_batch(ckpt.ctx, npu_ptrs, nvme_offsets, sizes, actual_chunks)
    dt_write = time.perf_counter() - t_write

    if rc != 0:
        print(f"  write_batch FAILED (rc={rc})")
        ckpt.close()
        return

    bw_mbs = actual_size / 1024 / 1024 / dt_write if dt_write > 0 else 0
    print(f"  write_batch: {actual_size/1024**3:.2f} GB in {dt_write*1000:.1f} ms → {bw_mbs:.0f} MB/s")

    # ── 4. Parse profiling CSV for HW-only SPDK time ──
    csv_path = os.path.join(prof_dir, "time_write.csv")
    spdk_hw_ms = None
    csv_stats = None

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        if len(lines) > 1:
            buf_spdk_us = {}
            buf_total_us = {}
            total_npu_us = total_spdk_us = 0
            n_samples = 0

            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    buf = int(parts[1])
                    npu_us = int(parts[2])
                    spdk_us = int(parts[3])
                    e2e_us = int(parts[4])
                    buf_spdk_us[buf] = max(buf_spdk_us.get(buf, 0), spdk_us)
                    buf_total_us[buf] = max(buf_total_us.get(buf, 0), e2e_us)
                    total_npu_us += npu_us
                    total_spdk_us += spdk_us
                    n_samples += 1

            pipeline_us = sum(buf_spdk_us.values())
            pipeline_ms = pipeline_us / 1000
            spdk_hw_ms = pipeline_ms

            csv_stats = {
                "num_buffers": len(buf_spdk_us),
                "num_chunks": n_samples,
                "pipeline_spdk_us": pipeline_us,
                "pipeline_spdk_ms": round(pipeline_ms, 2),
                "avg_npu_us_per_chunk": total_npu_us // max(n_samples, 1),
                "avg_spdk_us_per_chunk": total_spdk_us // max(n_samples, 1),
                "csv_file": csv_path,
            }
            print(f"  Profiling CSV: {n_samples} chunks, {len(buf_spdk_us)} buffers")
            print(f"  Pipeline SPDK time: {pipeline_ms:.1f} ms")
    else:
        print(f"  No profiling CSV at {csv_path}")

    # ── 5. Compute bandwidth ──
    bw_mbs = actual_size / 1024 / 1024 / dt_write if dt_write > 0 else 0
    pipeline_bw = actual_size / 1024 / 1024 / (spdk_hw_ms / 1000) if spdk_hw_ms and spdk_hw_ms > 0 else None

    print(f"\n{'='*70}")
    print(f"STEP 1c SPDK FULL CKPT BW RESULTS")
    print(f"{'='*70}")
    print(f"  Model size:      {total_gb:.2f} GB FP16 (772 params, {total_elems:,} elems)")
    print(f"  Data written:    {actual_size:,} bytes ({actual_size/1024**3:.2f} GB)")
    print(f"  Write time:      {dt_write*1000:.1f} ms")
    print(f"  Bandwidth:       {bw_mbs:.0f} MB/s")
    if pipeline_bw:
        print(f"  Pipeline BW:     {pipeline_bw:.0f} MB/s (HW DMA only)")
    print(f"  Config:          pipeline_depth={args.pipeline_depth}, chunk={args.chunk_mb}MB, shm_id=80")
    print(f"{'='*70}")

    # ── 6. Save results ──
    results = {
        "experiment": "Step 1c: SPDK FULL Checkpoint BW Benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "GPT-2 XL (48L/1600d)",
        "config": {
            "n_params": len(params),
            "total_elems": total_elems,
            "total_bytes_fp16": total_bytes,
            "total_gb_fp16": round(total_gb, 3),
            "actual_bytes_written": actual_size,
            "actual_gb": round(actual_size / 1024**3, 3),
            "num_chunks": actual_chunks,
            "pipeline_depth": args.pipeline_depth,
            "chunk_mb": args.chunk_mb,
            "slot_size_gb": args.slot_size_gb,
            "spdk_shm_id": 80,
            "pci_addr": PCI_ADDR,
            "device_id": args.device_id,
        },
        "timing": {
            "model_init_s": round(t_model, 2),
            "spdk_init_s": round(dt_init, 2),
            "write_wall_s": round(dt_write, 3),
            "write_wall_ms": round(dt_write * 1000, 1),
            "spdk_hw_ms": round(spdk_hw_ms, 2) if spdk_hw_ms else None,
        },
        "bandwidth": {
            "wall_clock_mbs": round(bw_mbs, 1),
            "pipeline_dma_mbs": round(pipeline_bw, 1) if pipeline_bw else None,
        },
        "profiling_csv": csv_stats,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "step1c_spdk_bw.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Saved: {out}")

    # ── 7. Cleanup ──
    ckpt.close()
    print("[DONE Step 1c]")


if __name__ == "__main__":
    main()
