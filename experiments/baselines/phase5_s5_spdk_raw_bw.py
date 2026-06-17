#!/usr/bin/env python3
"""
Phase 5 S5: SPDK Raw Bandwidth Benchmark
========================================
Pure SPDK write BW test on the actual NVMe device.
Runs with pipeline_depth=8, chunk_size=4MB (matching the 4380 MB/s config).

Usage:
  echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
    python phase5_s5_spdk_raw_bw.py --device-id 1'
"""
import os, sys, time, json, struct, ctypes
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
PCI_ADDR = "0000:83:00.0"
DEVICE_ID = 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--pci-addr", type=str, default="0000:83:00.0")
    parser.add_argument("--data-gb", type=float, default=3.2,
                        help="Amount of data to write (GB). Default 3.2 = GPT-2 XL.")
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "80")
    os.environ["NPU_NVME_LISTENER_MODE"] = "off"

    print("=" * 70)
    print("Phase 5 S5: SPDK Raw Bandwidth Benchmark")
    print(f"  Data size: {args.data_gb:.1f} GB  |  PCI: {args.pci_addr}  |  NPU: {args.device_id}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Init DirectCheckpoint
    # ═══════════════════════════════════════════════════════════════
    print("\n[1] Init SPDK (pipeline_depth=8, chunk=4MB)...")
    t0 = time.perf_counter()

    from direct_checkpoint import DirectCheckpoint, lib
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci_addr,
        npu_device_id=args.device_id,
        pipeline_depth=8,
        requested_chunk_size=4 * 1024 * 1024,
        enable_profiling=True,
        profiling_dir=os.path.join(REPO, "output", "profiling", "S5_raw_bw"),
        spdk_shm_id=80,
        keep_last_n=2,
        slot_size_gb=10,
    )
    dt_init = time.perf_counter() - t0
    print(f"    SPDK init: {dt_init:.1f}s  |  Total NVMe: {ckpt.total_bytes/1024**3:.1f} GB")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Allocate NPU buffer and write it via SPDK
    # ═══════════════════════════════════════════════════════════════
    target_bytes = int(args.data_gb * 1024**3)
    chunk_size = 4 * 1024 * 1024  # 4MB
    num_chunks = (target_bytes + chunk_size - 1) // chunk_size

    print(f"\n[2] Allocating NPU buffer ({args.data_gb:.1f} GB, {num_chunks} chunks)...")
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=args.device_id)

    # Allocate NPU memory: use a dummy Parameter tensor
    buf_np = np.random.bytes(target_bytes)
    buf_np_padded = buf_np + b'\x00' * (chunk_size - len(buf_np) % chunk_size) if len(buf_np) % chunk_size else buf_np
    actual_size = len(buf_np_padded)
    actual_chunks = (actual_size + chunk_size - 1) // chunk_size

    t = ms.Parameter(ms.Tensor(np.frombuffer(buf_np_padded, dtype=np.uint8).reshape(-1), ms.uint8),
                     requires_grad=False, name="raw_bw_buf")
    # Init parameter data to force MS to allocate device memory
    t.init_data()
    time.sleep(0.5)  # let MS runtime allocate HBM
    from direct_checkpoint import get_dev_ptr
    ptr = get_dev_ptr(t)
    if ptr == 0:
        d = t.value() if hasattr(t, "value") else (t.data if hasattr(t, "data") else t)
        ptr = int(d._data_ptr()) if hasattr(d, "_data_ptr") else 0
    print(f"    NPU ptr: {hex(ptr)}, actual bytes: {actual_size} ({actual_size/1024**3:.2f} GB)")

    if ptr == 0:
        raise RuntimeError("Cannot get device pointer for NPU buffer — MS lazy allocation?")

    # Compute NVMe offsets: sequential write from the current stack_start
    base_offset = ckpt.stack_start_bytes
    npu_ptrs = (ctypes.c_void_p * actual_chunks)()
    nvme_offsets = (ctypes.c_uint64 * actual_chunks)()
    sizes = (ctypes.c_size_t * actual_chunks)()

    for i in range(actual_chunks):
        npu_ptrs[i] = ctypes.c_void_p(ptr + i * chunk_size)
        nvme_offsets[i] = ctypes.c_uint64(base_offset + i * chunk_size)
        sz = min(chunk_size, actual_size - i * chunk_size)
        sizes[i] = ctypes.c_size_t(sz)

    print(f"\n[3] Executing SPDK write_batch ({actual_chunks} chunks)...")
    # Sync NPU stream
    if hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
        ms.runtime.synchronize()

    t_write = time.perf_counter()
    rc = lib.npu_nvme_write_batch(ckpt.ctx, npu_ptrs, nvme_offsets, sizes, actual_chunks)
    dt_write = time.perf_counter() - t_write

    if rc != 0:
        print(f"    write_batch FAILED (rc={rc})")
        ckpt.close()
        return

    bw = actual_size / 1024 / 1024 / dt_write if dt_write > 0 else 0
    print(f"\n{'='*70}")
    print(f"S5 RAW BW RESULTS:")
    print(f"  Data:      {actual_size/1024**3:.2f} GB ({actual_chunks} chunks)")
    print(f"  Time:      {dt_write*1000:.1f} ms")
    print(f"  Bandwidth: {bw:.0f} MB/s")
    print(f"{'='*70}")

    # Try to read profiling CSV for detailed breakdown
    prof_dir = os.path.join(REPO, "output", "profiling", "S5_raw_bw")
    csv_path = os.path.join(prof_dir, "time_write.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        if len(lines) > 1:
            # Parse: item,buf_idx,npu_async_us,spdk_nvme_us,total_e2e_us
            # Pipeline duration = max(total_e2e_us) among first chunk of each buffer
            buf_total_us = {}
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    buf = int(parts[1])
                    total = int(parts[4])
                    buf_total_us[buf] = max(buf_total_us.get(buf, 0), total)
            # pipeline time: sum of each buffer's max E2E (pipelined)
            pipeline_us = sum(buf_total_us.values())
            pipeline_ms = pipeline_us / 1000
            pipeline_bw = actual_size / 1024 / 1024 / (pipeline_ms / 1000) if pipeline_ms > 0 else 0
            # Also compute per-buf breakdown
            avg_npu_us = 0; avg_spdk_us = 0; n_samples = 0
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    avg_npu_us += int(parts[2]); avg_spdk_us += int(parts[3]); n_samples += 1
            if n_samples > 0:
                avg_npu_us //= n_samples; avg_spdk_us //= n_samples
            print(f"  CSV profiler: pipeline=")
            print(f"    {}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_s5_spdk_raw_bw.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 S5: SPDK Raw BW",
            "config": {"pipeline_depth": 8, "chunk_size_mb": 4, "data_gb": args.data_gb},
            "results": {
                "init_sec": dt_init,
                "write_ms": dt_write * 1000,
                "bw_mbs": bw,
                "actual_bytes": actual_size,
                "num_chunks": actual_chunks,
            },
        }, f, indent=2)
    print(f"  → {out}")
    print("[DONE S5]")

    ckpt.close()


if __name__ == "__main__":
    main()
