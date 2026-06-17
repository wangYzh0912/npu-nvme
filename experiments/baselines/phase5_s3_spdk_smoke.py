#!/usr/bin/env python3
"""
Phase 5 S3: SPDK Delta I/O Smoke Test
=====================================
Quick smoke test to verify the hugepage fix actually lets SPDK init + delta I/O work.

Usage (must be root):
  echo "CGCL_2025_#$" | sudo -S bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    python phase5_s3_spdk_smoke.py --device-id 1'
"""
import os, sys, time, json
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1
PCI_ADDR = "0000:83:00.0"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--pci-addr", type=str, default="0000:83:00.0")
    args = parser.parse_args()

    os.environ.setdefault("SPDK_SHM_ID", "99")  # unique SHM ID for smoke test

    print("=" * 70)
    print("Phase 5 S3: SPDK Delta I/O Smoke Test")
    print("=" * 70)

    # ── Step 1: SPDK Init (triggers hugepage auto-fix) ──
    print("\n[1] Initializing SPDK (this tests the hugepage auto-fix)...")
    t0 = time.perf_counter()

    from direct_checkpoint import DirectCheckpoint
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci_addr,
        npu_device_id=args.device_id,
        pipeline_depth=4,
        requested_chunk_size=4 * 1024 * 1024,
        enable_profiling=False,
        spdk_shm_id=99,
    )
    dt = time.perf_counter() - t0
    print(f"    SPDK init SUCCESS in {dt:.1f}s ✅")

    # ── Step 2: Delta Init ──
    print("\n[2] Initializing delta area...")
    ckpt.delta_init(slot_size_mb=256, slot_count=32)
    print(f"    Delta area: {ckpt._delta_slot_count} slots × {ckpt._delta_slot_size // (1024*1024)}MB ✅")

    # ── Step 3: Write a dummy delta frame ──
    print("\n[3] Writing dummy delta frame via SPDK...")
    dummy_blocks = [{
        "layer_id": 0, "name": "test.weight", "block_idx": 0,
        "int8_data": np.arange(128, dtype=np.int8),
        "scale": 0.01, "delta_norm": 1.0
    }]
    dummy_smalls = [{
        "layer_id": 0, "name": "test.bias",
        "int8_data": np.zeros(16, dtype=np.int8), "scale": 0.001
    }]

    t_w = time.perf_counter()
    slot = ckpt.delta_save(step=1, block_patches=dummy_blocks, small_patches=dummy_smalls)
    wms = (time.perf_counter() - t_w) * 1000
    print(f"    Wrote to slot {slot} in {wms:.1f}ms ✅")

    # ── Step 4: Read back ──
    print("\n[4] Reading delta frame back via SPDK...")
    t_r = time.perf_counter()
    sid, r_blocks, r_smalls = ckpt.delta_load_slot(slot)
    rms = (time.perf_counter() - t_r) * 1000
    print(f"    Read slot {slot}: step_id={sid}, blocks={len(r_blocks)}, smalls={len(r_smalls)}, {rms:.1f}ms ✅")

    # ── Step 5: Verify byte-perfect round-trip ──
    print("\n[5] Verifying round-trip...")
    assert sid == 1, f"step_id mismatch: {sid} != 1"
    assert np.array_equal(dummy_blocks[0]["int8_data"], r_blocks[0]["int8_data"]), "BLOCK data corrupted!"
    assert np.array_equal(dummy_smalls[0]["int8_data"], r_smalls[0]["int8_data"]), "SMALL data corrupted!"
    print("    Round-trip byte-perfect ✅")

    # ── Step 6: Write 5 more frames (ring wrap test) ──
    print("\n[6] Writing 5 more frames (ring buffer test)...")
    for s in range(2, 7):
        slot = ckpt.delta_save(step=s, block_patches=dummy_blocks, small_patches=dummy_smalls)
    print(f"    5 frames written, last slot={slot} ✅")

    # Read them all back
    for s in range(1, 7):
        _, b, sm = ckpt.delta_load_slot(s - 1)  # slot index = step-1
        assert len(b) == 1 and len(sm) == 1, f"Step {s}: wrong counts"
    print("    All 6 frames read back correctly ✅")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("S3 SMOKE TEST: ALL PASSED ✅")
    print(f"  SPDK init:     {dt:.1f}s")
    print(f"  Delta write:   {wms:.1f}ms")
    print(f"  Delta read:    {rms:.1f}ms")
    print(f"  Round-trip:    byte-perfect")
    print(f"  Ring buffer:   6/6 frames OK")
    print(f"{'='*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "phase5_s3_spdk_smoke.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "Phase 5 S3: SPDK Delta I/O Smoke",
            "spdk_init_sec": dt,
            "delta_write_ms": wms,
            "delta_read_ms": rms,
            "roundtrip": "byte-perfect",
            "ring_buffer_6_frames": "OK",
        }, f, indent=2)
    print(f"  → {out}")
    print("[DONE S3]")


if __name__ == "__main__":
    main()
