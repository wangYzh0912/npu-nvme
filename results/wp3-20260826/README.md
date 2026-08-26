# WP3 correctness evidence (2026-08-26)

This bundle records the first S2/R0 correctness gates on branch
`exp/wp2-wp3-remaining`. Existing v1/v2 additive frames remain compatible;
the new v3 frame is replacement-valued and carries a manifest digest,
native dtype, lineage generations, and CRC32.

## Completed gates

- I0/Z0-Z9: CPU oracle, parameter-local blocks, ACK-only reference advance,
  generation ordering, native replacement, non-finite rejection, and CRC;
  29 Python tests pass in the combined Python suite.
- I1 real gate: GPT-2 (589 state arrays, 21 non-finite arrays) and GPT-2 XL
  (772 state arrays, 2 non-finite arrays) both fail at the post-warmup
  numeric gate. No invalid trajectory sample is admitted. The runs are
  retained as reproducible MindSpore/MindFormers environment evidence.
- I1 synthetic: deterministic 100-step sparse/dense/hot-block trajectory
  replay remains a pass, but is not a training result.
- I2 NPU graph equivalence: norm, Top-K values/indices, selected values,
  per-block scale, and INT8 quantization all match the CPU reference on NPU 5
  (`32 x 257`, variable valid lengths, `k=7`).
- I4 base: `FileS2Ring` writes a complete frame with flush+fsync and atomic
  slot replacement; wrap and corruption rejection pass.
- I5 first loopback: `s2_host_spdk.py` wrote and read a 4,159-byte S2 frame
  at the 83.0.0 safe offset 64 GiB. The returned bytes matched exactly;
  ACK and independent recovery reached generation 1. C-layer write latency
  was 348 us and the aligned transfer was 8,192 bytes.
- I5 NPU loopback: `s2_npu_spdk.py` transferred the same 4,159-byte frame
  through an ACL HBM buffer and 83.0.0 at 64 GiB + 16 MiB. HBM→SPDK→HBM
  bytes matched exactly, with aligned transfer 8,192 bytes and generation 1.

## Not yet closed

I1 real MindSpore weight/optimizer trajectories remain blocked by the
post-warmup non-finite model state. I3 HBM buffer lifecycle, I4 cross-process
replay on the target, and I6 raw-device restart/chain fault matrix remain open.
The GPT-2 13B A7 short-sequence attempt is recorded as a MindFormers 1.3.2
static-shape harness boundary; GPT-2 XL A7 remains the valid real-training
FaF/Reactor result.
