# WP3 correctness evidence (2026-08-26)

This bundle records the first S2/R0 correctness gates on branch
`exp/wp1-wp2-closeout`. Existing v1/v2 additive frames remain compatible;
the new v3 frame is replacement-valued and carries a manifest digest,
native dtype, lineage generations, and CRC32.

## Completed gates

- I0/Z0-Z9: CPU oracle, parameter-local blocks, ACK-only reference advance,
  generation ordering, native replacement, non-finite rejection, and CRC;
  26 Python tests pass in the combined Python suite.
- I1 smoke: deterministic 100-step sparse/dense/hot-block trajectory replay,
  with per-step frame bytes, selected IDs, Jaccard, generation, and exact
  final recovery. This is a CPU synthetic trajectory, not a training result.
- I4 base: `FileS2Ring` writes a complete frame with flush+fsync and atomic
  slot replacement; wrap and corruption rejection pass.
- I5 first loopback: `s2_host_spdk.py` wrote and read a 4,159-byte S2 frame
  at the 83.0.0 safe offset 64 GiB. The returned bytes matched exactly;
  ACK and independent recovery reached generation 1. C-layer write latency
  was 348 us and the aligned transfer was 8,192 bytes.

## Not yet closed

I1 real MindSpore weight/optimizer trajectories, I2 CPU/NPU graph equivalence,
I3 HBM buffer lifecycle, I4 cross-process replay on the target, I5 NPU-HBM
frame loopback, and I6 raw-device restart/chain fault matrix remain open.
The GPT-2 13B A7 short-sequence attempt is recorded as a MindFormers 1.3.2
static-shape harness boundary; GPT-2 XL A7 remains the valid real-training
FaF/Reactor result.
