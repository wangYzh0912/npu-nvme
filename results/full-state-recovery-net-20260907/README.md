# Net full-state recovery comparison (2026-09-07)

This directory supersedes the recovery-latency numbers in `../full-state-recovery-20260907/`. The earlier timing boundary included a full 1.485 GB SHA-256 oracle scan before `state_ready`. The implementation now ends the performance metric after state application and device synchronization, and performs byte-exact validation only in a separate verification process.

| method | backend | state-ready mean | median | min–max | independent verification |
|---|---|---:|---:|---:|---|
| MindSpore native save | XFS `/models` | 7.0999 s | 7.1046 s | 6.9714–7.2902 s | persisted bytes exact; applied NPU state exact; 3-step loss deviation 0 |
| ByteCheckpoint host-adapted | XFS `/models` | 15.4111 s | 15.3883 s | 15.2781–15.6297 s | persisted bytes exact; applied NPU state exact; 3-step loss deviation 0 |
| Ours | Raw SPDK `0000:83:00.0` | not measured | — | — | runtime attach failed; no filesystem fallback |

`state-ready = state_ready - restore_begin`, after model construction and device synchronization. Timing repetitions execute one first training step but do not hash the restored state. Each method also has one independent `verify` process that hashes both the persisted snapshot and a fresh capture of the applied model/optimizer/control state, then continues training for three steps.

Mean phase split:

| method | adapter/metadata ready | read + deserialize | apply + NPU sync | state-ready total |
|---|---:|---:|---:|---:|
| MindSpore native save | 0.0003 s | 4.3404 s | 2.7592 s | 7.0999 s |
| ByteCheckpoint host-adapted | 0.0672 s | 12.3583 s | 2.9856 s | 15.4111 s |

ByteCheckpoint is 2.17× the Native state-ready latency in this limited same-filesystem run; most of the difference is in its Host-side planner/load/deserialization path. This is a five-sample reference, not a P99 or confidence-interval claim.

The first-step measurement includes graph compilation and is reported separately: Native mean 25.7280 s; ByteCheckpoint mean 26.4678 s. Application time from process start is diagnostic only.

The source checkpoints use the same GPT-2 configuration, seed, batch file/fixture lineage, step 35, 591 state fields and 1,485,030,841 logical bytes. The two source processes have different final state digests, so the comparison is same configured workload rather than the same checkpoint byte image.

The `/models` filesystem is PCI `84:00.0`; Ours targets raw PCI `83:00.0`. `same_physical_storage_verified=false`, so no three-way storage speedup is claimed. Ours remains blocked because SPDK requires PA IOVA while the active process reports VA mode and cannot attach `83:00.0`.

Supporting inventory, device evidence, runtime-probe failure and C smoke evidence remain in `../full-state-recovery-20260907/`.
