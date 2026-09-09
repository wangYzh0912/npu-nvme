# Net full-state recovery comparison (2026-09-07)

This directory contains the current recovery-latency reference. Earlier timing records remain in Git history. The earlier timing boundary included a full 1.485 GB SHA-256 oracle scan before `state_ready`. The implementation now ends the performance metric after state application and device synchronization, and performs byte-exact validation only in a separate verification process.

| method | backend | state-ready mean | median | min–max | independent verification |
|---|---|---:|---:|---:|---|
| MindSpore native save | XFS `/models` | 7.0999 s | 7.1046 s | 6.9714–7.2902 s | persisted bytes exact; applied NPU state exact; 3-step loss deviation 0 |
| ByteCheckpoint host-adapted | XFS `/models` | 15.4111 s | 15.3883 s | 15.2781–15.6297 s | persisted bytes exact; applied NPU state exact; 3-step loss deviation 0 |
| Ours | Raw SPDK `0000:83:00.0` | 4.1823 s | 4.1787 s | 4.1598–4.2140 s | DirectCheckpoint per-field checksum; applied NPU state exact; 3-step loss deviation 0 |

`state-ready = state_ready - restore_begin`, after model construction and device synchronization. Timing repetitions execute one first training step but do not hash the restored state. Each method also has one independent `verify` process. The filesystem methods hash the loaded snapshot and a fresh capture of the applied state; Ours uses DirectCheckpoint's per-field disk/readback checksums plus the same fresh applied-state digest. Every verification process then continues training for three steps.

Mean phase split:

| method | adapter/metadata ready | read + deserialize | apply + NPU sync | state-ready total |
|---|---:|---:|---:|---:|
| MindSpore native save | 0.0003 s | 4.3404 s | 2.7592 s | 7.0999 s |
| ByteCheckpoint host-adapted | 0.0672 s | 12.3583 s | 2.9856 s | 15.4111 s |
| Ours | 0.1547 s | 3.9804 s | 0.0472 s | 4.1823 s |

ByteCheckpoint is 2.17× the Native state-ready latency in this limited same-filesystem run; most of the difference is in its Host-side planner/load/deserialization path. Ours is 41.1% below Native and 3.68× below ByteCheckpoint under the currently configured endpoints, but these are descriptive endpoint references rather than strict software speedups because Ours uses a different physical SSD. This is a five-sample reference, not a P99 or confidence-interval claim.

The first-step measurement includes graph compilation and is reported separately: Native mean 25.7280 s; ByteCheckpoint mean 26.4678 s; Ours mean 29.1199 s. Application time from process start is diagnostic only.

The source checkpoints use the same GPT-2 configuration, seed, batch file/fixture lineage, 591 state fields and 1,485,030,841 logical bytes. Native and ByteCheckpoint reuse step 35; the unavailable raw generation was regenerated from the same fixture as the planned minimal 3-step/1-generation source, at logical step 8. The source processes have different final state digests, so the comparison is same configured workload rather than the same checkpoint byte image.

The `/models` filesystem is PCI `84:00.0`; Ours targets raw PCI `83:00.0`. `same_physical_storage_verified=false`, so no strict three-way storage speedup is claimed. The original non-root run selected VA IOVA and could not attach the `uio_pci_generic` device. Running the existing MindSpore/CANN environment as root selected PA IOVA; the C V2 round-trip, Python runtime probe, FULL save, fresh restore, checksum and continuation gates then passed. No driver rebinding, formatting or filesystem fallback was used.

See `failure_analysis.json` for the invalid-run causes and fixes. Supporting inventory/preflight remain in `../full-state-recovery-20260907/`; the final Ours source evidence is in `../full-state-recovery-ours-root-20260907/ours_source_oracle/`.
