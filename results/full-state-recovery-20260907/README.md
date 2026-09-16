# Full-state recovery comparison (2026-09-07)

> Superseded for latency reporting by `../full-state-recovery-net-20260907/`. The values below include a full-state SHA-256 oracle scan before `state_ready`; retain this directory only as raw audit evidence and preflight/runtime-gate evidence.

This run compares complete GPT-2 state recovery on the same NPU configuration. The measured state contains model parameters, Adam state, controls, global step and data cursor.

| method | backend | state-ready mean | median | min–max | verification |
|---|---|---:|---:|---:|---|
| MindSpore native save | XFS `/models` | 8.6699 s | 8.7281 s | 8.4891–8.7602 s | 5/5 byte-exact; 3-step loss deviation 0 |
| ByteCheckpoint host-adapted | XFS `/models` | 16.6694 s | 16.5990 s | 16.3958–16.9262 s | 5/5 byte-exact; 3-step loss deviation 0 |
| Ours | Raw SPDK `0000:83:00.0` | not measured | — | — | runtime attach failed |

`state-ready` is measured from `restore_begin` (after model construction and device synchronization) through complete state loading, byte validation and state application. First-step timings are separate and include graph compilation.

The `/models` filesystem is backed by PCI `84:00.0`, while the Raw SPDK target is PCI `83:00.0` (`uio_pci_generic`). `same_physical_storage_verified=false`; these results are a recovery reference, not a strict same-disk speedup comparison.

Ours failed before any checkpoint read: SPDK required PA IOVA but the process was in VA mode and the controller attach failed. No filesystem fallback was used.

The focused Python suite passed 10/10 and the complete Python suite passed 117/117. The C pure-logic smoke passed 10/10; the V2 hardware smoke failed at `spdk_nvme_probe`, matching the runtime probe above. No CTest targets are registered in either build directory. See `c_smoke.json`.

Raw evidence is in `restore_timing/`, `restore_verify.json`, `checkpoint_inventory.json`, and `ours_runtime_probe.json`. Five repetitions are reported as raw values, mean, median and range only; no P99 or confidence interval is inferred.
