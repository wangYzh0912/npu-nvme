# C1 unified frozen training acceptance — 2026-09-14

C1 formal acceptance **passed** at `cb3e21d70bf8a83f2e395603fc47691829fd2f01`.
The join result is [acceptance-002/acceptance.json](acceptance-002/acceptance.json);
[summary.json](summary.json) aggregates the measured values without cross-device ranking.

- Software: **213 cases** passed in the frozen C1 profile. A separate development
  regression selection passed 217 cases; these overlap and must not be added.
- Hardware: **12 training runs**, four methods × seeds 41/42/43. Each used a shared
  per-seed fixture, deterministic ON, batch 1, 129 input tokens (128 model tokens),
  dropout 0, 5 warmup + 30 formal steps. Each saving method persisted 10 checkpoints.
- Recovery: **9 correctness restores** in new processes, exact loaded-state bytes
  and control readback, three continuation losses identical to source oracle.
  Fixed per-step thresholds remain rtol=1e-5 / atol=1e-6.
- Timing: **9 warmup + 27 measured restores**. Mandatory integrity and state
  application remain before ready. Model/target construction is included; extra
  verification oracle scanning is separate. No p95/p99 claim is made.
- Environment: old stack, MindSpore 2.5.0, MindFormers 1.3.2, CANN 8.0.RC3, NPU7.
  ByteCheckpoint CPU worker uses upstream `6f00167c153f3e65a67240aaa5ef4850a1c740fd`.
- Binary: unchanged verified D1/retirement library SHA-256
  `0f521c87a95b18de2d87f48417abd45b603414e8586ff568c226fc4b36fc3c7d`.
  This is explicit legacy synchronous Host transfer, not B2 asynchronous acceptance.
- Raw target is only 0000:83:00.0. File methods use 0000:84:00.0 mounted on /models.
  Native/ByteCheckpoint and Ours belong to different storage comparison groups.

All formal process-tree Host and total NPU7 HBM samples were below the frozen
32 GiB limits. Samples are collected about every five seconds; they are not
allocator-level peak proofs. The maximum sampled values were approximately
15.52 GiB Host and 7.36 GiB HBM. The model config uses FP32 parameter state with
FP16 compute and FP32 layernorm/softmax, identified by its resolved configuration hash.

Raw evidence is retained at:
`/models/npu_nvme_exp/user7-stack/c1-runs/acceptance-002/`.
The compact copies preserve JSON, logs and JUnit reports. Original evidence manifests
also name fixture binaries retained at the raw location, so validate the full raw
package rather than treating the compact directory as a complete binary archive.
`compact_manifest.json` separately authenticates the selected copied files.

## Preserved development history

- `pilot-001`: failed Native control readback because a control summary was mistaken
  for the full state. The implementation was fixed; this run remains failed.
- `pilot-002`: four-method pilot passed, including actual upstream ByteCheckpoint
  save/restart; one measured restore per method, not formal C1 acceptance.
- `acceptance-001`: software gates blocked because root could not import user-site
  pytest. No hardware matrix ran. Explicit software-only PYTHONPATH fixed this.
- `acceptance-002`: software and hardware passed at the same frozen source identity.

Reproduce using `tools/run_c1_acceptance.py` in the documented old/root shell, with
an unused `--out` directory. It freezes actual HEAD in an external resolved config.
See `docs/migrations/C1_UNIFIED_ENTRY.md` for entry semantics and failure codes.

This does not establish B2/C2, candidate environment promotion, Qwen/Ours TP4,
live, incremental, resharding or power-failure safety. The next batch is B2.

## Public entry and compatibility checks

`fit-entry-001` completed the none workload; `inspect-001` completed read-only inspection;
`verify-entry-001` passed an independent Native restart against the frozen source.
Subsequent frontend commit `fd86de8ea13f7c25760764bb974eeddb74b96528` adds versioned-config forwarding
and capability code 3 for unsupported live mode. **110 CPU regression cases passed**
(`frontend-tests.xml`); these overlap the formal suite. The source audit confirms
all non-main definitions in the two changed legacy scripts are unchanged. Hardware
acceptance remains attributed to `cb3e21d`; it was not rerun on the frontend commit.
