# C1 unified frozen training

C1 uses the verified strict FULL implementation with explicit `legacy_sync` transport.
The legacy `frozen_async` facade spelling does not mean asynchronous data movement.
B2/C2 own the replacement of this transfer path. The C ABI and media format remain D1/V2.

`train.py` is the public entry; its supervisor imports only the standard library.
`preflight --dry-run` validates configuration and identity without importing MindSpore,
NumPy, ctypes or opening a device. The helper creates an external config with the actual HEAD; checked-in configs are templates. `fit` prepares a shared fixture and runs one method;
`verify-restart` launches a new process against an existing source run; `benchmark`
runs every declared method/seed and includes independent correctness and timing restores.
`inspect` mounts strict media for inspection. None of these commands formats media.

```bash
python tools/run_c1_acceptance.py --dry-run --out /models/npu_nvme_exp/user7-stack/c1-runs/my-dry-run
python tools/run_c1_acceptance.py --out /models/npu_nvme_exp/user7-stack/c1-runs/my-acceptance
```

The hardware command requires the old environment/root template from
EXECUTION_ENVIRONMENT_AND_COMMANDS.md, with cwd, PYTHONPATH and NPU_NVME_LIBRARY_PATH
pointing to this checkout. Preserve vendor PYTHONPATH entries after sourcing CANN.
Only PCI 0000:83:00.0 is a raw target; PCI 0000:84:00.0 remains mounted on /models.
Run IDs/directories must be new. The formal helper resolves expected_commit to the
actual committed source and refuses unversioned source changes. Config/model, source,
loaded library, child process, resource and result identities accompany the evidence.
Pilot configurations allow recorded dirty source for development; they cannot pass
formal C1 acceptance.

## Compatibility

The old `experiments.baselines.repro.runner` module aliases the canonical workload
module; there is one training loop. The old CLI keeps its commands but validates the
actual commit instead of accepting only the historical f3c0861 baseline. Its `run`
and `suite` propagate required restore failures; continue-on-failure never resets the
aggregate exit status. Existing other baseline algorithms remain available through
their original adapters, outside the four-method C1 acceptance matrix.

C1 supports full_state, frozen, block admission, one pending FULL, two retained
generations and at most 1 MiB aligned chunks. Unsupported async/live/TP configurations
are explicitly rejected. Independent low-level legacy save/load entrypoints remain
retired; this CLI does not resurrect them.

## Results and measurement

Exit codes are 0 completed declared goal, 1 execution/validation failure, 2 malformed
configuration, 3 blocked capability/environment, 4 deadline/unknown result, 5 integrity
failure. Dry-run is planned/not_applicable; fit success is not a recovery verdict.
The none control has no recovery and never ranks as a checkpoint method.

Every method uses deterministic ON, the same initial-state fixture and fixed input
batches. The source exits before restore starts. Mandatory method checks, construction,
Host bridges, application and device completion are included before state_ready;
additional oracle scanning is recorded afterward. Timing runs retain mandatory checks.
The training interval ends after checkpoint drain/close and before source continuation
oracle. Restore samples report observed mean/range, without p95/p99 inference.

Native and ByteCheckpoint use filesystem-84. Ours uses raw-83; no cross-device software
speedup ranking is produced. ByteCheckpoint is a Host-adapted semantic port invoking
its locked upstream worker, not an official NPU backend.

Phase deadlines or unknown device progress stop the campaign. The supervisor retains
process identity and writes a device quarantine record. It never equates process kill
with DMA stop, and a subsequent run refuses device admission until reconciliation.

## Acceptance

C1 requires the software profile plus all four methods at seeds 41/42/43, 5 warmup and
30 formal steps, checkpoint interval 3, and three exact-oracle continuation steps.
Each saving method must pass byte/control verification in a fresh process. Each seed
then has one warmup restore plus three measured restores. Thresholds remain
rtol=1e-5/atol=1e-6 per individual oracle value; absent or nonfinite oracle values fail.
The acceptance join rejects missing methods, altered evidence, mixed revisions,
false none restores, failed children, skipped required tests and incorrect timing bounds.

C1 does not establish live, XL, Qwen/Ours TP4, incremental, B2 asynchronous transport,
new-media durability, candidate environment promotion or power-failure safety.

The workload dtype field denotes FP32 parameter state; the locked GPT-2 configuration uses FP16 compute with FP32 layernorm/softmax. Resource samples include the process tree and total NPU7 HBM at roughly five-second intervals; they are sampled peaks, not allocator-level peak proofs.

Versioned C1 configs passed to the legacy repro CLI are forwarded to the unified
supervisor; the single-card harness also forwards explicit `--config` invocations.
Historical flat configs/standalone flags retain their existing behavior. Unsupported
live modes return capability code 3. Frozen hardware acceptance at `cb3e21d` and
the subsequent frontend CPU checks are recorded separately in
`results/long-term-v1.3/C1/README.md`.
