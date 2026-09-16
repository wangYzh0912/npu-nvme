# Qwen training baseline delivery — 2026-09-16

Original exhaustive objective: deliver a working Qwen3-8B TP4 training entry on the candidate
environment, with none / Native / Ours / ByteCheckpoint Host, configurable steps,
periodic FULL checkpoints, explicit/latest restore, and save after restore.
Use the fixed local sample only. GPT-2/XL compatibility and live do not block
this delivery. Incremental checkpoints, Qwen live and reshard are deferred. The
final publication boundary below keeps the usable Native/Ours mainline and
defers the unfinished exhaustive four-method statistics matrix.

Implementation order:
1. Preserve pilot failure evidence, recover all eight NPUs, fix Host/device pointer
   classification, and validate direct SPDK hugepage transfers.
2. Unify preflight/fit/verify-restart/benchmark/inspect, explicit environment and
   build identity, persistent multi-operation sessions and generation contracts.
3. Validate four methods against uninterrupted training, including periodic
   save, explicit/latest restore, subsequent save and second fresh restore.
   Include resource, corrupt/incomplete generation and lifecycle gates.
4. Archive old implementation refs and uncommitted code/experimental evidence
   to origin under archive/qwen-baseline-20260916/*. Keep large arrays, models
   and checkpoints in external local storage with hashed inventory. Never
   publish credentials. Verify remote archives before cleaning local worktrees.
5. Curate the release tree, build and test it, publish to origin/master without
   rewriting remote history, and synchronize /home/user7/npu-nvme to that clean
   master. Keep only current implementation, supported adapters, tests,
   configuration, necessary documentation and current reproducible evidence.

Errors initiate diagnosis (environment, previous evidence, official upstream
documentation), fixes and affected reruns; a failed attempt is not task
completion. A training process still reports errors honestly. DMA retention is
not bypassed to keep a failed job training.

Boundaries: raw writes only 0000:83:00.0; protected 84 filesystem unchanged;
Qwen [256,1280) GiB; D2 scratch [1280,1408) GiB; serial hardware admission;
header digest before initialization. Controlled recovery of this failed job's
processes and all eight NPUs was explicitly approved. Preserve hugepage mappings until
device reset/resource checks permit reuse. No whole-machine reboot authorized.

Acceptance: 24-step fixed-sample trajectory with saves every four steps,
source exit at eight, explicit step-four recovery, latest step-eight recovery
to sixteen, then another fresh recovery to twenty-four; compare state/control
and continuation with uninterrupted execution. Also retain the four-method
three-source comparison with one warmup and three measured fresh restores per
persistence method. Loss tolerances remain rtol=1e-5 / atol=1e-6. Performance
is measured and disclosed without the cancelled five-percent regression gate.

## Release status (2026-09-16)

The usable Qwen3 baseline is ready for publication. The supported training
topology is TP4/DP1/PP1 on NPU 0--3 with fixed local text and explicit
MindSpore 2.7.1, MindFormers 1.7 and CANN 8.3.RC1 identities. `train.py`
provides preflight, fit, explicit/latest restore, restart verification,
inspection and the optional exhaustive benchmark.

Validated behavior:

- The 24-step uninterrupted oracle passed. Native passed source8, explicit
  step4-to8, latest step8-to16 and a second fresh latest step16-to24 restore.
  Model, Adam and control state were exact; loss used rtol=1e-5/atol=1e-6.
- Ours passed a real Qwen TP4 save, fresh-process restore, continuation and
  subsequent save in `qwen-entry-pilot-{source,resume}-20260916-001`.
  Attempt 004 also passed step4/8 periodic Ours saves and exact source8 oracle
  comparison. All four rank sessions and the storage owner closed cleanly.
- Candidate D2 fresh-process fixtures passed save and restore for TP2 and TP4.
  Candidate ABI2 also passed a real single-NPU HBM-to-SPDK-to-HBM byte-exact
  round trip through `aclrtMemcpyAsync`; MindSpore device/Host placement and
  direct hugepage transfer probes passed on NPU 0--3. These fixtures validate
  the transport and rank protocol; the public Qwen entry remains fixed to TP4
  rather than advertising arbitrary topology.
- The ByteCheckpoint attached-shared-memory lifetime fix passed a real
  independent worker save/fresh-load probe. The full Qwen ByteCheckpoint matrix
  is deferred.
- The final software gate passed 394 tests. A clean rebuild at `ef5d113a`
  produced the same `libnpu_nvme.so.2.0` SHA256 as the hardware-tested runtime:
  `ba6dc8156be8d1f9c353d309e881afe6e0391916471be6152df390aa13c4d2b7`.

The user approved publishing this usable baseline before the exhaustive
four-method statistics campaign. Attempt 004 is therefore recorded as
`deferred`, with 6 completed runs and 5 passing oracle comparisons. It was
stopped after a safe checkpoint boundary with all eight NPUs idle. The exact
remaining matrix is maintained in `docs/plans/DEFERRED_VALIDATION.md` and must
not be described as passed.

Historical worktrees and superseded attempts are preserved under remote
`archive/qwen-baseline-20260916/*` refs. Large model/checkpoint payloads and the
original history bundle remain in the hash-verified external archive. Commands
and selected host paths are documented in `EXECUTION_ENVIRONMENT_AND_COMMANDS.md`.
