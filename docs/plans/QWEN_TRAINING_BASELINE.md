# Qwen training baseline delivery — 2026-09-16

Approved objective: deliver a working Qwen3-8B TP4 training entry on the candidate
environment, with none / Native / Ours / ByteCheckpoint Host, configurable steps,
periodic FULL checkpoints, explicit/latest restore, and save after restore.
Use the fixed local sample only. GPT-2/XL compatibility and live do not block
this delivery. Incremental checkpoints, Qwen live and reshard are deferred.

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

## Execution status (2026-09-16)

- Allocation classification fix: commit 556b757. Real MindSpore 2.7.1 Host
  scalar probe passed. Direct NPU hugepage probes passed on devices 0–3 after recovery.
- Training entry and periodic catalog: commit 369436b. Persistent owner/rank
  sessions, explicit/latest generation selection, reader pins, four backends,
  multicycle and repeated-restart campaign entry implemented; TP4 execution
  and final acceptance remain pending.
- Software gate: 398 tests passed (Python, D2, campaign, regression and E1).
  Real Native CPU model/Adam/control save and restore passed. Candidate entry
  preflight passed and reported the unreconciled hardware lease.
- Authorized eight-device reset completed without a host reboot (boot ID unchanged).
  Eight-rank HCCL all-reduce and device 0–3 direct hugepage DMA probes passed.
  The failed pilot lease was reconciled after these checks; old hugepage evidence remains.
- Real Qwen Ours source saved FULL at step 4. A fresh process restored it,
  continued to step 8 and saved FULL again; both runs passed with owner close verified.
  Evidence: qwen-entry-pilot-source-20260916-001 and
  qwen-entry-pilot-resume-20260916-001 under /models/npu_nvme_exp/user7-stack.
- Twenty-one additional historical worktree snapshots were pushed and remote-verified;
  archive-qwen-baseline-20260916/stage-snapshots.json records their exact refs.
- Main dirty workspace archived and remote verified at
  `archive/qwen-baseline-20260916/main-dirty`, commit a7eaab5.
  Dirty legacy-cleanup and long-term-v1.3 archives are also remote-verified;
  exact refs and external payload manifests are in config/qwen_archive_inventory.json.
  GitHub rejected the original history push at its 2 GiB pack limit. Those two
  remote snapshots use origin/master as their parent; all original history is
  preserved in the verified external original-history.bundle. Local main remains unchanged.
- Formal four-method training acceptance, release-tree cleanup and publication
  to master are NOT complete. CPU results must not be substituted for them.

Current entry commands (run from the implementation checkout):

```bash
python3 train.py preflight --config config/qwen_training.json
python3 train.py fit --config config/qwen_training.json --output /models/NEW_RUN
python3 train.py fit --config config/qwen_training.json --output /models/NEW_RESUME \
  --resume --checkpoint-root /models/NEW_RUN/checkpoints --generation latest --stop-step 24
python3 train.py verify-restart --config config/qwen_training.json \
  --output /models/NEW_VERIFY --checkpoint-root /models/NEW_RUN/checkpoints \
  --oracle-run /models/ORACLE_RUN --stop-step 24
python3 train.py benchmark --config config/qwen_training.json \
  --output /models/NEW_CAMPAIGN --profile all
python3 train.py inspect --config /models/CONFIG_WITH_CHECKPOINT_ROOT.json
```

Ours requires root for SPDK access. Commands intentionally reject an existing
unreconciled hardware lease. Example paths are placeholders for new directories.

Candidate-only build entry (validated without initializing devices):

```bash
python3 tools/build_training_runtime.py \
  --manifest /models/npu_nvme_exp/user7-stack/final-env-20260915-002/environments.json \
  --out /models/npu_nvme_exp/user7-stack/NEW_RUNTIME \
  --spdk /home/user7/npu-nvme/third_party/spdk \
  --dpdk-ring /home/user7/npu-nvme/build/dpdk_fix/librte_mempool_ring_fixed.a
```

The current successful build is qwen-training-runtime-20260916-001. Its generated
manifest is selected by config/qwen_training.json. Source/build dependencies
must be preserved or relocated before eventual workspace cleanup.
