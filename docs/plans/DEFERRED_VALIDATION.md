# Deferred validation register

Status: full performance campaign deferred by scope decision on 2026-09-16.
The targeted ABI3 correctness chain below has since passed independently; it
does not complete the original performance campaign.

## Published baseline

- Qwen3-8B, TP4/DP1/PP1, NPU 0--3, deterministic local text.
- MindSpore 2.7.1, MindFormers 1.7, CANN 8.3.RC1, ABI major 2.
- `none`, `mindspore_native_save`, `ours`, and `bytecheckpoint_host` are
  implemented entry methods.
- Native completed the full two-restart 24-step chain.
- Ours completed real TP4 save, fresh-process restore, continuation and another
  save. The current source8 run also published step4/8 and matched the oracle.
- Candidate ABI2 completed a single-NPU real HBM-to-SPDK-to-HBM byte-exact
  round trip through the asynchronous ACL path. This is a transport result,
  not a single-NPU Qwen training claim.
- Candidate small-state D2 fixtures completed fresh-process TP2/TP4 save and
  restore. Single-device direct transfer probes passed on each of NPU 0--3.
- The public Qwen workload remains fixed to TP4. A single-NPU Qwen workload has
  not been accepted and must not be inferred from the single-device transport
  probes.

Primary evidence is external because model and checkpoint payloads are too
large for Git:

- `/models/npu_nvme_exp/user7-stack/qwen-training-acceptance-20260916-004`
  (`deferred`, 6 completed runs, 5 passing comparisons)
- `/models/npu_nvme_exp/user7-stack/qwen-entry-pilot-source-20260916-001`
- `/models/npu_nvme_exp/user7-stack/qwen-entry-pilot-resume-20260916-001`
- `/models/npu_nvme_exp/user7-stack/d2-fresh-sessions-20260915-001`
- `/models/npu_nvme_exp/user7-stack/c2b-hardware-20260915-001/candidate`
- `/models/npu_nvme_exp/user7-stack/placement-npu{0,1,2,3}-20260916-001`
- `/models/npu_nvme_exp/user7-stack/bytecheckpoint-worker-probe-20260916-002`
- `/models/npu_nvme_exp/user7-stack/release-software-20260916-005/junit.xml`

## Remaining campaign

Resume with a new output directory. Attempt 004 is immutable partial evidence.

The ABI3 NPU4--7 targeted acceptance completed the Ours explicit generation 1
step4-to8 and latest step8-to16 / step16-to24 chains, with all-rank oracle
comparisons, on 2026-10-22. The independent 2026-10-24 rerun is recorded in
`DEVELOPMENT_STATUS.md`. These are correctness results, not timing samples for
the old campaign; environment/source identity rules still govern reuse.

1. Run the source/explicit/latest/latest chain for ByteCheckpoint Host.
2. Run three independent source repetitions for each method to step11 with a
   step8 checkpoint. For each persistent method, run one warmup and three timed
   fresh restores from step8 to step11.
3. Require 37 run records, 35 comparisons, three timing samples for every
   persistent method, retention of three committed generations, clean owner
   shutdown and no hardware lease after completion.

```bash
sudo python3 train.py benchmark --config config/qwen_training.json \
  --output /models/NEW_QWEN_ACCEPTANCE --profile all
```

The command above starts a fresh campaign, including its oracle and Native
group. Attempt 004 references the oracle in attempt 002; the current reuse
verifier requires a local `trajectory-none` directory and does not follow that
indirection. Passing attempt 004 directly to `--reuse-multicycle` therefore
rejects reuse and reruns the groups. Supporting chained evidence reuse remains
a later convenience improvement. Repeated timing always runs again.

The candidate D2 commit-fault matrix and TP2/TP4 fresh-process rank fixture
passed on 2026-09-18. Their evidence and exact scope are recorded in
`docs/plans/D2_FINAL_VALIDATION_20260918.md`. The real V2 offline-migration
acceptance remains deferred because the current raw device has no valid strict
D1 source superblock. It requires a preserved D1 source image; do not format
the low-address region to manufacture it. Injected multi-process rank
disconnect/owner-failure hardware campaigns also remain deferred.

## Later product work

Incremental checkpoints, Qwen live checkpointing, TP4-to-TP1 resharding,
single-NPU Qwen training, GPT-2/XL compatibility on the candidate environment,
multi-owner operation and power-loss claims remain outside the published
baseline. They require their own workload and acceptance evidence.
