# v1.3 implementation status — 2026-09-13

Implementation branch: `codex/long-term-v1.3`, isolated checkout `/models/npu_nvme_exp/user7-stack/checkouts/long-term-v1.3`. Base `5b164b6`. The original experimental checkout remains separate. No push or master merge was performed. Canonical plan: `/home/user7/npu-nvme/docs/LONG_TERM_DEVELOPMENT_PLAN.md` (§8.4/§9).

| Work | Commit | Result / evidence |
|---|---|---|
| A0 baseline and inventories | 56b9c76 | Complete; A0/ |
| A safety implementation | 88ecc91, 646760c | Software gate-002: 70 cases pass, including 17 production C ASan/UBSan cases; A/ACCEPTANCE.md |
| B pure Python/C migration | 6187c00 | Production build, 3,641 identical exports, ctypes/C layout, 17 C sanitizer and 160 portable Python tests; B/ |
| RF-03 scheduler/capture ownership | a3b5b97 | Partial: real scheduler and frozen capture own their state; cells canonicalized. Portable 160 pass, final ownership/admission 20 pass; B/rf03-partial/ |
| E0/EN migration and native restart | 5b7ef43 | Original eight-step replay pass; deterministic fresh-process TP4 positive restart pass; E0/, EN/oracle-002.json |

B remains in progress. e167868 further extracts the legacy V2 catalog, metadata I/O, full-state restore runtime/target and batch transport. Latest portable regression: 181 pass, with five additional actual Host restore target cases passing. Five metadata fault fixtures preserve pre-migration bytes and rollback semantics. Write worker/live/FaF/incremental and weights-only compatibility remain in the facade; H01-compat hardware gate passed for GPT-2 NPU7 seeds 41/42/43; strict D1 restore remains separate. Subsequent D1/C1/B2/C2 and D2/E1/E2/F stages have not passed their entry/exit gates.

The deterministic Qwen pair restored 952 observed training-network parameters per rank exactly before continuation. Steps 9–11 losses and final state/control sidecars were bit-identical across source and restore on all four ranks. Source processes exited before new restore processes; all launchers exited 0. The frozen tolerance was not relaxed. Raw framework strategy and payload hashes establish the step8 disk schema of 291 model tensors + 291 m + 291 v + 9 per-rank scalars. No reshard or Ours TP4 claim is made.

First source/restore-001 continuation failed despite exact initial parameter restoration. Its original sources and outputs remain under EN. Positive source/restore-002 evidence does not overwrite that failure. The latest extra training-identity guard was checked against actual saved configs and CPU negative fixtures after the hardware pair; exact executed source snapshots remain in the evidence tree. Native software checks: 43 pass before six additional real worker fault cases; the expanded native contract suite is 24 pass. These counts overlap and must not be added as independent total coverage.

EN-native is now completed for its fixed deterministic TP4 scope at c460b83. Fresh restore-003 additionally reads back and compares all captured control/RNG state before ready, then exactly matches three-step loss and final state. EN-native/gate-001 has 39 passing post-run evidence and software fault cases; original hardware invocations/logs are preserved. E0 remains in progress: old/candidate GPT-2/Ours hardware regressions and environment promotion are outstanding. Native runs used only NPU plus filesystem checkpoints, with no raw SPDK I/O or environment installation changes.

`config/raw_test_region.json` records user-authorized whole-device destructive testing on 0000:83:00.0 (offset 0, length = device capacity). It was formatted as V2 on 2026-09-12. H01 GPT-2 frozen seeds 41/42/43, the isolated H02 subset, and FULL-IO have passed. 0000:84:00.0 and its /models filesystem remain protected. No whole-disk sector-by-sector verification is claimed.

Reviewer: self-reviewed. Rollback tags: batch-A0-entry, batch-A-entry, batch-B-entry. See EN-native/ACCEPTANCE.md and B/commit-restore-migration/README.md for the latest evidence. Exact per-task states/commits/limitations are in config/execution_status.json; artifacts have SHA-256 indexes beside their logs. Next: finish RF-03 facade extraction; enter D1 only after the B exit gate. TR-01/02/03 remain planned and are not aliases for hardware gates.


## H02 isolation update (2026-09-13)

The previous same-process reopen failure is preserved under `results/hw-v13/h02-stage4`. The isolated worker implementation is `8c82189`; software orchestration tests report 35 passed. Hardware rounds `h02-isolated-r1`, `r2`, and `r3` each completed 11/11 cases. The timeout case returned `-ETIMEDOUT` at approximately 50 ms, short close returned `-ETIMEDOUT`, extended close drained in approximately 406 ms, and independent verify workers reopened the device successfully. FULL-IO rerun passed. This evidence covers the declared H02 subset only; no DMA-stop, power-loss, or full D1 lifecycle claim is made.


## 2026-09-14 continuation

B software profile `gate-002` completed with 258 tests and no changed sources. Entry→exit hardware compatibility seeds 41–43 completed; the separate exit→entry direction remains unverified. Seed 43 exposed a real continuation allclose failure under the existing tolerance and is preserved as a failure. D1 commits `d955540` and subsequent strict restore/owner work remain in progress. A repeated SPDK probe lifecycle fix now destroys exited SPDK thread objects after join, addressing the observed resource exhaustion path.
