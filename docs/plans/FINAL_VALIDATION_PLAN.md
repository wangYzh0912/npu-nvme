# Final validation execution plan

Approved scope: finish the remaining v1.3 gates, with Qwen3 four-method FULL as the primary baseline after acceptance and GPT-2 as fallback. Model environments are independent: GPT-2/old, Qwen3/candidate. GPT-2 uses pinned repository model code; candidate compatibility is supplemental, not a Qwen gate.

1. Integrate E1/F1/E2, pin model provider and environment identities, add resumable fail-closed campaign supervision.
2. Complete D2 authoritative TP geometry, bounded exports, migration/crash/retention/rank-fault gates.
3. Complete Qwen none/Native/Ours/ByteCheckpoint Host: fixed TP4, seed42, batch1, seq128, zero dropout, FP32 state/BF16 compute, checkpoint8/continue11. Three independent source runs per method; fresh verification plus one warmup and three timed restores per persistence group.
4. E1 GPT-2 then XL: guarded split optimizer, strict restore, slow storage, pressure and failure retention.
5. F1 single-rank GPT-2: bytewise NPU change capture, receipt-only reference ACK, chain replay, compaction and continuation.
6. Final committed-source software/C/hardware gates, environment switchback, evidence/status/command documentation.

Raw writes only on 0000:83:00.0. Qwen [256,1280) GiB; D2 fault/migration scratch [1280,1408) GiB; F1 [1536,1664) GiB. Protected 84 remains filesystem-only. Every initialization verifies the old header digest. Hardware campaigns are serial; unknown DMA ownership blocks resource reuse. User worktree remains untouched; no mainline merge or push.

Keep exact source/config/library identities and failed attempts. Process exit is not validation pass. Existing small-state D2 and Host-array F1 passes are not training gates. Budgets include live/pending/pinned generations and all Host/HBM/IPC copies; phase deadlines freeze from successful pilots. Qwen live, TP4 delta and reshard remain out of scope.
