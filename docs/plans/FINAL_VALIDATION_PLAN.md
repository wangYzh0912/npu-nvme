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

## Execution update — 2026-09-15 final-validation

- Pinned GPT-2/old provider, runtime Qwen TP4 schema, ByteCheckpoint Host adapter and NPU byte capture implemented. Candidate GPT-2 remains a disclosed supplemental incompatibility.
- Frozen `9f533b8` C2 software campaign `final-software-c2-20260915-002` passed 295 tests; F0 `final-software-f0-20260915-001` passed its 51 checks. NPU byte probe `npu-byte-capture-20260915-002` passed signed-zero/NaN/subnormal/unchanged checks. These do not replace E1/F1 training acceptance.
- `final-pilot-campaign-20260915-002` is a diagnostic pilot. Its checkout changed from `192aa60` to `1c87962` during Ours source (F1-related files only); source drift is recorded in the external sidecar `final-pilot-campaign-20260915-002-source-drift.json`. Do not promote this campaign to final committed-source acceptance.
- Campaign supervision now fingerprints tracked working bytes and HEAD before/after every stage. Drift invalidates the result after safe process completion and prevents subsequent launches.
- Restore timing begins before framework imports and ends after global-ready barrier, including mandatory state/control verification. Global duration uses earliest rank start and latest rank ready on the same host. Timing excludes launcher/owner startup and post-ready continuation; this boundary must remain explicit in comparisons.
- `tools/prepare_qwen_campaign.py` prepares 12 independent sources and 12 fresh restores, plus the final gate. Timed paths must be distinct accepted restores. Run only from a new frozen checkout after all pilot issues are resolved; retain failures and never reuse output directories.
- Remaining acceptance: successful Qwen pilot then full repeated campaign; E1 GPT-2/XL and pressure/fault cases; F1 training/compaction/continuation; full D2 fault/migration/rank coverage; committed-source final software/C/hardware gates and environment switchback. Primary model remains unchanged until all required gates pass.

### Hardware stop — Qwen pilot 002

The pilot reported device error `507001 (ts internal error)` on all four ranks at approximately 23:36. The owner failed with rank EOF and `closed=true`; rank processes remain alive without a DMA stop proof. The campaign supervisor was interrupted and its result explicitly marked `retained`; child processes and the hardware lease are preserved. E1/F1/fault stages were not launched. Do not infer that CPU affinity caused the device error: sampling established CPU-0 contention, but the device failure still needs diagnosis. A diagnostic affinity change and debugger samples are retained as external sidecars, disqualifying pilot timing from formal acceptance.

`6264356` restores the caller CPU affinity after EAL initialization. Both libraries built and loaded ABI2 successfully in `final-env-20260915-002`; hardware acceptance is pending. New rank/session diagnostics persist the last copy receipt and report retained resources before waiting. The Qwen launcher detects this state without waiting for the full campaign deadline. Recovery must establish stopped DMA before resource reuse; increasing deadlines or killing retained ranks is not acceptance.
