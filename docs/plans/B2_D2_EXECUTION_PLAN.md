# B2—D2 execution plan (user-approved revision, 2026-09-14)

Base: 193408f; branch codex/b2-d2. Complete IO-01..04, FM-01..02,
MR-01..03, QW-02/06/07. No E1 live, F0/F1 incremental, reshard,
power-loss claims, driver upgrade, automatic owner takeover or publishing.

## Frozen decisions

* Reuse SPDK DMA hugepage buffers directly with ACL. D2H has recorded hardware
  evidence; independently validate async H2D and each environment. No automatic
  two-tier Host allocation or synchronous fallback.
* Correctness/lifecycle/resource limits are mandatory. Measure and disclose
  performance, but the user explicitly removed the proposed 5% regression gate.
* Distinguish checkpoint bytes/fields/chunks from per-submit bytes/items and
  in-flight credits. Actual Qwen state is 24,574,980,144 bytes/rank, four ranks;
  1/4/16 MiB chunks produce 95,520/25,224/7,872 total descriptors. Do not apply
  C1's 1 MiB, 65,536-descriptor, 64 GiB batch or 10 GiB/rank bounds globally.
* Start new transport at 4 MiB/depth4; test chunks 1/4/16 MiB and depths
  1/4/8/16, with resource-admitted 32/64 probes. No silent depth clamping.
  Fragment by namespace geometry and NVMe transfer limits where necessary.
* Configure tick submit/copy/checksum work separately; bounded checksum slices
  and request rotation must preserve metadata and completion progress.
* Retain frozen snapshots. Select HBM snapshot, chunked Host snapshot, or
  explicit blocking capture during pilot based on measured feasibility;
  freeze the selected mode before formal runs, never change it implicitly.
* Count framework reserved/allocated peaks, external ACL, SPDK/hugepages,
  snapshots, IPC and metadata separately; deduplicate shared Host mappings.
  Keep C1's 32 GiB budget only for its control. Qwen's historical framework
  limit is 58 GB, not measured usage. Measure actual peaks and runtime headroom.
* Separate compile/capture/checkpoint/restore/communication/close deadlines.
  Pilot uses known budgets; absent Qwen process budget starts at 7200 seconds.
  Formal deadline = max(existing budget, 3 * slowest successful pilot).
  Observation timeout does not prove DMA stop or authorize freeing buffers.
* Single SPDK/NVMe owner; rank-local ACL and NPU ownership. Probe shared Host
  pool mapping before choosing IPC; shared offsets/epochs, never raw remote
  NPU pointers. Socket transport is an explicit pre-run selection if shared
  mapping is unavailable; account for copying. No extra NVMe owner in ranks.
* Retain actual per-rank state; no mandatory replicated-tensor deduplication.
  Distinguish sharded/replicated parameters from per-rank controls.
* New media uses independent registered extents/offline copies, protected
  headers, immutable payload/manifests, two anchors and ordered flushes.
  Page manifests; separate page limits from aggregate decode/retention budgets.
  Retention 2/3 configurable, default 2; account for union of fallback roots,
  pins and pending generations rather than assuming three physical slots.
  V2 read-only/offline migration remains explicit; no V2 writer on new regions.

## Eight sequential acceptance batches

1. B2a: verify sources/buffers, versioned transfer request/receipts, async write
   and checksum; D2H H07 and impacted H01/H02.
2. B2b: async read/Host/metadata/flush, fragmentation and scheduling; four-way
   H07, G16 and H01/H02. Only then B2 complete.
3. C2a: all callers including strict FULL capture/restore use one async kernel;
   frozen old/new comparison; wrappers only submit+wait; retire sync bulk.
4. C2b: remove old blocking batch APIs; ABI major 2, canonical bindings,
   symbol/layout/build checks, archive locks and G16/H01/H02/H07 acceptance.
5. E0 + QW-06: old/candidate GPT-2/Ours; candidate Qwen; measured capture,
   topology, actual library identities and resource/layout/deadline budgets.
   Independent schema/resource probes may start earlier.
6. D2-format: separate worktree/config at verified C2; format, paged manifests,
   anchors/pins/retention/retry; G06/G08/G09, H01/H04-format and V2 migration.
7. D2-rank: 2/4 rank fixtures, shared pool/IPC, collective commit and bounded
   resources; missing/wrong/slow/disconnected rank and owner-failure gates.
8. Global ready + E2-tp4: small-shard global restore first, then Qwen weights
   and full_state in train.py with existing Native workload/oracle reused.

## Validation and evidence

CPU/C_IMPL cover streaming >64 GiB, >65536 descriptors, paged metadata,
retention 2/3, multiple requests, overflow, corruption, timeout/late completion,
quarantine and rank failures. Large simulated/sparse fixtures are not hardware.
Run ASan/UBSan for real C with external devices stubbed.

GPT-2 retains C1 four methods, seeds 41/42/43, 5 warmup + 30 formal steps,
checkpoint every3 and 3-step continuation. Add XL frozen large-state regression,
not E1 live acceptance. Qwen uses EN TP4/seq128/batch1/seed42/dropout0,
FP32 parameters/Adam and BF16 compute, save step8/source through11. Native/Ours
have three independent full-state source/restart repetitions; timing records
one warmup and three measured restores per group. Byte-exact/control checks
mandatory; per-value loss rtol=1e-5/atol=1e-6 unchanged. Model construction and
required integrity/application are inside state_ready; extra oracle separate.

Each stage binds actual commit, source, loaded binary/dependency hashes,
configuration, devices, child exits and resource traces. Preserve failures.
Final source reruns impacted combined gates; no combining old and new source
into false acceptance. Performance groups retain raw83 vs filesystem84 split.

## Execution safety and continuation

User worktree remains independent. Raw only 0000:83:00.0; protect /models on
0000:84:00.0. NPU7 for HW1; NPU0..3 for TP4; serial hardware campaigns.
Existing root authorization applies. Old/candidate use independently built
libraries; no global environment promotion. Keep stage entry/exit refs.
Proceed automatically after each gate. Fix failures, retain unknown DMA owners,
and continue independent software work when hardware is unavailable. Update
LONG_TERM_DEVELOPMENT_PLAN.md, execution status and command documentation only
with actual completed acceptance. No merge/push in this scope.

## Measured implementation progress (2026-09-14)

B2 remains in progress. Versioned request-owned receipts/digests, direct shared
SPDK DMA D2H/H2D, metadata/flush dependencies, failed-write history, bounded
Host copy/checksum ticks, safe request rotation and effective capability queries
are implemented. At a210d16, old and candidate each passed 36 real configurations
(72 total); depth64 actually reached 64 DMA slots and the largest payload was
1,090,519,557 bytes. Combined C_IMPL/layout:39 pass; affected C1 software:216 pass.
These development matrices do not replace the remaining combined H01/H02/H07 gates.

Additional verified boundary corrections: H02 request peaks/backpressure use
runtime capabilities instead of a fixed16; expanded C_IMPL orchestration gets
720 seconds instead of180. Request quantum defaults to max(4,pipeline depth),
while copy/checksum bytes and submission work have separate explicit budgets.
Per-submit 65536 items/64GiB are reported limits; larger streams use multiple
bounded requests. Metadata per-I/O1MiB remains distinct from D2 paged manifests.
C2, E0/QW-06 and D2/E2 acceptance have not yet been completed.


## Execution update, 2026-09-15

The later user instruction to finish all remaining items extends this execution
through E1 and F0/F1; the initial exclusions above describe the earlier batch.

C2 software: 294 pass at 7461cde. Old/candidate independently built ABI2 libraries
passed 36 transfer configurations each and the read fault campaign. Frozen
four-method, three-seed training acceptance is running at 019ec68; no combined
C2 completion claim until this closes. The isolated environment now includes
system administration command paths and explicitly resolves stage libraries.

D2 [256,1280) GiB was not blank. The original 12 KiB header was saved, failed D2
parsing, then explicit initialization compared its SHA-256 before clearing it.
Generation 1 passed a separate-process mount/payload check at e6fd3d3. This does
not prove power-loss behavior, migration, or FULL+delta retention.

Roles now separate one Host-only SPDK/NVMe owner from rank-local private SPDK
pools and ACL copies. Owner creates no ACL device context; ranks use no-PCI
SPDK initialization and cannot submit storage operations. The failed shared
memzone path is not used by these sessions. Two-rank and four-rank small fixtures
passed; old/candidate TP4 passed. These used same-process targets. Next is the
committed-build save/exit/fresh-restore session fixture, followed by real Qwen.

Training integration uses explicitly blocking capture until global commit;
source tensors remain stable. Partition metadata must come from the actual
strategy/schema, never tensor-size heuristics. Host fallback is permitted only
for parameters whose placement is recorded before capture; no DMA-failure
fallback. Restore requires tensor/control verification and a framework collective
barrier after the socket readiness decision and before optimizer updates.

F0 proceeds in the independent f0-integrity worktree. Strict pending-frame ACK,
manifest ownership, native-byte replacement, prefix descriptor validation and
bounded ring indexing are implemented; its final CPU gate remains separate from
F1 durable receipt/chain integration. E1 live is still gated on C2 acceptance.

Concrete evidence and remaining limitations are in config/execution_status.json.
