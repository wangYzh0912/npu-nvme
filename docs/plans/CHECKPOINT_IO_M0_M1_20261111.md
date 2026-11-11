# Checkpoint I/O M0/M1 execution contract

Registered on 2026-11-11, before the corrected hardware queue. Roadmap source:
`codex/aic-aiv-io-roadmap-20261110`, commit
`cb535fa33b0d92e977f49f26efc51abbaaf9c78b`.
Implementation starts from current ABI3 commit
`d1aa9dfdda2450eeb6c3509dbaae60eeaf1f7ef2`; the earlier roadmap tree is not
the training implementation. Work is isolated from the user's existing edits in
`/tmp/npu-nvme-checkpoint-io-next` on
`codex/checkpoint-io-m0m1-20261111`. The initial prototype remains frozen in
`/tmp/npu-nvme-checkpoint-io-m0m1` on `codex/checkpoint-io-m0m1-20261110`
while its full regression executes. Its incorrect logical-as-physical event
labels are reported as coverage gaps; the new branch fixes those labels and
repeated sink initialization before short diagnostic acceptance.

## Scope and execution

Only M0 and M1 are entered. M2 requires measured attribution and a separate
registered intervention. AIC/AIV scheduling and KV offload are deferred.

Entry: `tools/checkpoint_io.py preflight|m0|diagnostic|m1|analyse|queue`.
The queue stops on any failure; it does not kill or reclaim retained DMA owners.
Existing `train.py fit` owns hardware locking, lease retention, raw-region
registration, process cleanup, source identities, and fresh-process selection.

The frozen candidate environment is
`/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json`.
Runs use physical NPU 4–7, Qwen3-8B TP4/DP1/PP1, sequence length 128,
micro-batch 1, seed 42, FULL state, 4 MiB chunks, native depth 4, a single
host NVMe owner, and raw83 [256,1280) GiB with retention 3. No formatting.
Converted weights and raw run artifacts use tmpfs because durable storage does
not have enough headroom for model conversion. Compact evidence is archived to
`/models/npu_nvme_exp/user7-stack/aic-aiv-io-20261110`.

M0 runs the existing complete targeted FULL acceptance, including fresh-process
explicit/latest restore, exact state/control comparison, loss tolerance
rtol=1e-5/atol=1e-6, and interrupted-publication reconciliation. Detailed event
logging is enabled. Ordinary `none` runs retain periodic oracle manifests.
The first attempt saved successfully but failed comparison because an experiment
change omitted an intermediate T0 oracle. Its logs remain under tmpfs `m0`.
The fix restricts that omission to explicitly registered performance T0 runs.
The corrected run uses tmpfs `m0-retry`.

Short diagnostics separately run T0/T1/T2 through step 32 with LR horizon 128.
All four ranks' T1 and T2 tensor order, chunk offsets, lengths and SHA256 values
must match at all eight save boundaries. Final state, control and loss must
match T0. Diagnostics do not establish formal P95 or probe-overhead acceptance.
An additional new process restores retained step 28 and continues to step 32,
verifying the corrected source tree against the diagnostic T0 oracle.

M1 disables detailed events, uses LR horizon 128 and ends at step 64. Warmup
is steps 1–24; formal observations are steps 28–64, ten saves per process
(28,32,...,64). The fixed process orders are:

| Independent repetition | Order |
| --- | --- |
| 0 | T0 none, T1 host sink, T2 blocking FULL |
| 1 | T1 host sink, T2 blocking FULL, T0 none |
| 2 | T2 blocking FULL, T0 none, T1 host sink |

Each repetition verifies exact final states/control and tolerant loss against
its T0 oracle. Each T2 run is followed by a new process restoring retained step
60 and continuing to step 64 against that oracle. These restore observations
are separate from steady-state statistics.

## Timing and event meanings

All added timestamps are host `CLOCK_MONOTONIC` nanoseconds on this machine.
No device/PMU clock conversion or engine-overlap claim is made.

| Event | Boundary |
| --- | --- |
| training_step | Step callback has read loss and overflow |
| source_read_begin / data_ready | Source preparation/copy starts; its context yields readable bytes |
| chunk_send_begin / source_safe_ack | Hash-validated source goes to socket; owner acknowledges source reuse |
| source_released | Consumer exits the source context |
| sink_consumed_source_safe | Synchronous SHA256 sink consumed this chunk; no durable/publication claim |
| owner_received / owner_validated | Bounded socket payload received; digest/length accepted |
| owner_payload_begin / owner_payload_end | Owner lock acquired; collective admission and raw payload write completed |
| native_submitted / native_completed | Native submit through terminal checked receipt; request ID is native-context local |
| payload_flush_begin / payload_flush_end | All payloads flushed |
| metadata_begin / metadata_flush_end | Immutable manifests/control/catalog written and flushed |
| anchor_flush_end / raw_commit_receipt | Alternate anchor flushed; committed raw generation returned |
| rank_complete_written / catalog_published | Rank evidence written; global filesystem catalog published |
| training_resume | Callback has completed the checkpoint boundary |
| owner_restore_chunk_begin / owner_restore_chunk_end | Raw read begins; rank acknowledges applying the verified bytes |
| restore_chunk_received / restore_chunk_applied | Rank validates the received chunk; apply returns after the native stop proof for device data |
| restore_controls_verified / restore_collective_ready | State/control readback passes; framework restore barrier completes |

Records include run/rank/physical NPU/step/raw request/native request/chunk,
bytes/status, and nullable buffer/slot identity. A device capture's buffer ID
and generation refer to `RankCopyExecutor`'s private Host bridge buffer lease,
not the C runtime's internal DMA slots. Host tensor views have no such slot.
Native request IDs need run, role and rank to be unique. Raw generation and
filesystem catalog generation are different counters, joined through the raw
receipt. Mount-time native requests legitimately have no training step/chunk.
An absent event is unknown coverage, never zero latency. Failed requests must
retain their existing stop-proof/close rules.

T1 uses the same `BlockingState`, tensor order and private rank-copy executor as
T2. It hashes one readable chunk synchronously and releases the source context
only after sink consumption. Device sources share the rank's fixed-size Host
buffer; Host fallback retains the existing one-tensor 2 GiB budget. T1 omits
socket backpressure, owner serialization, raw storage, filesystem preparation
journals, flush and global publication. Their latency difference is a screening
measurement, not a direct measurement of media time.

Per-step compute ends before checkpoint processing. The blocking save metric
starts before resource admission and full manifest hashing and ends after
save/publication. Formal step duration is begin-to-resume; final T0 oracle
validation is excluded. Formal throughput is 37×128 tokens divided by the
earliest rank's step-28 begin to latest rank's step-64 resume. This is a formal
window throughput, not total process throughput. Compilation, initial/final
validation and close/drain require separate reporting.

Each aligned window uses the maximum of the four rank durations. Each process
has ten samples and no reported P95. Pooled groups have 30 samples from three
processes; nearest-rank P95 is reported alongside every process distribution.
Ranks and repeated saves are not counted as independent process repetitions.

## Acceptance limits

CPU tests establish source contracts only; stubbed C tests establish their
specific ownership checks. Hardware acceptance is recorded separately.
The M1 analysis status means host timings measured. It does not imply engine
attribution or performance acceptance. PMU/HBM/device tasks, queue high-water
marks, full process throughput/drain and a matched no-probe overhead comparison
remain explicit coverage gaps until measured. The roadmap's 10% observation
perturbation gate cannot be passed merely by disabling detailed events.
No optimization or engine collaboration is approved by a timing-only report.
