# Incremental Observation Minimal Matrix (2026-09-04)

This branch contains observation-only experiments.  Incremental checkpoint
production paths, Top-K publication, R1/R2, quantization, and live parameter
capture remain frozen.  I/O acceptance is documented separately in
`docs/IO_MINIMAL_RESULTS_20260904.md`.

## INC-1: Real Training PMU

The corrected GPT-2 and GPT-2 XL seed-41 ArithmeticUtilization and Memory
runs completed with 10 warmup and 30 formal steps.  The graph-side
`AssignAdd(step_counter)` marker emitted exactly values 1..30 in each formal
run, and the exported task timeline contains the marker node.  Results retain
raw PMU CSVs, profiler exports, `events.jsonl`, and environment metadata.

The marker and host samples do not share a calibrated device clock:
`common_clock_alignment=false` and `idle_window_inference_allowed=false`.
Therefore these runs report resource averages and marker validity only; they
do not claim precise low-utilization windows or an optimizer deadline budget.
The earlier uncorrected/failed attempts are excluded from the committed
formal evidence and corrected count. Their profiler binary intermediates
were moved to the local, recoverable archive
`/tmp/npu-nvme-inc-profiler-intermediates-20260905`; the committed runs retain
the required exported PMU CSVs and logs.

## INC-2: Graph-Edge Load

Nine formal GPT-2 runs completed: five seed-41 modes (baseline,
marker-only, compute-only, memory-scan-only, incremental-chain), then
baseline/incremental-chain for seeds 42 and 43.  Each uses 20 warmup and 50
formal steps, graph-resident auxiliary work, and no host worker or
step-internal synchronization.  All runs include `config.json`,
`environment.json`, `events.jsonl`, and `result.json`.

The incremental-chain wall-clock overhead relative to the same-seed baseline
is 2.36%, 2.89%, and 2.33% for seeds 41, 42, and 43 respectively; the mean is
2.53%.  With only three paired seeds, the Student-t 95% CI extends above 3%,
so the planned ±3% equivalence gate is **not met**, even though each point
estimate is below 3%.  This result must not be promoted to an incremental
checkpoint performance claim.  Device-level per-step timing is intentionally
not claimed because the only synchronization is after the formal loop.

## INC-3: Tensor Change Coverage

Historical GPT-2 trajectories are reused where their raw manifests permit;
their older schema does not contain explicit adjacent/persisted reference
dimensions and is reported as field-limited historical evidence.

The current collector now records both `reference_semantics=["adjacent",
"persisted"]`, state categories, changed tensor/byte ratios, L2 energy
coverage, aligned physical bytes, Jaccard, block age, overdue and permanently
unselected blocks.  Persisted-reference statistics are materialized before
the simulated ACK advances the reference.  GPT-2 XL scoring uses FP32 block
reductions with deterministic FP64 global aggregation.  The adjacent 256K
view is aggregated from aligned 64K scores; persisted views are scored
independently because each block size advances a different ACK reference.
The FP32/FP64 CPU check has max relative score error below `1e-6`; the
aligned aggregation/direct 256K check has max relative error about `5.1e-6`,
with identical ranking.

The three requested GPT-2 XL seed runs passed: 120 training steps, windows
1–20/51–70/101–120, and exactly 60 observations per seed.  Each observation
covers the complete 9,822,624,004-byte model plus Adam state; all sampled
losses/states are finite.  Formal results are under the
`XL_seed{41,42,43}_formal_final_v2` directories and are combined in
`results/incremental-observation-20260904/summary.json`.

Across the three seeds, a 20% block budget covers 97.35% (64K) or 97.13%
(256K) of adjacent-step L2 energy on average.  Against the ACK-based
persisted reference it covers 93.95% or 93.59%, with about 2.04 GB or 2.07 GB
of 4 KiB-aligned block payload.  The aggregate number hides a categorical
failure: model-weight persisted coverage at 20% is only 64.77% (64K) or
63.85% (256K), while Adam-m is about 90% and Adam-v reaches 100%.

Selection stability is also weak: mean adjacent selection Jaccard is about
0.392 (64K) and 0.401 (256K).  Maximum block age reaches 120 steps, and every
seed has tens of thousands of overdue 64K blocks at the observed maximum.
Therefore this matrix does not satisfy the rule that a <=20% budget must
stably cover all major state categories.  A fixed hot-block cache is not a
candidate; any later sparse experiment would require category-aware budgets
plus max-age/residual handling.  The prior aborted and diagnostic runs remain
in separate directories and are not included in the formal summary.

## Decision Boundary

These measurements characterize training resource use and tensor-change
statistics only.  They do not authorize enabling an incremental checkpoint
algorithm.  Any future design must use the I/O deadline and restore gates
from the I/O branch together with these observation results.
