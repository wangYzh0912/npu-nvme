# AIC/AIV follow-up probes

This follow-up separates three evidence sources: runtime KV metadata, native
HBM bandwidth counters, and task dependency/readiness observations. It does not
equate low AIC busy time with recoverable FLOPS.

## Current baseline and target

The v2 measurement package reports card 4 at AIC/AIV 10.45%/78.57% for
inference batch 1, input 128, and 10.83%/76.32% for batch 8. Six training PMU
runs show AIC <=10% in every sampled window where AIV >=50%. The inference
trace attributes about 97.6% of AIV task time to Contiguous/Slice. These facts
support an engine imbalance and a KV data-movement hypothesis. They do not show
that the Slice cost is HBM-bound, that a cache layout change is equivalent, or
that a ready AIC task was omitted by scheduling.

## Probe order

1. Freeze the v2 source identity, config, random seed, output hashes, and
   `PIECEWISE` graph mode. Use the original KV configuration as baseline and
   keep each control in an isolated vLLM source/container copy.
2. In one short warmup-plus-measurement request, record key/value tensor
   shape, stride, storage offset, dtype, cache layout, block size, block-table
   hash, layer, prefill/decode phase, and source location. Do not
   dump tensor contents or synchronize the device for metadata collection.
   In the isolated vLLM source copy, call `record_kv_tensor(...)` from the
   measured cache-view/update path to append JSONL, then normalize it with
   `tools/aic_aiv_followup.py kv-layout`. Keep the hook off in performance
   comparison runs after the metadata probe.
3. Collect AIC/AIV sample PMU, `--sys-hardware-mem=on`, and
   `--sys-hardware-mem-freq=50` in the same measured request where the pinned
   CANN release permits coexistence. First validate an idle and sustained
   matrix/vector positive control. Export the HBM tool CSV with timestamp and
   explicit read/write bytes/s columns, then normalize with `hbm-csv`. If the
   exporter provides only aggregate values or the collectors conflict, keep
   the run as aggregate/diagnostic evidence; do not align unlike executions as
   one time series.
4. Screen one variable at a time: baseline; fixed `kv_cache_memory_bytes`;
   fixed `num_gpu_blocks_override`; then a block-size control only if the
   backend supports it without changing attention semantics. Run 3 paired
   process restarts per condition, b1/l128 and b8/l128. Expand to l512 only
   when l128 produces a stable signal. Keep prefix caching disabled and retain
   output hashes, TTFT, decode latency, throughput, PMU, HBM, and task counts.
5. For a layout control, first test the cached tensor/view contract and the
   exact branch that triggers `.contiguous()`/Slice in the measured version.
   A proposed layout rewrite remains an experiment patch until all output
   hashes match and cache write/read positions plus block tables remain equal.
   Record its source diff and runtime identity with every run.
6. For training, capture one dependency-focused diagnostic at seq128 and one
   at seq512, while retaining sample-only PMU as the activity authority. Mark
   each AIC candidate as `ready`, `blocked`, `waiting_dependency`, or
   `unknown`, with an event/graph edge source. Do not infer readiness from an
   absent task or from temporal overlap. Classify windows with
   `classify-windows` and retain unknown when no dependency event exists.
7. Only take the strongest one or two candidates to isolated serial/co-run
   controls. Require equal work and output, legal dependencies, reproducible
   engine overlap, no HBM or collective regression, and end-to-end improvement
   before proposing a scheduler change.

## Normalized evidence

`kv-layout` expects JSONL with `run_id`, `layer`, `cache_kind`, `shape`,
`stride`, `dtype`, and `phase`; optional fields include `cache_layout`,
`storage_offset`, `block_size`, `block_table_hash`, `tensor_name`, and `source`.
`contiguous` is derived from shape/stride, not trusted from a caller.

`hbm-csv` accepts only `timestamp_ns` or `timestamp_us`, plus
`read_bytes_per_s` and `write_bytes_per_s`. It rejects ambiguous units,
negative values, and non-monotonic timestamps. Keep the original native export
with the normalized file.

`classify-windows` consumes PMU CSV rows (`window_start_ns`, `window_end_ns`,
`aic_activity_pct`, `aiv_activity_pct`, optional `valid`), task JSONL
(`task_id`, `engine`, `start_ns`, `end_ns`), and dependency JSONL
(`task_id`, `state`, and source evidence). It labels a high-AIV/low-AIC window
as ready only if an observed AIC task has a recorded ready state.

`compare` consumes one JSONL row per variant/process replicate with
`variant`, `replicate`, `latency_ms`, `output_sha256`, and `valid`. It rejects
unpaired, invalid, or output-mismatched comparisons and only screens candidates
whose paired measured latencies all improve. This is not a significance test.

Example commands:

```sh
python tools/aic_aiv_followup.py kv-layout --input kv-layout.jsonl --output normalized/kv-layout.json
python tools/aic_aiv_followup.py hbm-csv --input native-hbm.csv --output normalized/hbm.json
python tools/aic_aiv_followup.py classify-windows --windows pmu.csv --tasks tasks.jsonl --dependencies dependencies.jsonl --output normalized/dependency-windows.json
python tools/aic_aiv_followup.py compare --input variants.jsonl --output normalized/variant-screen.json
```

## Acceptance and stop conditions

- Keep PMU only when at least 95% of expected bins are valid and the interval,
  clock mapping, and engine core counts pass the existing PMU quality checks.
- Keep profiler timing only when measured overhead is <=10%; higher-overhead
  traces may identify call paths but cannot support latency claims.
- Require three independent process restarts per performance condition,
  matching output hashes, stable baseline variation (CV <=5%), and no graph
  fallback. Do not average physical cards into a utilization curve.
- Stop layout screening if the metadata shows contiguous views with no
  associated Slice/copy, or if the control fails output equivalence.
- Stop cache-size screening if only capacity changes and neither Slice/task
  counts, HBM traffic, nor latency responds.
- Stop scheduler-candidate work when dependency evidence shows all candidate
  AIC work blocked, or when isolated co-run fails to improve end-to-end time.

The host currently exposes CANN `msprof` support for `--sys-hardware-mem` and
`--sys-hardware-mem-freq`; that confirms CLI capability only. The vLLM runtime
metadata hook, simultaneous HBM/PMU export, and dependency event source still
require validation in the pinned inference and training environments.
