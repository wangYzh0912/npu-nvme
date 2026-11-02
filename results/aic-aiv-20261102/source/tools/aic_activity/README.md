# AIC/AIV measurement tools

This experiment measures Qwen3-8B on physical NPU 4–7. It does not implement an acceleration technique. Experiments use an isolated copy of the repository; training supervisors record its source identity. Do not edit the source of an active supervised run.

## Metrics and acceptance

- `task_envelope`: per-card union of device task intervals. Five disjoint states are AIC-only, AIV-only, both, neither, and mixed/unknown. A task envelope is not hardware arithmetic utilization. Mixed kernels mask classification instead of counting as idle.
- `sampled_core_busy`: sum of busy cycles divided by core frequency and the capacity of **all** engine cores. Read native sample-based SQLite counters, discard each core's first sample, validate interval/counter capacity, and keep missing bins null. Display resolution is at least 10 ms.
- `phase` and `category` are separate. Communication can occur during backward or recompute; Contiguous/Slice is data movement even when executed by AIV. MindSpore's `recompute_` compiled scope also names the original forward invocation: tasks before the first explicit gradient task in a step are classified as forward. This boundary is an inference recorded in `phase_quality.json`.
- Inference stage labels follow scheduler snapshots and CPU scope → enqueue/dequeue → CPU/device correlation flows → exact device task keys. Unmatched tasks stay unclassified. Async host time overlap never determines device phase.
- No-profiler controls precede/follow each inference capture. Training compares steady steps before capture. Execution validation, warmup stability (CV ≤5%), profiler overhead (≤10%), and task-count anomaly checks are separate evidence. High-overhead traces are diagnostic. A count check alone cannot prove completeness.
- Six complete training windows require seven GetNext anchors. Six complete inference requests have exactly 128/512 input tokens and 32 output tokens per request, ignoring EOS. Three model-process restarts are the independent inference replicates; the six requests within one process are not six independent experiments.
- AIC/AIV task overlap and sampled engine activity cannot by themselves establish a legal scheduling opportunity or predict a speedup. A 10 ms sample can contain serial AIC and AIV work.

## Entry points

Use the repository root for offline tools (`PYTHONPATH=.`). Only `infer`, `worker`, and `calibrate` require the inference software stack; offline analysis uses the standard library. Plotting also needs matplotlib and numpy.

```sh
python -m tools.aic_activity.postprocess --kind training --run RUN --output ANALYSIS
python -m tools.aic_activity.postprocess --kind inference --run CASE --output ANALYSIS
python -m tools.aic_activity.render --summary ANALYSIS/summary.json --tasks ANALYSIS/timeseries.csv --stages ANALYSIS/stages.csv --output FIGURE --title TITLE
```

Inside the pinned vLLM container add the source `tools` directory to its existing PYTHONPATH. Preserve the CANN Python path. The current driver rejects FULL graph updates; select the explicit compatible PIECEWISE mode. Do not silently change to eager execution.

```sh
python -m aic_activity.infer --output OUTPUT --piecewise --matrix --mode task
python -m aic_activity.infer --output OUTPUT --piecewise --matrix --mode pmu
```

`--matrix --mode task` runs batch 1/8 × prompt 128/512 plus stochastic batch8/prompt128. Each model process is one independent repeat. PMU mode selects prompt128/batch1 and batch8 with a separate 35-second system collector and at least 30 seconds of requests. Task traces are exported **after** the worker process exits, since vLLM workers are daemons:

```python
from torch_npu.profiler.profiler import analyse
analyse('OUTPUT/CASE/trace')
```

Training uses the existing `train.py fit` entry point and its hardware lock. Do not wrap this command in another flock on that same lock. The optional `activity_profile` object accepts an interior start/stop step, `aic_metrics=pipe|none`, `profiler_level=0|1`, and `activities=cpu_npu|npu`; only `method=none` is supported. `train_sample` supervises unprofiled training and attaches a sample-only collector after warmup. Its PMU windows are clipped to actual per-core sampling bounds, not collector process lifetime.

`pmu.export(device_dir, output, windows)` produces counters and 10 ms curves; `statistics.core_conditionals` reports low thresholds 5/10/20% and high thresholds 30/50/70%. `statistics.aggregate` first time-weights windows within a run and then gives independent runs equal weight. Never average physical cards into a utilization curve.

## Outputs

`summary.json`, `timeseries.csv`, `stages.csv`, `events.jsonl.gz`, `phase_quality.json`, and `acceptance.json` preserve window definitions, engine states, attribution quality, and source hashes. PMU exports add `pmu_quality.json`, `core_timeseries.csv`, and `per_core_samples.jsonl.gz`. Figures are standalone PNG/SVG/PDF. Tmpfs contains raw exports; durable results retain compressed normalized events and counter inputs within the experiment storage budget.

The calibration is a real-device positive/negative control (idle, matrix, vector, serial, concurrent). Its serial and concurrent task traces establish that the measurement pipeline can detect the difference; neither a dictionary of device capabilities nor a synthetic plot would establish this.

Run focused checks with `pytest -q tests/activity tests/python/test_qwen_config.py`. These verify interval accounting, missing coverage, phase correlation across async delay, scheduler snapshot classification, and independent-run statistics. They supplement hardware acceptance rather than replace it.
