# KV conversion observation on the pinned Ascend stack

This experiment identifies which operator creates Contiguous/Slice work, then
separates allocated cache capacity from effective context length. It measures
Qwen3-8B, BF16, TP4, physical cards 4–7, explicit PIECEWISE graphs. It does not
modify the shared vLLM container, change the model, or implement an optimization.

The supervisor uses the retained `user7-aic-aiv-030-priv` experiment container,
holds `/models/npu_nvme_exp/user7-stack/hardware-campaign.lock`, and snapshots
its experiment Python sources before each stage. Each model process receives
that immutable source path; editing repository files does not alter active runs.
Failure logs are retained and failed model jobs stop their owned container so
orphan workers cannot retain NPU memory. Do not run a second model campaign
against the same card group.

Durable output is `/models/npu_nvme_exp/user7-stack/aic-aiv-kv-20261103`.
Raw exports are in the experiment container's `/output/kv-20261103`, backed by
`/dev/shm/user7-aic-aiv-20261031/inference/kv-20261103`. The retained original
experiment container maps `/models` read-only and `/output` read-write. Large
artifacts and compilation caches stay in tmpfs; the durable package limit is
1 GiB. No baseline result directories are removed.

The metadata probe records cache allocation and bounded first-layer branch
metadata without copying NPU tensor contents. Instrumentation is restored
before timing. The capacity campaign makes nine independent model launches,
with three cache capacities in rotating order. Each process performs stable
warmup, six unprofiled controls, six task-profile requests, a separate >=30 s
simultaneous PMU/HBM segment, then six controls. Task exports occur after model
workers exit. Six requests in one process are not six independent repeats.

The work campaign makes three independent model launches and measures the
eight b1 combinations of input 128/512 and output 1/2/8/32, plus b8/input128/
output32, at fixed actual capacity. The isolated FIA replay uses the measured
layout, heads, and a native FIA parameter probe; it compares strided/dense
inputs and separately charges conversion. Synthetic cache values are finite,
and dense views are refreshed after active-cache updates. Its three timing
batches are within-process repeats and are labeled as such.

Example entrypoints (each stage creates a new source snapshot and log):

```sh
python -m tools.aic_kv.campaign --stage calibration
python -m tools.aic_kv.campaign --stage metadata --tag a2
python -m tools.aic_kv.campaign --stage capacity --metadata-run metadata-a2
python -m tools.aic_kv.campaign --stage metadata --tag native
python -m tools.aic_kv.campaign --stage work --metadata-run metadata-a2
python -m tools.aic_kv.campaign --stage micro --metadata-run metadata-native
python -m tools.aic_kv.reduce --raw /dev/shm/user7-aic-aiv-20261031/inference/kv-20261103 --output /models/npu_nvme_exp/user7-stack/aic-aiv-kv-20261103/analysis
PYTHONPATH=/tmp/user7-aic-aiv-20261031/plotdeps python -m tools.aic_kv.report --root /models/npu_nvme_exp/user7-stack/aic-aiv-kv-20261103
```

`provenance` consumes a completed case with native task CSV and trace JSON,
joins CPU/device flow IDs to exact device endpoints, and stores compressed
per-task evidence. PID/thread namespaces and stream/task/time identity are
retained. Temporal adjacency does not establish a caller.

`hbm` reads the pinned CANN 9.1 `HBMOriginalData` SQLite schema. The installed
parser already maps HBM timestamps to absolute time, unlike the raw AIC/AIV
sample tables. Each HBM count represents 32 bytes according to the captured
CANN `HbmModel.SCALE`; counters are per-interval, not cumulative. It discards
the first interval and requires four channels with both read/write counters.
Only native 20 ms interval means are exported; request boundaries use weighted
overlap, with coverage recorded. The raw DB and parser/model identities are
preserved. Logical Slice output bytes are reported separately from HBM bytes.

Focused CPU checks:

```sh
pytest -q tests/python/test_aic_aiv_followup.py tests/python/test_aic_kv_evidence.py
```

These checks verify data handling, not hardware functionality. Real calibration,
native counters, output tokens, allocator geometry, timing perturbation, and
independent model runs provide hardware acceptance. Unknown attribution and
failed timing gates remain explicit; no utilization or speedup is invented.

## Applicability and real cache-layout controls (2026-11-04)

`python -m tools.aic_kv.scope --stage screen` compares the local release-tagged
0.30 image with the prior custom image, then tests PIECEWISE, DEFAULT,
FULL_DECODE_ONLY, NONE and explicit PA shapes. The actual attention/cache source
hashes are compared: identical sources are not counted as distinct versions.
The 0.18 image has a separate, explicit observation adapter and is tested by
`--stage version`; source inspection and a basic device operation are not
substitutes for a model run.

`--stage confirm` measures Qwen3-8B/4B with native and dense layouts and, if the
PA screen succeeds, native FULL_DECODE_ONLY with `pa_shape_list=[1,8]`. Each
condition has three independent model processes and four cases: b1/input128/
output1, b1/input128/output32, b1/input2048/output32 and b8/input128/output32.
`--stage extend` adds b1/input128/output128 and b8/input2048/output32 on 8B for
all three paths and three independent process repetitions; untested cross
products are not marked as measured. `--stage measure` rotates native/dense/qualified PA
conditions at 6616 blocks with three process repeats and simultaneous PMU/HBM.
The cache-layout flag is experiment-only: before the runner binds its freshly
zeroed, unpadded ordinary GQA allocation, the same storage is reinterpreted as
all-K/all-V. Every subsequent native cache writer and reader sees the new
views. There is no shadow cache or per-decode conversion. Connectors, cache
restore, hybrid/quantized caches and production deployment are outside this
control's contract.

`--stage corun` runs three independent synthetic matrix/copy controls. Its
separate-stream timing and task-envelope overlap do not imply that equivalent
independent work is ready in the model. Without a native readiness/dependency
source, model scheduling conclusions remain unknown.

New evidence is isolated under
`/models/npu_nvme_exp/user7-stack/aic-aiv-scope-20261104`; raw data and compiler
caches use `/dev/shm/user7-aic-aiv-20261031/scope-20261104`. New containers are
`user7-aic-scope-official`, `user7-aic-scope-current` and
`user7-aic-scope-v018`; shared service containers and the previous evidence
package are unchanged. The durable package budget is 2 GiB. Each stage locks
cards 4–7, snapshots code and preserves failed runs with their original cause.
Unsupported graph updates are environment failures, never negative copy
observations. Zero Slice tasks alone do not establish absence of all traffic.


The scope supervisor can resume after an active owned stage with
`python -m tools.aic_kv.scope_finish --screen-pid <pid> --remaining-stages corun correctness hybrid`.
It does not duplicate completed runs. Model-run failures retain their cause and
are never recorded as no-copy negatives. Finalization refuses success if the
required model matrix, varied-prompt correctness or PMU/HBM acceptance is
incomplete. `--finalize-only` rebuilds the compact evidence/report from completed
runs without launching model jobs.

`--stage correctness` uses eight distinct fixed-length prompts and compares
per-request greedy outputs through repeated cache allocation/update/reuse,
with real engine TTFT/decode timestamps enabled. It is not a performance claim
because it adds metrics to the main benchmark configuration. The optional
hybrid architecture probe runs native Qwen3.5-27B only; it never applies the
ordinary-GQA dense-view experiment to a hybrid cache contract.

## Coordination decomposition (2026-11-05)

`python -m tools.aic_kv.coordination --stage inference-cold` uses independent
AOT/compiler caches for every run. The experiment-only
`--force-fia-with-pa-cache` changes attention_v1's runtime dispatch before
compilation; allocation still uses the official PA eligibility predicate.
It requires FULL_DECODE_ONLY with explicit PA shapes. Actual decode kernels
must be checked in a device trace before accepting the backend intervention.
Reusing a prior AOT cache is not an accepted intervention check.

`--stage piecewise-cold` records three dense PIECEWISE controls;
`--stage diagnostics` records metadata, task and simultaneous PMU/HBM probes.
Results use a new coordination directory; earlier archives stay immutable.

The follow-up window classifier now requires run/device/stream/task/start
identity, a timestamp, an explicit source and all_dependencies_observed=True
for a ready claim. It examines pending tasks which may start after the
window, ends states at later observations, and prevents reuse of numeric task
IDs across devices/streams/executions. Legacy observations without those
fields are unknown; no compatibility inference invents readiness.

`--stage piecewise-pa-cache` changes only allocation eligibility for ordinary
unpadded GQA: the official allocator creates separate dense K/V tensors while
PIECEWISE runtime dispatch remains FIA. It checks the allocation-ownership
confound of comparing shared-storage dense views with native separate PA
allocations. The first repeat includes a diagnostic task and block-table probe.
`--stage correctness-full` retains strict diverse-prompt repeated-output checks
for native FULL/PA and experimental FULL/FIA separately. Neither a successful
fixed-prompt request nor an official allocator constitutes general correctness.

Training dependency probes use an isolated copy of the frozen v4 observation
source with the explicit `AIC_AIV_SAVE_DEPENDENCY_GRAPHS` hook. Invoke
`python -m tools.aic_kv.training_dependency_campaign --source SOURCE` only
after the inference campaign has released cards 4–7. The training entry point
owns its hardware lock; do not wrap it in a second flock.
`graph_evidence` archives static operands and assigned stream metadata; it
does not derive runtime readiness from static stream IDs.
`archive_training_dependency` retains sample counters, identities, quality and
training outputs, excluding regenerable converted-weight/compiler caches.

## Training JIT and workload investigation (2026-11-09)

Use a fresh isolated copy of the frozen training observation source and apply
the archived training-matrix.patch. It adds experiment-only jit_level and
micro_batch inputs; the production checkpoint entry point is unchanged.
The workload remains Qwen3-8B, TP4/DP1, FP32 parameters/states, BF16 compute,
sequence parallel and recomputation. Raw outputs use a dedicated tmpfs directory.

training_jit_campaign supports O1/O2 smoke and baseline stages, the O1
prevalence matrix (seq128/512/2048 x micro-batch1/2 x three independent
processes), and a single o0-hbm supplemental system control. It refuses
overwritten outputs. --retry-lock only admits the retained pre-model lock
rejection and writes a distinct retry directory. Compilation failures are not
negative performance observations.
--retry-interrupted separately admits a documented SIGTERM before preflight,
configuration creation or rank training output; the original attempt remains
in the failure archive and the retry has a different directory.
o0-hbm-confirm records an independent same-configuration confirmation if a
supplemental HBM quality gate fails; the original failed-quality card stays in
the comparison data. It does not silently replace or relabel the earlier run.

training_fragment_campaign compares installed AdamW O0/O1 and official
FusedAdamW on three actual parameter-shard shapes. --resident-gradients
captures the fixed gradient as a device Parameter and records a separate suite.
The ordinary Tensor-input suite includes repeated input transfers and must not
be interpreted as resident model optimizer timing. Both check parameter and
first/second moments against an eight-step Float64 oracle. Diagnostic profiler
captures follow all unprofiled timing blocks.

Keep model and fragment work in one serial device queue. They share the same
hardware lock. Do not insert fragments during an offline exporter: the next
supervised model may already be launching. PMU and HBM share one system
collector; per-task profiles are separate diagnostic executions. CANN 8.3 HBM
needs the device-to-host delta from native clock anchors before adding the wall
clock offset.

training_fragment_audit extracts TensorMove geometry and backward MatMul pairs
from archived final IR; these do not prove measured traffic or runtime readiness.
It also preserves the global-clipping AddN/AllReduce path into the optimizer,
which constrains immediate layerwise optimizer/backward overlap.
training_jit_report records low-AIC, high-AIV, joint and conditional fractions
separately and never pools physical cards. training_jit_render emits standalone
figures and a report. training_jit_finish checks required matrix/fragment
completeness, refuses active devices or leases, and writes bounded manifests.
training_counter_replay reconstructs PMU SQLite inputs from retained replay0
columns and rebuilds PMU/HBM tables without reading the original raw directory.
Finalization runs this archive-only replay before freezing the package.
training_model_fragment uses the existing training activity_profile hook for
a short separate O1 task diagnostic. training_model_fragment_report joins
scoped kernels to native task types by stream, task ID and exact timestamp,
checks all native tasks overlapping the optimizer tail (including boundary
crossers), and draws a phase/task figure. Its high profiling overhead excludes
it from the normal PMU matrix and normal critical-path claims.
