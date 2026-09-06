# GPT-2 baseline reproduction report

Project baseline: `f3c086157d2ce65d8b686cd938874b1b3541ec5d`.
Hardware and filesystem identity are captured in `preflight/preflight.json` and
each run's `environment.json`.

| Adapter | Result | Storage path | Restore evidence |
|---|---|---|---|
| `none` | trend measured | none | not applicable |
| `mindspore_sync` | 30-step trend measured | `/models` XFS filesystem | fresh process, byte exact, loss oracle pass |
| `ours` | build failed | raw SPDK requested | SPDK probe failed before generation |
| `datastates_acl` | mechanism-only G2 passed | common Host bridge on `/models` | locked source requires CUDA/nvcc/liburing; upstream ACL core not invoked |
| `pccheck_acl` | mechanism-only G2 passed | common Host bridge on `/models` | locked source is CUDA/x86 (`clwb/sfence`); ARM writer not invoked |
| `bytecheckpoint_host` | mechanism-only G2 passed | common Host bridge on `/models` | CPU torch 2.5.1 import probe passes; planner/engine is not connected to MindSpore |
| `fastpersist_host` | mechanism-only G2 passed | common Host bridge on `/models` | CPU torch 2.6.0/AIO worker is prepared; FastFileWriter serializer is not connected |

The completed 30-step mechanism-only runs produced ten generations and passed
fresh-process byte-exact restore plus the three-step loss oracle:

| Adapter | Wall time (s) | Generations | Restore |
|---|---:|---:|---|
| `datastates_acl` | 81.09 | 10 | pass |
| `pccheck_acl` | 82.26 | 10 | pass |
| `bytecheckpoint_host` | 83.35 | 10 | pass |
| `fastpersist_host` | 82.37 | 10 | pass |

These are Host-file mechanism measurements, not CUDA artifact performance
claims. Shared-memory bridge events (`ipc_begin`/`ipc_ready`/`ipc_receive`),
durable writer events and every request checksum are retained in each run's
`events.jsonl`. The common Host path is intentionally slower than the framework
reference because it exercises an explicit bridge copy.

The successful formal framework reference wrote ten 1.485 GB generations on
the `/models` test filesystem (about 14 GB), took 66.74 s for 30 formal steps,
and restored generation 10 (logical step 35) in a fresh interpreter.  All three
continuation losses matched the source oracle exactly under the configured
`rtol=1e-5`, `atol=1e-6` gate.  The no-checkpoint 30-step reference took
27.59 s and is retained only as a training wall-clock baseline.

Raw JSON, event timelines, state schema, fixture hashes, failure records and
restore logs are kept beside each run. Mechanism-only runs deliberately
exercise the common NPU capture, shared-memory bridge, durable Host writer and
fresh-process restore, but do not claim the corresponding CUDA planner or
serializer. The two early smoke runs are retained as historical diagnostics;
formal records include `storage_backend` and `kind` so they cannot be mixed
into a performance comparison accidentally. Worker versions and probe status
are locked in `experiments/baselines/repro/worker_environment.lock.json`.
