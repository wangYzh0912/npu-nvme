# GPT-2 baseline reproduction report

Project baseline: `f3c086157d2ce65d8b686cd938874b1b3541ec5d`.
Hardware and filesystem identity are captured in `preflight/preflight.json` and
each run's `environment.json`.

| Adapter | Result | Storage path | Restore evidence |
|---|---|---|---|
| `none` | trend measured | none | not applicable |
| `mindspore_sync` | 30-step trend measured | `/models` XFS filesystem | fresh process, byte exact, loss oracle pass |
| `ours` | build failed | raw SPDK requested | SPDK probe failed before generation |
| `datastates_acl` | build failed | not attempted | locked source requires CUDA/nvcc |
| `pccheck_acl` | build failed | not attempted | locked source is CUDA/x86 and missing `main.cpp` |
| `bytecheckpoint_host` | dependency blocked | not attempted | dedicated CPU torch worker missing |
| `fastpersist_host` | dependency blocked | not attempted | dedicated CPU torch worker missing |

The successful formal framework reference wrote ten 1.485 GB generations on
the `/models` test filesystem (about 14 GB), took 66.74 s for 30 formal steps,
and restored generation 10 (logical step 35) in a fresh interpreter.  All three
continuation losses matched the source oracle exactly under the configured
`rtol=1e-5`, `atol=1e-6` gate.  The no-checkpoint 30-step reference took
27.59 s and is retained only as a training wall-clock baseline.

Raw JSON, event timelines, state schema, fixture hashes, failure records and
restore logs are kept beside each run.  The two early smoke runs are retained
as historical diagnostics; formal records include `storage_backend` so they
cannot be mixed into a performance comparison accidentally.
