# GPT-2 baseline reproduction report

Project baseline: `f3c086157d2ce65d8b686cd938874b1b3541ec5d`.
Hardware and filesystem identity are captured in `preflight/preflight.json` and
each run's `environment.json`.

| Adapter | Result | Storage path | Restore evidence |
|---|---|---|---|
| `none` | trend measured | none | not applicable |
| `mindspore_sync` | 30-step trend measured | `/models` XFS filesystem | fresh process, byte exact, loss oracle pass |
| `ours` | build failed | raw SPDK requested | SPDK probe failed before generation |
| `datastates_acl` | NPU semantic port G3/G4 pass | real ACL D2H -> pinned Host -> durable XFS | tier ordering, bounded pool and source/persist split preserved |
| `pccheck_acl` | NPU semantic port G3/G4 pass | real ACL D2H -> bounded slots -> 4 MiB writer -> durable XFS | bounded writer/backpressure; XFS fsync/atomic publish replaces CLWB/SFENCE |
| `bytecheckpoint_host` | Host-adapted semantic port G3/G4 pass | ACL -> POSIX shared memory -> ByteCheckpoint CPU worker | model/optimizer planners, official extra-state workflow, shared 3-component counter |
| `fastpersist_host` | Host-adapted semantic port G3/G4 pass | ACL -> POSIX shared memory -> FastFileWriter/AIO | patched legacy storage-list hook; CPU byte-view Utils substitution; GDS disabled |

The completed semantic-port 30-step runs each produced ten generations and
passed fresh-process byte-exact restore plus the three-step loss oracle:

| Adapter | Wall time (s) | Generations | Restore |
|---|---:|---:|---|
| `datastates_acl` | 42.85 | 10 | byte exact; 3/3 loss pass |
| `pccheck_acl` | 42.20 | 10 | byte exact; 3/3 loss pass |
| `bytecheckpoint_host` | 62.48 | 10 | byte exact; 3/3 loss pass |
| `fastpersist_host` | 45.93 | 10 | byte exact; 3/3 loss pass |

These are semantic-port Host-file measurements, not CUDA artifact performance
claims. Shared-memory bridge events, worker hook evidence, durable writer
events and every request checksum are retained in each run's `events.jsonl`.
The common Host path is intentionally slower than the framework reference
because it exercises an explicit bridge copy.

The successful formal framework reference wrote ten 1.485 GB generations on
the `/models` test filesystem (about 14 GB), took 66.74 s for 30 formal steps,
and restored generation 10 (logical step 35) in a fresh interpreter.  All three
continuation losses matched the source oracle exactly under the configured
`rtol=1e-5`, `atol=1e-6` gate.  The no-checkpoint 30-step reference took
27.59 s and is retained only as a training wall-clock baseline.

Raw JSON, event timelines, state schema, fixture hashes, failure records and
restore logs are kept beside each run. The formal ACL schema contains 590
device tensor fields (196 model and 394 optimizer) plus one Host control
payload. All model and optimizer fields are copied from real NPU addresses
with `aclrtMemcpy`; no
formal tensor field falls back to `asnumpy()`. The source-release and durable
ACK timestamps remain separate (roughly 2.9 s DataStates, 3.0 s PCcheck, 10.0 s
ByteCheckpoint and 5.5 s FastPersist on this host), so API return is not
mislabelled as persistence. Earlier mechanism-only runs remain historical
downgrade probes and are excluded from formal comparison.
Early smoke and failed dependency probes are retained as historical diagnostics.
`summary.json` has a separate `formal_semantic_ports` table that requires 30
steps, ten `PERSISTED` generations, fresh byte-exact restore and the strict loss
oracle, so those records cannot be mixed into the formal comparison. Worker
versions, memlock requirements and probe status are locked in
`experiments/baselines/repro/worker_environment.lock.json`.
