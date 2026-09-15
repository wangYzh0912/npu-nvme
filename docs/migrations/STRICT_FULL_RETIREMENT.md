> 本文记录 590fab4 的首次 FULL 退役验收。后续 ABI 2/旧导入直接删除以 [完整清理](../LEGACY_IMPLEMENTATION_CLEANUP.md) 为准。

# Strict FULL migration

D1 accepted revision: `093fff6`; combined evidence:
`results/long-term-v1.3/D1/acceptance-002.json` (116 software cases,
H01 seeds41/42/43, H02 11 cases, lifecycle five cases including 32 reopens).
Rollback/archive: `archive-legacy-full-20260914`, `batch-D1-exit-20260914-verified`.
The archive is a Git version, not a second executable backend.

`DirectCheckpoint` now selects strict FULL by default. The `direct_checkpoint`
module alias retains canonical identity. Constructor `strict=False`, multiple
ranks, retention other than2, or multiple pending FULLs are rejected before
framework/device initialization. There are three physical FULL slots, two
retained generations, one pending FULL, one active restore, and chunks <=1MiB.

Save requires `save_state(components, controls, step, expected_spec=spec)`.
Build the specification with `training_spec` from explicit workload identity,
model/optimizer parameters and all control applicability decisions. Only frozen
FULL is supported. Restore uses
`restore_full_state(target_factory, expected_spec, step, deadline=...)` and
returns a ready target and receipt. The factory creates a private fresh graph;
never pass an existing training model for in-place mutation. Integrity checks
are mandatory, including in timing runs. Observation timeout is not cancellation.

The V2 container and envelope v2 remain. Envelope v1 and nonstrict catalogs are
rejected; there is no automatic conversion or fallback. The formatter defaults
to strict, with `--keep-last-n 2` meaning logical retention, and initializes three
physical slots. Formatting is destructive and the current utility accepts only
the registered scratch namespace 0000:83:00.0. Never format 84:00.0 or /models.

Supported callers migrated: `run_single_card_full.py` (serial waits for frozen
FULL completion, queue/frozen_async use the same strict contract) and Ours in
`experiments/baselines/repro`. Ours configuration explicitly records its own
1MiB/two-retained/one-pending limits; filesystem/reference adapters retain their
independent configuration and algorithms. The old fixed baseline commit field
remains a historical reference; actual source identity is recorded separately.

Legacy raw heap export and R0 writer/reader reject use. FaF/Delta/multirank
checkpoint entry points reject pending their own D2/G05/H03 acceptance. Pure
Delta algorithms and independent reference baselines remain available. The
existing public C batch ABI remains until the separate B2/C2 work.

The four old admission/catalog/in-place restore test modules were removed with
the implementations. Their accepted historical versions are in the archive;
current coverage is `tests/d1/test_runtime.py`, `test_catalog_crash.py`,
`test_strict.py`, plus `tests/acceptance/test_retirement.py`. Historical B/A
profiles map those removed paths to strict successors; their historical gate
semantics and evidence must be obtained from the archive, not rerun as a claim
of legacy compatibility on this version.

Quarantine remains conservative. Injected sync failures test retention and
independently safe test-process exit, not production quarantine recovery,
controller power-loss safety, or a general DMA-stop guarantee.

Retirement accepted at `590fab4`: `results/long-term-v1.3/retirement/acceptance-002.json`.
135 software gate cases, full H01/H02/lifecycle reruns and both migrated caller
flows passed. Exit tag: `batch-legacy-retirement-exit-20260914-verified`.
