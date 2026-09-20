# D2 implementation and acceptance — 2026-09-20

Development branch: `codex/d2-completion-20260920`, based on `64730754`.
The original worktree and its ongoing experiments were left unchanged.

## Implemented

- Validation-only callers require the exact PCI, protected PCI, region ID,
  offset, length and explicit write authorization. A purpose marker cannot
  relabel the production region as scratch.
- Duplicate rank-session saves validate the retained step, topology and schema
  and return the prior receipt without re-entering payload transfer.
- Native D1 mount and read-only D1 images share the committed-superblock parser.
  Source selection validates the selected record rather than silently choosing
  an older record when it is damaged. Migration reserves capacity before writes
  and checks raw source metadata/data extents for destination overlap.
- A file fixture uses the real D1 FULL runtime and commit coordinator to save
  steps 4/8/12. It includes cross-chunk model and Adam payloads plus seven
  control fields. It does not claim historical Qwen recovery.
- Independent spawned processes exercise rank disconnect, owner exit after
  file fsync, and restore control rejection. All clients fail, the old commit
  remains selected, and a new process group restores successfully. This is
  durable-file/socket evidence, not SPDK DMA termination or power-loss proof.

## Results

Evidence root:
`/models/npu_nvme_exp/user7-stack/d2-final-validation-20260920-001`.

| Evidence | Result |
| --- | --- |
| `faults/result.json` | Seven controlled NVMe failure/retention/capacity cases pass |
| `ranks/result.json` | TP2 and TP4 save, process exit and fresh restore pass |
| `migration/result.json` | Strict D1 generation 3, step 12 migrated to D2 generation 13 |
| `migration-fresh/result.json` | Independent process verifies all 10 fields, 12,607,951 logical bytes |
| `formal-readonly/result.json` | Formal region unreadable: `invalid page header`; transport closed |

The migration source SHA256 is
`b9150979595627d87854d40e9fdb206c31a5667f077c4976d2552a47f944e9e4`;
full source hashes match before and after migration and verification.
All hardware writes were confined to `0000:83:00.0` `[1280,1408) GiB`.
The protected `0000:84:00.0` controller was not rebound or formatted.

Software command (candidate environment):

Result: **128 passed in 9.72s**. Machine-readable hardware summaries are
archived in `docs/results/d2-completion-20260920/`; the source image is retained
as `d1-source.img` under the external evidence root.

```bash
python3 scripts/run_user_environment.py --manifest config/user_environments.json \
  --profile candidate -- python -m pytest -q tests/d1 tests/d2
```

Hardware runners: `tools/run_d2_faults.py`, `tools/run_d2_sessions.py`,
`tests/hardware/d2_v2_migration.py` (first `--source-image`, then a separate
invocation with `--verify-generation 13`), and
`tests/hardware/d2_inspect_readonly.py`. Each uses a fresh output directory,
candidate ABI2 library and distinct SPDK shared-memory ID. Migration targets
only `config/d2_validation_region.json`; inspection uses
`config/d2_qwen_region.json` and exposes no write method to the region parser.

## Remaining execution order

1. Reconcile raw writers before rebuilding production media. Previous raw
   incremental experiments have extents overlapping the formal Qwen range;
   overlap alone does not prove the exact cause of the unreadable header.
   Preserve this read-only evidence and stop using historical catalogs.
2. Explicitly rebuild the formal Qwen region and rerun fresh FULL save,
   process restart, restore and oracle continuation. The current training
   implementation remains available; existing on-media checkpoints are not
   presently a usable recovery baseline.
3. Extend fault campaigns to real NPU/DMA rank and owner failures with retained
   resource evidence. Do not extrapolate file-process tests to DMA safety.
4. Keep exhaustive four-method timing, Qwen live/E2, resharding and incremental
   development under their separate gates in `DEFERRED_VALIDATION.md`.

This delivery advances D2; it does not mark full D2/E2 acceptance complete.
