# Historical Pre-Cleanup Reproducers

These reproduce defects observed in the 2026-10-20 pre-cleanup working tree
(based on `3df53dee5ae1a23b3ccf43319d393413aab6d62a`). They are archived
evidence, not current test entry points. Do not run them against ABI3: the
drain private fields and FaF implementation they reference have been retired.

Current regressions are `tests/d1/test_runtime.py`,
`tests/python/test_full_closure_contracts.py`, and `tests/c2/test_retirement.py`.
See the root `FULL_CHECKPOINT_FLOW.md` for implementation and acceptance status.
