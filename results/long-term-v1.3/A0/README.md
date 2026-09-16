# A0 implementation baseline

Implementation starts at 5b164b6 in the isolated codex/long-term-v1.3 worktree.
The v1.3 planning document remains in the original development worktree; its
hash and input source hashes are recorded in inventory/base_manifest.json.

The portable baseline executed 84 tests successfully (cpu_baseline.json).
The inventory contains 391 C API source references, all 15 historical regression
seeds, existing mode mappings, local migration inputs, and C test limitations.
Historical regression descriptions are explicitly not new reproductions.

Raw hardware tests remain blocked: no current dedicated extent ownership has
been established. Historical offsets 64/128 GiB are not permission to overwrite
current data. A0 records that limitation; CPU safety repair can proceed.
No device I/O or environment changes were performed for this inventory.
