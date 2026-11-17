# M1 profiler-owner rerun diagnostic (2026-11-17)

Source commit: `7666596964ef4b76d7100b7e00b47a158904ef8a`.

This is a retained diagnostic capture, not an E4 pass. All four ranks ran under UID 22633, reached step 32, and recorded `device_profile.status=captured`. Their profiler outputs include positive-duration traces and 61,770 positive-duration device-task rows per rank; MAC, vector and MTE PMU columns were present. `device_summary()` verified each rank's CSV and trace, and `dependency_bound()` returned `verified` for the synchronous path.

The R3 campaign stopped after the run because its T0 oracle staging copied only initial state/control sidecars. The E4 comparison requires the T0 training trajectory and final step-32 state/control files too, so comparison raised `FileNotFoundError` before the campaign could validate and record the E4 capture. No hardware safety failure occurred; the rank workers and supervisor exited and released the lease. This run is excluded from E4 acceptance. R4 repeats the E4 group with the complete oracle staged and hash-verified.

`profiler/` contains per-file gzip copies of the raw device-profiler exports. `profiler-inventory.json` records each raw path, byte count and SHA-256 alongside the compressed archive path. `evidence/runs/e4-none/` retains the normal structured run archive.
