# M1 R4 partial hardware result (2026-11-17)

R4 used the profiler-owner source `7666596964ef4b76d7100b7e00b47a158904ef8a` and the corrected complete T0 oracle.

The `e4-none` run completed all four ranks through step 32, safely closed, passed its T0 loss/state/control comparison, and passed device CSV/trace validation for all ranks. Each rank exported 61,770 positive-duration device-task rows, positive trace durations, and MAC/vector/MTE metric columns. The four profiler outputs are archived as gzip copies under `profiler/`, with source SHA-256 and sizes in `profiler-inventory.json`.

The next E4 run, `e4-checkpoint_io_sink`, failed before its first optimizer step because user7's rank processes could not open the root-created cross-process native-owner lock file with `O_RDWR`; native init returned `-EACCES`. The failed ranks, supervisor and native owner all exited safely; no worker remains and the hardware lease was released. Consequently R4's overall E4 gate is not complete, and E2/E3 were not started.

The source fix for that failure is in execution commit `6e147e7e3e97172bf13edb6cc611d2f37b1d9bce`. It falls back to `O_RDONLY` only when opening an existing owner-lock file returns `EACCES`; Linux permits `flock(LOCK_EX)` on that descriptor. The C regression creates a mode-0444 lock file and verifies read-only open and exclusive flock. The fix passed 45 C host-safety cases, 337 Python/D1/D2 cases, and 23 C2 cases. R5 will repeat E4 under this new frozen source identity.
