# M1 continuation result (2026-11-17)

The frozen execution source was `bfcabd3753148dafeea014c3d41c6a2a5bf2f67c`; the candidate native library hash remained `070f87d5bc3a2a32723274c9319e5b08356f5f36163e3ad3594b81a12daf4189`. The completed E1 runs were relocated to `/models` and all 528 files were checked by path, size and SHA-256 before reuse.

## Passed

- Revalidated T1 (`checkpoint_io_sink`) and T2 (`ours`) against the four-rank T0 control: final FULL state SHA-256/geometry and control state match exactly; losses satisfy `rtol=1e-5`, `atol=1e-6`.
- Revalidated full-boundary event coverage for T1 and T2, including all 32 rank×save boundaries. T1/T2 chunk streams match: 32 rank streams, 204032 chunks, 786399382656 logical bytes.
- Fresh step 28 → 32 restore ran on `/models`, selected committed generation 7, completed four ranks to step 32 and passed exact final state/control plus loss comparison against T0. Launch-to-safe-close was 1356.494 s (512 tokens, 0.3774 tokens/s); this includes startup, compilation, restore, validation and close, and is not a pure restore or training throughput measurement.
- The three reused E1 runs all had safe close evidence. Their original launch-to-safe-close values remain tied to their original `/dev/shm` execution and are not pooled with the `/models` run.

## Failed and pending

The queue stopped at the first E4 profiling run (`e4-none`), so `e1=pass`, while `e4`, `e2`, and `e3` remain `not_run`. All four ranks reached optimizer step 27, then MindSpore Profiler construction at step 28 failed with:

`PermissionError: CANN package user id: 22633, current user id: 0`.

The candidate CANN installation belongs to UID 22633 but the training ranks ran as root. The distributed job exited; the supervisor reports no remaining children or retained resources and released its hardware lease. NPU 4–7 were idle at inspection. No profiler CSV/trace was collected, so this is not an E4 result and no A/B observer matrix ran.

The raw failure result and archived failed-run evidence are included here. A follow-up profiling execution must run the profiler under the CANN-owning UID while preserving the root-owned raw owner lifecycle, or otherwise satisfy the runtime's ownership check. The frozen E1 results should remain unchanged; no claim is made that E4/E2/E3 passed.

## Profiler ownership fix and continuation

The runner now identifies the owner of the selected `ASCEND_HOME_PATH` and starts only the MindSpore distributed training worker under that UID/GID when device profiling is requested. The supervisor and raw owner remain root processes. The child receives the matching supplementary groups and HOME/USER/LOGNAME, and the run output directory is owned by the worker so rank logs and profiler exports are writable. This profiler-only run initially succeeded, but R4 showed native `copy_rank` workers also need to open the root-created `/tmp` advisory lock: owner lock files are `0644` under the system umask. The native lock helper now retries `O_RDONLY` on `EACCES`; Linux `flock(LOCK_EX)` works on that descriptor and still excludes concurrent readers/owners. A C regression makes the existing file read-only and verifies exclusive locking.

The profiler identity fix is frozen as `7666596964ef4b76d7100b7e00b47a158904ef8a`; the read-only native owner-lock source fix is frozen as `6e147e7e3e97172bf13edb6cc611d2f37b1d9bce`. Both are separate from the already-passed E1 source commit. A CPU-only subprocess probe under UID 22633 successfully imported MindSpore's CANN version check and returned `8.3.RC1`. Regression results after the native fix: Python/D1/D2 337 passed, C2 23 passed, and C host safety 45 passed, including the read-only flock regression. The first broad C2 invocation used a missing default `build/libnpu_nvme.so` path; rerunning against the unchanged, already-accepted candidate library passed all C2 tests.

The R5 hardware retry still used the original pre-fix candidate `.so`; the source-tree fix had not yet been rebuilt into the native runtime. It failed again at rank native init `-EACCES`, before any optimizer step, then safely released workers and lease. A new ABI3 library has now been built from the frozen source into `r5/execution/native-build/libnpu_nvme.so.3.0` (SHA-256 `7d37304203e1cf6e7760cfbe8796cb8d2396e355dce76a1611b8b307ab73543d`). Candidate preflight loaded that library, verified ABI major 3, and resolved its dependencies to CANN 8.3.RC1. The historical ABI3 library and manifest remain untouched. R6 points a private candidate manifest at this new library and reruns E4/E2/E3.

`continue_after_profiler_fix.py` starts a new `/models/...-r5` continuation. It imports the passed E1/restore gate with explicit old source identity and copies the complete small T0 comparison oracle (training trajectory plus initial/final states and controls) before running E4 and E2/E3 with the new source commit. R3 reached step 32 on all ranks and exported profiler files under UID 22633, but stopped because the orchestration copied only initial oracle files. R4 verified E4 `none`, then showed the root-created advisory lock denied `O_RDWR` to rank workers; both attempts are archived separately and are not complete E4 gates. The script waits for NPU 4–7, the hardware flock and lease before initialization and before every run; it does not terminate other workloads. The initial r2 profiler ownership failure remains preserved.
