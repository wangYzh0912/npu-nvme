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
