# Checkpoint I/O results snapshot

This snapshot records M0 and diagnostic acceptance, with M1 still running. The
implementation was frozen at `266bb35fe17b445a9c73ee09c6c496a4a870dfb0`.
The publication branch adds evidence only; the active execution worktree keeps
its original HEAD so the source-identity gate continues to hold.

| Stage | Result | Evidence |
| --- | --- | --- |
| M0 FULL correctness | Pass: 7 scenarios, 5 trajectory comparisons | [M0 result](snapshot/m0/m0-result.json) |
| Diagnostic T1/T2 | Pass: 32 rank/checkpoint streams, 204,032 matching chunks | [Diagnostic result](snapshot/diagnostic/result.json) |
| Fresh restore step 28 to 32 | Exact state/control and tolerant loss comparison pass on four ranks | [Diagnostic result](snapshot/diagnostic/result.json) |
| Event correlation | Eight save boundaries per method, zero missing pairs or physical mapping errors | [T1 timeline](snapshot/diagnostic/checkpoint_io_sink/timeline.json), [T2 timeline](snapshot/diagnostic/ours/timeline.json) |
| M1 | In progress; see the frozen result for completed processes | [M1 partial result](snapshot/m1/result.json) |

The M0 publication-interruption run intentionally returns a failed training
status. Its fault proof records injection after raw commit and before catalog
publication, its owner closed safely, and the following fresh process recovered
generation 1 and passed comparison. This is an expected acceptance case.

M1 has not completed its three independent repetitions. This snapshot establishes
no P95, performance improvement, device-engine attribution, or observation
overhead acceptance. Detailed logs/events and model data remain at the paths
recorded in the evidence. [manifest.json](snapshot/manifest.json) gives the snapshot
time, SHA256 and size of every copied file.

Earlier verification recorded 285 Python/CPU checks and 45 ASan/UBSan checks
passing; the later archive change passed 11 focused checks. These source-level
checks are distinct from the hardware acceptance above.
