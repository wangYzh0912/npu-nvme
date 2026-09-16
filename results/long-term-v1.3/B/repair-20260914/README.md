# B compatibility acceptance — 327500e

Software: 259 passed, zero skips/errors/failures and no changed sources (`software/result.json`).
Hardware: deterministic MindSpore ON before graph construction, GPT-2/NPU7/raw 83.
`hardware-on/result.json`: entry 646760c → exit 327500e and exit → entry, seeds 41/42/43, six passes.
`diagnostic-on/result.json`: entry → entry and exit → exit, seed43, two passes.
Original tolerances are unchanged: rtol=1e-5, atol=1e-6.

The earlier nondeterministic seed43 continuation failure remains in
`../completion-20260914/hardware-002/entry-to-exit-s43/restore.log`.
It is a failed historical run, not reclassified as a pass. Deterministic mode
makes the compatibility gate reproducible for this workload.

The earlier C_IMPL build failure was missing SPDK lifecycle stubs, not a failed
production library compile. SPDK exit is asynchronous: poll until EXITED, join,
then destroy exactly once. A second probe while a prior context remained owned
was not evidence of resource exhaustion. Native admission now rejects that
conflict before EAL/probe. This B gate does not accept strict D1 or retirement.
Large NumPy oracle files remain local; compact JSON/logs and hash indexes retain
the provenance. Rollback: batch-B-exit-20260914-verified, batch-D1-entry.
