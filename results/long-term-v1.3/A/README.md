# A safety implementation — incomplete stage

Implementation branch: `codex/long-term-v1.3`, isolated from the original dirty workspace.
`gate-001/evidence_manifest.json` validates the recorded host subset. The complete
A gate is **blocked**, not passed: request lookup/retention, detailed quarantine
ownership ledger, and the hardware shutdown gate remain unimplemented/unrun.

Implemented and exercised:
- Wait timeouts preserve request state and permit later completion.
- Paired admission leases, finite admission deadlines, close notification,
  capture-inclusive drain watermarks, owned mutex release, thread-start rollback,
  per-call admission, and explicit returned reset errors.
- C caller and queue/reactor references; context survives outstanding handles;
  cancellation publishes before dropping the queue reference. New bounded close
  retains native resources on incomplete shutdown. Failed DMA stop proof retains
  slots and blocks new I/O. Python retains unsafe snapshots across cyclic GC.
- Checked namespace/delta arithmetic, aligned item/byte budgets, metadata bounds,
  and restore shape/dtype/size validation before allocation.
- Strict result booleans/status and executable gate/evidence tools. Missing tests,
  skips, zero tests, timeout, artifact damage and path escape cannot pass.

Evidence:
- Original Python safety corpus: 11 failures (one timeout fixture was corrected
  to use METADATA_COMMITTING before the post-fix run).
- Original C sanitizer failures: see `c_sanitizer_before/result.json` and raw logs.
- Portable existing suite plus Python regressions: **111 passed**. Excluded
  baseline_repro, live_async_capability, r0_pipeline and s2_delta; this is not a
  full dependency/device suite.
- Gate subprocess counts, raw stdout/stderr, JUnit, profile and source hashes are
  in `gate-001`; C tests print actual compiler commands and binary hashes.
- Production CMake shared-library build passed with existing SPDK/CANN; no device
  initialization or raw NVMe I/O was performed.

Sanitizers use GCC 10 instrumentation with Clang 12 compiler-rt (ASan v8/UBSan).
`--no-export-dynamic` permits linker GC of uncalled hardware entry points.
The harness includes production `src/npu_nvme.c`; only external ACL/SPDK calls
are replaced. It proves software ownership under injected responses, not actual
DMA stop, driver behavior, concurrency memory ordering, or power-loss durability.

Remaining compatibility boundary: legacy synchronous raw-pointer APIs still
wait for buffer safety after their observation timeout. Removing that wait before
migrating callers would expose existing host/HBM buffers to use-after-free. The
new request API observes finite deadlines; caller release drops only its request
reference, never ownership obligations for data buffers. B2/C2 must migrate callers
before retiring the legacy functions. Live framework capture, request lookup,
detailed per-resource quarantine diagnostics, and full A gates remain incomplete.
