# v1.3 hardware evidence

Device `0000:83:00.0` (Huawei ES3000/ES3500P, 3.84 TB) was authorized for destructive testing and formatted as V2. Device `0000:84:00.0` backs `/models` and remains protected.

- `full-io-roundtrip`: PASS for Host, HBM, mixed I/O, flush, restart readback, and invalid size/offset rejection.
- `h01-c1-002` seed 41: PASS. Exact restored state; losses and final state allclose (final state not byte exact).
- `h01-c1-003` seed 42: PASS. Exact restored state; losses and final state allclose (final state not byte exact).
- `h01-c1-004` seed 43: PASS. Exact restored state; losses and final state allclose (final state not byte exact).
- `full-io-rerun-001`: PASS after H02 changes; Host/HBM/mixed/negative/context-restart matrix.
- `h02-isolated-r1`, `r2`, `r3`: PASS, 11/11 cases each (eight injected faults, backpressure, two crash-window smoke cases). Timeout returned `-ETIMEDOUT` at ~50 ms, short close timed out, extended close drained in ~406 ms, and every verify worker reopened the device successfully. H02 remains limited to this subset; no power-loss or DMA-stop proof.

Three isolated H02 rounds were completed with stable hugepage totals and bounded peaks (DMA 1, NVMe 1, request ring 16). The original CANN environment failure is preserved in `/tmp/v13-h01-c1.txt`; it occurred before sourcing Ascend `set_env.sh` and MindSpore selected CPU.

Original `h02-stage4` remains a failed partial run. The old “resource exhaustion” diagnosis was unsupported: short close retained the controller, then the same process attempted another attach. New isolation and explicit successful close remove that test error without changing the production ABI. See H02/ACCEPTANCE.md for missing coverage and launch failures.
