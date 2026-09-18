# D2 final validation - 2026-09-18

Candidate ABI2 validation used only `0000:83:00.0` range `[1280, 1408) GiB`.
`0000:84:00.0` and the formal Qwen range `[256, 1280) GiB` were not opened by
the fault or rank harnesses. `config/d2_validation_region.json` carries the
required `purpose: validation` marker; the destructive harness rejects a
configuration without that marker.

The candidate environment is CANN 8.3.RC1 with
`libnpu_nvme.so.2.0` SHA256
`ba6dc8156be8d1f9c353d309e881afe6e0391916471be6152df390aa13c4d2b7`.
The repository environment manifest was corrected to select that ABI2 library,
rather than the obsolete library missing the required close interface.

## Passed hardware evidence

- `fault-campaign-retry/result.json`: named faults after payload, metadata and
  alternate-anchor flush; idempotent retry/expired request rejection; reader
  pin through retention rollover; newest-anchor corruption fallback; no-space
  pre-admission rejection. This is controlled flush-failure evidence only and
  does not claim power-loss safety.
- `rank-campaign-root/result.json`: candidate TP2 and TP4 save, source-process
  exit, fresh owner and rank-process restore. The fixture contains sharded,
  replicated and per-rank-control payloads and validates rank controls before
  global release. Every owner and rank process returned zero and every owner
  recorded `closed: true`.
- Candidate CPU gate: 41 focused D2 tests passed.

Evidence root: `/models/npu_nvme_exp/user7-stack/d2-final-validation-20260918-001`.
The first fault attempt is retained as diagnostic evidence; it exposed a
`bytearray` boundary in `RegisteredBackend.write`, fixed before the successful
retry. The first rank attempt is retained as diagnostic evidence; the native
library requires raw-device authorization in rank processes, so the final
candidate rank workers run with the same authorized account as the owner.

## Deferred migration precondition

`v2-migration/result.json` correctly refused the real device because its low
address does not contain a valid strict D1 superblock. The migration tool now
selects the highest valid strict D1 FULL record dynamically and keeps its
source read-only, but a real migration acceptance needs an independently
preserved strict D1 source image. Do not format the current low-address region
to manufacture that evidence.

The remaining D2 work is real-D1-image migration and injected multi-process
disconnect/owner failure campaigns. Qwen full-state E2 acceptance remains a
separate workload gate.
