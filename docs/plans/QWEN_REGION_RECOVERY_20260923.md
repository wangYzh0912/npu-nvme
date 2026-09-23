# Formal Qwen region recovery — 2026-09-23

## Accepted result

Qwen3-8B TP4 on NPU 0–3 again has a usable on-media FULL baseline.
Candidate MindSpore 2.7.1 / MindFormers 1.7 / CANN 8.3.RC1 and the existing
ABI2 library were used. No driver change, reset or host reboot was required.
All workload runs used source commit `804590ce`; the local MindSpore address
extension was rebuilt with `tools/build_mindspore_address.py`.

- D1/D2 regression: 128 passed in 10.07 seconds.
- Formal region initialization, small save and independent-process verify: pass.
- Uninterrupted `none` oracle through step 24: pass.
- Ours TP4 source through step 8, FULL commits at steps 4 and 8: pass.
- New owner and rank processes restored step 8, continued through step 12,
  saved another FULL and exited: pass.
- All four ranks matched oracle state SHA256/geometry and controls exactly;
  loss comparison passed with rtol=1e-5, atol=1e-6.
- Both owners report `closed: true`; the hardware lease was released.
- Final independent read-only metadata inspection: pass.

FULL save times observed on rank 0: 520.5, 526.0 and 523.0 seconds. These are
single-run correctness observations, not a performance comparison campaign.
Catalog generations 1/2/3 correspond to steps 4/8/12. Media generations are
2/3/4 because the initial small probe used media generation 1.

Evidence root: `/models/npu_nvme_exp/user7-stack/qwen-region-recovery-20260923-001`.
Compact machine-readable reports: `docs/results/qwen-region-recovery-20260923/`.

## Media diagnosis and boundaries

The pre-initialization 12 KiB header SHA256 matched the failed September 20
inspection: `01f06149be28c4bf5d13f34f7c2956dcd0762c678b91f045ec05209de921ebf5`.
The existing explicit `initialize --expected-header-sha256` operation accepted
the non-D2 header and rebuilt only `0000:83:00.0` `[256,1280) GiB`. Old header
bytes are preserved in the external preflight and initialization directories.
The 3,840,755,982,336-byte namespace and controller bindings were checked.
`0000:84:00.0`, containing `/models`, was not rebound or formatted.
The old catalog is not a recovery source; use only this campaign's new catalog.

The dirty main worktree contains an incremental archive allocator spanning the
namespace and a historical whole-device authorization. Its address space can
overlap Qwen. This establishes a conflicting writer configuration, not forensic
proof of the particular write that damaged the header. No such job was running.
Hardware jobs were serialized with the shared campaign lock.

This branch revokes `config/raw_test_region.json` whole-device write permission.
Explicit D2 registrations remain enabled. This is a caller configuration guard,
not an OS-level block on root or direct native I/O. The original dirty worktree
was preserved and still contains its old configuration: do not run its raw
experiments while retaining this Qwen baseline. Port them to non-overlapping
regions and the same lock before reuse. Disabling old whole-device tests here
is intentional; their authorization checks must fail before opening transport.

## Reproduce and continue

Use `train.py fit --method none --stop-step 24` for the oracle, then Ours
`fit --stop-step 8`, followed by `verify-restart --stop-step 12` with the new
source checkpoint root and oracle directory. All use
`--config config/qwen_training.json`, separate fresh output directories, and
root for Ours. The accepted source catalog has now advanced to step 12; further
resume must use a target greater than 12 and no greater than LR horizon 24.
Never rerun initialization against the now-valid D2 header.

At completion `/models` had about 31 GiB free. Conversion weights consume about
16 GiB per run. Budget storage before starting further runs; no old artifacts
were deleted to obtain space.

## Remaining work

The plan's single-NPU/eight-NPU Qwen claims were corrected to the implemented
TP4 workload. Single-device transport acceptance does not prove single-device
Qwen training. Other Qwen topologies require implementation and separate gates.

This cycle closes the formal-media usable-mainline regression. The full
step-24 multi-restart chain, exhaustive four-method timing, real NPU/DMA fault
injection, E2/live closure and incremental work remain deferred. File/socket
failure tests do not prove safe DMA termination. No full D2/E2 closure is claimed.
