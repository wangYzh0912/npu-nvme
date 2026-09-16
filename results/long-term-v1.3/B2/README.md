# B2 development evidence

Write pilot 001: 24 real HW1 cases passed on NPU7/raw83 scratch at 64 GiB.
HBM/Host × chunks 1/4/16 MiB × depths 1/4/8/16; 8 chunks plus 517-byte tail per case.
Actual readback and request-owned CRC32/SHA256 matched; HBM cases used ACL async directly
on SPDK DMA buffers with zero event-query errors or stream recovery. 24 C_IMPL cases
passed ASan/UBSan; 16 ABI/binding regression cases passed. These are development
results, not complete B2/H07/H01/H02 acceptance. Source identity is in the raw source.json.

Raw evidence: /models/npu_nvme_exp/user7-stack/b2-runs/write-pilot-001.

Roundtrip pilot 001 at 2ad04f7: all 24 configurations also passed the new
checksummed async read API. HBM readback used SPDK DMA buffers directly as ACL
H2D sources; target bytes matched exactly. C_IMPL added read corruption before
target write, pending-event ownership, late completion, record/query quarantine:
29 cases passed before the final profiling-only timestamp change. This remains
pilot evidence; final source/binary joined gates are pending.

Metadata request unification: 32 C_IMPL/layout cases passed (165.97 seconds),
including caller-released write dependency retention and failed-write flush
receipts. Native Release build passed. This software evidence does not replace
hardware metadata/flush or combined B2 acceptance.

Bounded work development: 33 C_IMPL/layout cases passed (171.34 seconds),
including multi-slice CRC/SHA continuation and idempotent completed digests.
Host copy and checksum each consume at most 64 KiB per direction/tick;
write DMA admission consumes at most four items/tick. Release build passed.
Scheduling configuration and request rotation are subsequent work.

Request rotation development: 35 C_IMPL/layout cases passed (181.97 seconds),
including read/write small-request progress ahead of a larger accepted request.
Rotation retains ownership and occurs only at a fully returned DMA prefix.
The shutdown loop was then extended to drain ready requests; Release rebuilt.
Final combined source verification remains pending.

Effective capability/configuration development: 38 C_IMPL/layout cases passed
(192.72 seconds), covering depth64 reporting, invalid budget rejection, global
pending admission, request rotation and cleanup. Release build passed. Expanded
C_IMPL suite exceeded the old 180-second orchestration budget; those gate budgets
are now 720 seconds, without changing individual test correctness requirements.
The next hardware probe includes metadata requests and depths32/64.

Flush failure history: three focused C_IMPL cases passed (21.17 seconds).
A context retains a prior failed write outcome even after its request has been
released; subsequent flush cannot turn that failure into a durable receipt.
Both old and candidate Release libraries build independently.

At a210d16: old and candidate each passed all36 scheduled direct-buffer cases
(72 total), including depth32/64, asynchronous metadata and durable flush receipts.
Largest payload:1,090,519,557 bytes; both directions reached64 owned slots.
Each worker records loaded library hashes and before/after resource snapshots.
Combined C_IMPL/layout39 passed (198.12 seconds); affected C1 software216 passed
with no changed sources. Read fault, H02 and full H01 combined gates remain pending.
The first C1 software launch failed to create its root-owned output directory;
the preserved retry used a new writable directory and passed.
