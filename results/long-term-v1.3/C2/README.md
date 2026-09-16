# C2 implementation evidence (not accepted)

Independent worktree codex/c2-async from e9e04da, while B2 final hardware is frozen.
One reactor now handles copy-only D2H/H2D via the SPDK DMA pool as well as NVMe,
metadata and flush. Strict FULL routes capture hashing, direct frozen write and
restore application through versioned requests. Unknown completion retains both
buffers and target; a target cannot publish ready or discard without stop proof.
C2a legacy bulk wrappers submit/wait the same kernel; synchronous ACL bulk branches
are removed. Frozen D2D snapshot and small synchronous control operations remain.

Copy C_IMPL3 passed; combined C_IMPL/layout42 passed; upper-layer141 passed;
ownership plus D1 protocol51 passed. These are software development results.
Actual C2 hardware, old/new comparison, ABI2 retirement and XL regression pending.

Bounded descriptor streaming:9 C2 tests passed, including a synthetic95,521-chunk
stream larger than64GiB, max4096 resident descriptors, logical tail, independent
checkpoint/request budgets and overflow. This is CPU geometry, not hardware.
Adapter/config/import regression99 passed. The C2 hardware probe additionally
checks copy-only D2H/H2D against independent byte oracles.

QW-06 preparation (not resource acceptance): explicit allocation deduplication,
per-rank HBM accounting, framework peak sampling, capture admission and phase
budgets added. Qwen training records actual HAL allocated/reserved peaks per step.
Five resource/schema tests match the actual882-field TP4 schema and
95,520/25,224/7,872 descriptors at1/4/16MiB. No Qwen peak measurement is claimed yet.

At f19e075, the affected B2/C1/D1 software profile passed275 cases with no changed
sources. C2's own profile additionally includes its new ownership/streaming/resource
cases. Frozen comparison keeps the old1MiB/depth4 configuration; the separate
async pilot uses the new4MiB/depth4 starting configuration. This makes old/new
comparison configurations equal without treating1MiB as a new capability limit.

Additional boundary fix before hardware freeze: reject overflowing pointer spans
before enqueue; failed metadata writes also poison subsequent durable flush
receipts. Four pointer/flush cases and one metadata-completion failure case passed.
Old/candidate libraries rebuilt. Final combined software rerun required.
