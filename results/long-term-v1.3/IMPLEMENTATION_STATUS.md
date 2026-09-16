# v1.3 implementation status — 2026-09-14

Implementation branch: `codex/long-term-v1.3`, isolated checkout
`/models/npu_nvme_exp/user7-stack/checkouts/long-term-v1.3`. Canonical plan:
`/home/user7/npu-nvme/docs/LONG_TERM_DEVELOPMENT_PLAN.md`. No push or master merge.
Original experimental checkout changes remain separate. Reviewer: self-reviewed.

| Stage | Accepted revision | Evidence / scope |
|---|---|---|
| A0 | 56b9c76 | baseline inventories; A0/ |
| A | 646760c | 70 software cases including 17 production C sanitizer cases; A/ACCEPTANCE.md |
| B | 327500e | 259 software cases; six bidirectional compatibility cases, seeds41/42/43; B/repair-20260914/README.md |
| D1 | 093fff6 | 116 software cases, strict H01 seeds41/42/43, H02 11 cases, five lifecycle extensions; D1/acceptance-002.json |
| Legacy retirement | 397408c + 590fab4 | Independent retirement and reporting fix; 298 portable regressions, 135 frozen software gate cases, H01 three seeds, H02 11 cases, all lifecycle extensions and two caller flows pass; retirement/acceptance-002.json |
| EN-native | c460b83 | deterministic TP4 native source/restore and complete controls; EN-native/gate-001, fixed Qwen scope |

D1 uses an explicit training specification, single publication owner, independent
request/checkpoint/catalog identities, two retained generations over three slots,
and reader pins through verification/apply/control readback/ready. Host staging
is chunk bounded (<=1MiB); control and decoded manifest budgets are 1MiB/16MiB.
Timeout observes ongoing work. Unproven safety retains target/source/native
resources and poisons admission. Query outcomes after restart remain unknown;
the V2 container does not claim durable per-request query indexing.

D1 was accepted only after joining software and hardware evidence at the same
source and native library. The common hardware library SHA256 is
`0f521c87a95b18de2d87f48417abd45b603414e8586ff568c226fc4b36fc3c7d`.
Full byte equality was checked immediately after restore; continuation losses
and final state met unchanged rtol=1e-5/atol=1e-6 for GPT-2 on NPU7.

The original D1 C_IMPL build failure was unresolved SPDK test stubs, while the
production library had built. SPDK thread exit also required a real fix: request
exit, poll to EXITED, join and destroy once. Native admission rejects an existing
owner before EAL/probe; stopped requests return ESHUTDOWN, quarantine returns EIO.
The earlier probe failure with a retained owner was not proven resource exhaustion.

All historical failures remain: B nondeterministic seed43 continuation,
D1/debug-001 control RNG ordering, and D1/lifecycle-001 closed-admission error
classification. MindSpore seed restoration now precedes restoring NumPy/Python
RNG, because set_seed mutates NumPy state. Positive later runs do not relabel
failed earlier evidence.

Lifecycle evidence covers 32 safe reopens, submit/close, withheld then delivered
NVMe callbacks, and injected event/stream-sync errors. Quarantine tests retained
production resources and separately proved a safe test-process exit. This is
not production quarantine recovery, general DMA-stop, or controller power-loss
acceptance. H02 crash cases are scratch/superblock smoke, not power cuts.

Only 0000:83:00.0 received raw writes. 0000:84:00.0 / /models remains the protected
filesystem device (evidence files use its filesystem normally). No whole-drive
scan is claimed. Mainline environment remains the documented old MindSpore2.5 /
CANN environment; E0 promotion and old/candidate rollback remain outstanding.

Rollback: `batch-B-exit-20260914-verified`, `batch-D1-entry`,
`batch-D1-exit-20260914-verified`, `archive-legacy-full-20260914`.
The archive is not an executable fallback. Migration details:
`docs/migrations/STRICT_FULL_RETIREMENT.md`.

Next after retirement validation: C1 unified training entry, failure propagation
and fair restore metrics; B2 directional async transport; then C2 default/ABI
retirement. TR-01/02/03, IO-01..04 and D2/E1/E2/F tasks are not marked complete
by these D1 or caller smoke results. Counts overlap; do not sum them as unique
coverage. Machine-readable task state: `config/execution_status.json`.


Final retirement acceptance: `590fab4`, exit tag
`batch-legacy-retirement-exit-20260914-verified`. The single-card FULL and Ours
caller manifests match the accepted Git revision, including experiment sources.
Ours restored full-state bytes exactly and matched all three continuation losses.
The initial caller reporting failure (`checksum` field) remains under
`retirement/callers-001`; `callers-002` is the independent passing rerun.
Eight acceptance-join regression cases additionally reject missing/mixed/unsafe
evidence. Later changes only index evidence, document status and harden evidence
validation; production source/native library identity remains the accepted revision.
Remote fetch completed: origin/master remains 5b164b6; no push/merge performed.
