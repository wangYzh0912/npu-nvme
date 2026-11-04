# AIC/AIV measurement and KV applicability delivery

This branch publishes a compact, reviewable experiment package. It does not
claim a production optimization or general correctness of the dense-layout
control. The final scope status is
`investigation_complete_correctness_unresolved`.

- [Initial training/inference observations](aic-aiv-v2/REPORT.md)
- [KV cause and capacity experiment](aic-aiv-kv-20261103/REPORT.md)
- [Version, graph, layout and attention-path applicability](aic-aiv-scope-20261104/REPORT.md)
- [Acceptance including failed correctness checks](aic-aiv-scope-20261104/acceptance.json)
- [Readiness and dependency evidence limits](aic-aiv-scope-20261104/DEPENDENCY_SCOPE.md)

The packages include reports, figures, normalized comparison data, runtime
identities, failure evidence and reproduction code. Each DELIVERY_MANIFEST.json
lists the compact files and their hashes. SHA256SUMS.json is the original full
archive manifest; it intentionally also lists raw files not uploaded here.
Complete archives remain at `/models/npu_nvme_exp/user7-stack/<package-name>`.
The latest archive's 4,129 manifest entries were checked with zero mismatches
before publication. Neither live supervisor log belongs to that manifest.

The isolated branch starts at origin/master 64730754. It excludes unrelated
uncommitted training/checkpoint changes and the two local FULL-closure commits.
Observation sources are taken from the frozen measurement snapshot; KV tools
and their focused tests are copied from the completed experiment implementation.
The training observation patch is preserved with the initial measurement
package and must be applied only to an isolated compatible training checkout.

CPU validation: 34 passed, 1 skipped (optional host torch unavailable).
This is not a replacement for device acceptance.

Next campaign order: inference layout/backend/graph decomposition; training
dependency observations; at most two legally independent co-run candidates;
fixed-arrival multi-request validation only after correctness and stability
gates. No ready-queue or reclaimable-FLOPS claim follows from low AIC busy.
