# AIC/AIV measurement and KV applicability delivery

This branch publishes a compact, reviewable experiment package. It does not
claim a production optimization or general correctness of the dense-layout
control. The latest coordination status is
`investigation_complete_correctness_and_runtime_readiness_unresolved`.

- [Initial training/inference observations](aic-aiv-v2/REPORT.md)
- [KV cause and capacity experiment](aic-aiv-kv-20261103/REPORT.md)
- [Version, graph, layout and attention-path applicability](aic-aiv-scope-20261104/REPORT.md)
- [Acceptance including failed correctness checks](aic-aiv-scope-20261104/acceptance.json)
- [Readiness and dependency evidence limits](aic-aiv-scope-20261104/DEPENDENCY_SCOPE.md)
- [Inference graph/backend decomposition and new training observations](aic-aiv-coordination-20261105/REPORT.md)
- [Coordination acceptance gates](aic-aiv-coordination-20261105/acceptance.json)

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

The coordination package records FULL/PA versus FULL/FIA b1 output agreement
and close latency, unstable b8 measurements, failed repeated diverse-prompt
correctness for both backends, and training seq128/512 sample-only AIC/AIV
observations. Runtime producer/consumer readiness remains unknown, so no
real-model co-run or multi-request optimization gate was entered. These are
investigation results, not production optimization acceptance.

CPU validation: see the coordination package's test record. This is not a
replacement for device acceptance.
