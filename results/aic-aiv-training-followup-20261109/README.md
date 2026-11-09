# Training AIC/AIV investigation delivery, 2026-11-09

Start with [REPORT.md](REPORT.md), [measurement contract](MEASUREMENT_CONTRACT.md),
[official solution audit](official/AUDIT.md), [acceptance](acceptance.json) and
[next experiments](NEXT_STEPS.md). Standalone PNG/SVG/PDF figures are in figures/.
The package records an investigation, not accepted production optimization.

Twenty completed system-sampling model processes include the registered18-run
O1 matrix and two separate O0/HBM controls. The first O0 control has one failed
HBM-quality card and remains recorded. The O2 compilation failure is retained.
Two optimizer suites contain27 independent processes each; the resident suite
is primary, ordinary Tensor input is a transfer diagnostic. A separate40-step
real-model task capture identifies the optimizer tail with high profiler overhead.
Raw source identifiers, counters, clock anchors and all failed/diagnostic outcomes
are preserved. Four cards from one process are reported separately, not counted
as four independent model runs.

Rebuild analysis in the repository with Python3/matplotlib:

```sh
python -m tools.aic_kv.training_jit_report --root PACKAGE
python -m tools.aic_kv.training_fragment_report --root PACKAGE
python -m tools.aic_kv.training_model_fragment_report --root PACKAGE
python -m tools.aic_kv.training_jit_render --root PACKAGE
python -m tools.aic_kv.training_counter_replay --root PACKAGE
```

Counter replay reconstructs SQLite PMU tables from retained replay0 columns
(including native half-nanosecond timestamps), decompresses HBM databases and
rebuilds results using only archived metadata. The final verification uses the
same metric implementation; it is evidence reproducibility, not independent
hardware calibration. Running analysis mutates derived reports; verify original
SHA256SUMS.json before rebuilding, or use a copy.

Experiment entry points and contracts are documented in tools/aic_kv/README.md.
The exact observed training Python tree is retained under
source/training-observation-tree, with per-file hashes; source/training-matrix.patch
is relative to the previous frozen observation tree. Worker sources are frozen
separately for each optimizer suite. External model weights, CANN/MindSpore/
MindFormers binaries and environment-specific native extensions are referenced
by identities, not redistributed. Repeating hardware experiments requires the
recorded pinned stack, model and a valid isolated training checkout; this package
is not a portable runtime environment.

Raw device captures remain at the paths in acceptance.json. The durable archive
is /models/npu_nvme_exp/user7-stack/aic-aiv-training-followup-20261109. DELIVERY_MANIFEST.json
lists every compact file intended for publication, with native SHA256SUMS.json
providing archive hashes. CPU_VALIDATION.json distinguishes CPU checks from
real device observations.
