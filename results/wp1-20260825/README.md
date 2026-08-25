# WP1 formal evidence bundle

This directory contains the small, reviewable evidence subset for the 2026-08-25
WP1 checkpoint-I/O experiments. The source runs were captured on branch
`exp/wp1-io-overhead` at commit `45edc8f`; the runner and documentation changes
were subsequently committed as `69adafb`.

Included files are the formal result/config/environment/sample/timeline records,
the same-device byte-stream traces, the model-matrix summary, and the external
`strace` control sample. Large per-request profiling CSV files remain in the
ignored local directory `experiments/output/wp1/current/` and are intentionally
not copied into the repository evidence bundle.

The 83.0.0 device was restored to `uio_pci_generic` after the destructive
filesystem comparison. The 84.0.0 device remained XFS-mounted at `/models`.
The GLM4 failed native KV-cache attempts are retained in the local output tree;
the successful `use_past=False` run is included here and is marked with its
limitation in the plan document.
