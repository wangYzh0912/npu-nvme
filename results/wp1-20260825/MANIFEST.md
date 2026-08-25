# Included evidence

The `raw/experiments/output/wp1/current/` subtree contains:

- `checkpoint_matrix_summary.json` and `model_compatibility.json`;
- 83.0.0 same-device buffered FS, O_DIRECT, and Host-SPDK traces;
- E1 raw-SPDK depth-1/depth-4 formal result records;
- GPT-2, GPT-2 XL, Llama2 7B, GLM4 9B, and GPT-2 13B P4 records;
- GPT-2 XL P1/P2/P5 and GPT-2 13B P2 records;
- the external filesystem trace and `strace -f -c` report.

The authoritative numerical interpretation is documented in
`docs/NEAR_TERM_WORK_PLAN.md` section 9.9. Results are kept as captured,
including failure evidence and environment snapshots where applicable; no
samples were removed from the ignored source output directory.
