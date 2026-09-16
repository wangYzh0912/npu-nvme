# H02 isolated fault/reopen acceptance — 2026-09-13

The declared HW1 subset passed three consecutive rounds: nine fault/backpressure cases and two separately classified process-crash smoke cases per round. Software regression passed 35 cases (17 CPU process/ownership checks and 18 production C ASan/UBSan cases). The production C library and public ABI were not changed.

Implementation: `8c82189`. Initial hardware evidence: `41615eb`. Each hardware run preserves the exact executed script, source diff against its run-time HEAD, loaded library hash, actual SPDK source root/commit and archive hashes, CANN library hash, device/configuration, child commands, PIDs, SHM IDs, exit codes, logs and resource snapshots. The run-time HEAD precedes the implementation commit; the recorded source snapshot matches that committed implementation. `run_index.json` maps the results. Review: self-reviewed.

## Failure and correction

The original `results/hw-v13/h02-stage4` failed after the 500 ms injected metadata delay exceeded both the 50 ms I/O deadline and the old cleanup wait. Cleanup retained the controller because drain was not proven; the test nevertheless attempted a second attach in the same process. SPDK rejects an already attached PCI device. The prior “resource exhaustion” description lacked supporting evidence.

The parent now avoids loading ACL/SPDK. It executes one fault worker and, only after a successful safe close and child exit, a separate healthy verify worker. EAL receives a fresh SHM ID at process startup. A close failure retains caller buffers, records diagnostics and stops subsequent device tests. HBM source storage and request handles remain alive until close/quiescence and terminal-request checks succeed. No production default timeout was increased.

The timeout probe retains the original 50 ms I/O deadline and 200 ms observation bound. It also deliberately exercises a 50 ms short close, then retries close with an independent 5 second cleanup budget:

| Round | I/O timeout | Short close | Successful retry | Result |
|---|---|---|---|---|
| r1 | -110, 50.751 ms | -110, 50.615 ms | 405.935 ms | 11/11 pass |
| r2 | -110, 50.740 ms | -110, 50.664 ms | 405.712 ms | 11/11 pass |
| r3 | -110, 50.770 ms | -110, 50.664 ms | 405.683 ms | 11/11 pass |

All non-crash workers returned close=0, had no retained DMA slots or outstanding requests after close, and were followed by successful independent write/flush/readback verification. Backpressure returned exactly -EBUSY and every accepted request succeeded. Observed peaks in each round were DMA=1, NVMe=1, request-ring=16; configured bounds were depth=2 and ring=16. HugePages_Total=1663 and HugePages_Free=1627 before and after each round. These are observed resource checks, not latency-tail estimates or an unbounded leak-proof claim.

## Coverage limits

- Faults: ACL copy, event query/record, NVMe submit/completion, metadata write, flush, metadata timeout; plus request-ring backpressure. Software injection around real devices is identified as such; it does not simulate a physical device failure.
- Quarantine caused by actual stream-sync failure, lost hardware completions, concurrent submit/close stress and driver DMA-stop guarantees are not newly hardware-qualified. C_IMPL quarantine/late-completion checks remain separate.
- The first crash smoke exits after accepting a Host request observed before NVMe submit, with no HBM transfer. The second exits after scratch data flush and before any metadata publish call. Expected exits 86/87 require a written boundary marker; both verify the existing superblock is unchanged in a fresh process. These are not actual checkpoint-commit fault windows, power-loss tests or proof that process exit stops DMA.
- Full H02, CR-03, B and D1 remain unaccepted where their other required cases are missing. TR-01/02/03 retain their original C1 task definitions and planned status.
- The user-authorized 83:00.0 was the only raw target; NPU7 and PA IOVA were used. 84:00.0 was never a raw-test target. No reformat was performed in this repair.

## Regression and failure records

FULL-IO passed after the change; the initial rerun log is `results/hw-v13/full-io-rerun-001.txt`. A final capture under `full-io-final/` records the return code, command, script/binary hashes, stdout/stderr and profiling files. It was repeated because the first launch had not written a structured exit-code artifact. H01 seeds 41/42/43 were not retrained; seed 43's existing final result is now committed. All three restore bytes exactly and continue within the frozen loss/state tolerance, but final states are not byte exact.

The original H02 failure directory remains unchanged and lacks a result.json; its terminal traceback was available in the session rather than an archived stdout/stderr file. The original CANN/MindSpore initialization failure log is copied to `logs/v13-h01-c1.txt`. This repair initially invoked a Python without pytest; `logs/v13-h02-software-001.txt` retains that failure, followed by successful environment-specific runs. Three launch attempts were rejected before device initialization because the output directory had been precreated. Their stderr files were overwritten by subsequent launch commands; the session records those FileExistsError failures, but no reconstructed text is presented as original raw output. The runner's refusal to overwrite old output is also covered by a passing CPU regression.

## Reproduction

Source `/usr/local/Ascend/ascend-toolkit/set_env.sh`, prepend this checkout's `build_out/lib` and `python` to the corresponding paths, and use `/home/user7/miniconda3/envs/ms_2.5/bin/python` as root. Run `tests/hardware/stage4_fault_lifecycle.py --pci 0000:83:00.0 --npu 7 --shm-id <unused-base> --output <new-directory>`. Do not precreate the output directory. Run serially on the authorized namespace. The default worker deadline is 60 seconds; close budget is 5000 ms. On failure the parent exits nonzero and stops further device access.
