# Checkpoint I/O final host timing results

M0 and diagnostic correctness passed. M1 completed on 2026-11-12 at 09:46
Asia/Shanghai, using frozen source commit 266bb35f. The earlier snapshot/
is historical partial evidence; the final result is under final/.

Nine independent training processes and three fresh-process step-60-to-64
restores completed. All nine trajectory comparisons passed on four ranks.
The saved analysis was independently recomputed during review on 2026-11-16.

| Method | Save/aligned-step median (s) | P95 (s) | Formal tokens/s by process |
| --- | ---: | ---: | --- |
| T0 none | 0.602136 | 0.850123 | 208.708, 200.805, 197.372 |
| T1 Host sink | 185.488923 | 192.693373 | 2.50745, 2.51035, 2.53561 |
| T2 blocking FULL | 546.058697 | 555.754207 | 0.858425, 0.864352, 0.866592 |

Each group has three independent processes and 30 aligned save windows. Formal
throughput covers steps 28--64 only. These timings do not establish NVMe service
time, engine overlap, performance improvement, or the 10% probe-overhead gate.
The diagnostic capture timeline omits approximately 24 seconds of pre-transfer
work per save; complete save timing already includes that work.

[Final results](final/m1/result.json), [analysis](final/m1/analysis.json),
[probe screening](final/m1/probe-overhead.json), and
[file hashes](final/manifest.json). Detailed logs remain at the recorded local
evidence root. The next campaign closes the measurement gaps independently.
