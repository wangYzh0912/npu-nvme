# Native TP4 restart and continuation

**EN-native is accepted for the fixed-token deterministic TP4 scope using source-002 / restore-003.** The third restore additionally reads back and compares every captured control/RNG field before publishing rank readiness. It again matches three-step loss and final state exactly; see oracle-003.json and ../EN-native/gate-001. The post-run evidence/fault profile has 39 passing cases; it does not run or simulate new training.

**Earlier source-002 / restore-002 also passed the positive restart experiment.** Both launchers exited 0; source ranks exited before four new restore processes began. Each rank restored all 952 observed training-network parameters exactly before continuing. Steps 9–11 losses were bit-identical to the source oracle. Final parameter manifests and complete captured control sidecars (including RNG/cursor/LR horizon) were identical on all four ranks. See oracle-002.json and restore-002/acceptance.json.

The workload is Qwen3-8B, TP4/DP1/PP1, BF16 compute with FP32 model/Adam, batch 1, sequence 128, fixed token rows, zero dropout, seed 42, checkpoint step 8, stop step 11 and LR horizon 32. Determinism is explicitly ON. Loss/state tolerances were frozen at atol=1e-6, rtol=1e-5 before the run and were not widened. All observed comparisons passed exact equality. This is one correctness pair, not a performance sample or reshard/stochastic-training claim; no raw NVMe I/O occurred.

Failed source/restore-001 evidence is retained: restored parameter bytes passed but nondeterministic continuation loss did not. Its original callback-step bookkeeping error and absent final-state capture are recorded in restore-001/interpretation.json. It is not overwritten by the successful pair.

Historical step8 files were fully hashed separately. Their missing explicit RNG/data cursor/LR configuration remains a partial-state boundary. state_schema.json preserves the historical observed-byte inventory. state_schema-002.json adds actual framework-strategy-derived TP partitions for the new step8 checkpoint: 291 model tensors, 291 m, 291 v, and 9 per-rank controls. For each tensor, global_shape = local_shape times device_matrix[-1-tensor_map_axis] (unmapped dimensions have factor 1); rank coordinates determine slice start/end. All 873 strategy entries were equal across the four ranks; replicated tensor hashes were checked equal. The raw strategy protobufs, decoded maps and decoding script are retained in source-002. capacity_plan.json rejects the old 10 GiB/rank slot assumption and does not invent HBM/Host peaks.

Source snapshots under each run identify the actually executed scripts. The latest code additionally rejects mismatched optimizer, model config, seed, deterministic setting, topology, LR and sink settings before build_context. That guard was tested against both actual saved configurations and negative CPU fixtures after the hardware pair; it was not part of the recorded hardware invocation. The worker subprocess fault traces cover missing/duplicate/wrong rank, wrong step/topology and damaged shard before communication, with nonzero exits and no ready markers. Ready timeout and config mismatch have software coverage.

The earlier restore-002 did not separately capture pre-continuation RNG readback; restore-003 closes that gap and executes the strengthened configuration guard. A generalized TP protocol and old/candidate hardware regression remain outside this EN acceptance and are still open. No claim is made that B, D1, Ours TP4, live capture or async H2D has passed.

Reproduce in a new output directory:

```bash
QWEN_RUN_OUTPUT=/models/npu_nvme_exp/user7-stack/NEW-source \
  bash scripts/run_qwen3_four_rank.sh --checkpoint-step 8 --source-stop-step 11 --lr-horizon 32 --deterministic
PYTHONPATH=.:python /home/user7/miniconda3/envs/ms_2.5/bin/python \
  tools/prepare_qwen_restart.py --source-run /models/npu_nvme_exp/user7-stack/NEW-source
QWEN_RUN_OUTPUT=/models/npu_nvme_exp/user7-stack/NEW-restore \
  bash scripts/run_qwen3_four_rank.sh --checkpoint-step 8 --source-stop-step 11 --lr-horizon 32 --deterministic \
  --resume-run /models/npu_nvme_exp/user7-stack/NEW-source
```

Each source saves about 183 GiB for step8 + final11; restore saves another final generation. Original run/model paths remain on the experiment machine; checkpoint payloads are not in Git. Ready directories must never be reused.
