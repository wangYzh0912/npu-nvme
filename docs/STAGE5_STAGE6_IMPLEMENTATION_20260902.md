# Stage 5/6 implementation note

The multi-rank FULL lane is now parameterized for two and four ranks.
`tests/hardware/c2_multirank_state.py` accepts `--world-size` and
`--rank-devices`, partitions the declared multi-rank area without overlap, and
publishes one `MULTI_TRAINING_STATE_FULL` record only after every rank has sent
its manifest, payload checksums, and `COMMIT_READY`.

The normal C2 path uses one coordinator and independent rank workers. With
`--hccl`, the coordinator starts one `msrun` job for all rank workers; each
worker joins HCCL, enables MindSpore data parallel mode, and then uses the same
rank-local FULL protocol. The source process emits `checkpoint_gate.json`,
which is consumed by `experiments/benchmarks/run_hccl_full.py`.

Examples:

```bash
conda activate ms_2.5
python tests/hardware/c2_multirank_state.py \
  --world-size 2 --rank-devices 1,2 --run-dir results/stage5/c2-2rank \
  --pci 0000:83:00.0 --coordinator-npu 7 --slot-size-gb 4 --shm-id 19602

python tests/hardware/c2_multirank_state.py \
  --world-size 4 --rank-devices 1,2,3,4 --run-dir results/stage5/c2-4rank \
  --pci 0000:83:00.0 --coordinator-npu 7 --slot-size-gb 4 --shm-id 19604

python experiments/benchmarks/run_hccl_full.py --c2-full \
  --world-size 4 --steps 2 --output results/stage6/hccl-4rank \
  --rank-table config/rank_table_4p.example.json
```

Fresh restore remains serial per rank because the current SPDK shared-memory
instance permits one primary attach. The result records this explicitly as
`restore_transport=standalone-spdk-primary`; it is a correctness gate for
rank-local state and continuation, not a claim of concurrent HCCL restore.

On the validation host used on 2026-09-02, NPU devices 1-4 and
`0000:83:00.0` were present, but the unprivileged run could not open
`/dev/hugepages` (`spdk_env_init`/DPDK permission failure). Formal C2/HCCL
acceptance therefore requires the documented root/hugepage setup before the
commands above can be considered executed.
