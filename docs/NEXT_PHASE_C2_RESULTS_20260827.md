# C2 两 rank 完整训练态恢复结果（2026-08-27）

## 结论

C2 correctness-first 门禁通过。两个真实 MindSpore/MindFormers 训练进程完成
各自的 model、Adam optimizer 和 control state 快照，经单一 SPDK coordinator
写入 `0000:83:00.0` 后，两个新进程分别恢复自己的 rank shard，并完成下一步
训练；continuation loss 与保存进程一致。

该结果验证的是多 rank manifest/checksum、单 owner 写入、全局 commit 和跨进程
恢复语义；当前 rank 仍为独立 stand-alone 训练进程，没有 HCCL 梯度同步，不能
作为四卡真实分布式训练结论。

## 配置与结果

| 项目 | 值 |
|---|---|
| 设备 | Huawei ES3000 V6 NVMe `0000:83:00.0` |
| NPU | 训练 rank 0/1 使用 NPU 1/2；coordinator 使用 NPU 7 |
| 模型 | MindFormers GPT-2，seq_len=129 |
| 保存步 | step 2；每 rank 597 fields，约 1,485,402,112 bytes |
| 提交 | generation 8，A/B metadata，两个 rank shard |
| 恢复 | fresh process，串行初始化 SPDK，rank 0/1 各继续 1 step |
| continuation loss | 两 rank 均 `10.875582695007324` |
| 结果 | `PASS` |

## 复现与证据

```bash
conda activate ms_2.5
python tests/hardware/c2_multirank_state.py \
  --run-dir results/next-correctness/c2-multirank-20260827-retry \
  --save-step 2 --continue-steps 1 --model gpt2 --seq-len 129 \
  --shm-id 96 --coordinator-npu 7 --pipeline-depth 4 --slot-size-gb 10
```

机器可读结果见
`results/next-correctness/c2-multirank-20260827-retry/result.json`（本地实验
产物未纳入源码提交）。SPDK restore 必须串行运行，因为同一 shared-memory
实例不支持多个独立进程同时作为 primary attach controller。

## 边界与后续

- 尚未覆盖 HCCL 梯度同步、GPT-2 XL/13B、四卡恢复和 rank 故障注入矩阵。
- Unix-socket host 转发只用于正确性，不用于性能结论。
- 后续顺序：C2 故障矩阵 → C3 四卡 GPT-2 XL → C3 四卡 13B → 再进行
  `aclrtMemcpyAsync` 实现与 C1-C3 复验。
