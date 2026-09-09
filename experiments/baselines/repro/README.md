# 统一检查点 baseline

共享 MindSpore GPT-2 训练 runner、完整状态 schema、fixture、独立进程恢复及续训 oracle。
支持 `none`、`mindspore_sync`、`mindspore_native_save`、`ours`、`datastates_acl`、
`pccheck_acl`、`bytecheckpoint_host`、`fastpersist_host`。

Native 使用真正的 `ms.save_checkpoint/load_checkpoint`；sync 是 raw-byte 参考。
DataStates/PCcheck 是 ACL/pinned-Host 语义适配，ByteCheckpoint/FastPersist 是隔离 CPU
worker 适配，不能标成未修改的 CUDA/GDS 原版。缺少依赖记录 blocker，不静默替换后端。

## 准备与执行

按 `upstream.lock.json` 的 URL/commit 准备外部源码，按 `worker_environment.lock.json`
准备 worker 环境和扩展。锁文件中的历史机器路径需要替换；它们不表示依赖已随仓库安装。
从 `configs/gpt2_30step.json` 复制本机配置，调整 project_root、worker、upstream、文件
测试目录、NUMA/NPU、裸盘和输出目录。`project_commit` 是锁定的复现基线标识，实际
运行 HEAD/dirty 由环境记录另外采集。配置中的历史裸盘授权不应当作新机器的授权。

```bash
python -m experiments.baselines.repro.cli preflight --config /path/to/local.json --all
python -m experiments.baselines.repro.cli prepare --config /path/to/local.json
python -m experiments.baselines.repro.cli run --config /path/to/local.json \
  --adapter mindspore_native_save
```

CLI 子命令和恢复计时选项用 `python -m experiments.baselines.repro.cli --help` 查看。
同模型各方法复用同一 fixture、batches、step 和 schema；正式 run 默认 30 steps/10 代，
`--steps`、`--checkpoint-every` 是明确的 smoke 覆盖。不能只验证权重或派发完成。

## 当前证据

最新三方法恢复证据见 [净恢复结果](../../../results/full-state-recovery-net-20260907/README.md)。
Ours 在 root/PA IOVA 下的 attach、FULL 保存、新进程恢复与续训已通过；早期 attach
失败不是最终状态。文件系统与 Raw SPDK 使用不同物理 SSD，且 source step 不同，结果
是当前配置参考，不能作为严格跨软件加速比。全状态哈希在独立验证进程执行，不计入
state-ready；每方法五个计时样本不支持 P99 声明。

论文方案的最新 30-step 保存/恢复运行保留在
`results/baseline-gpt2-repro-20260906/`，过往调试运行已删除。上游来源及平台替换保持可追溯。
