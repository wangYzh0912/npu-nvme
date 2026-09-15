# NPU–NVMe 严格 FULL 检查点

当前开发基于长期规划 v1.3 的 D1 严格提交与恢复链。旧探索实现已移出活动源码，原生 ABI 升级到 2；本次清理的验收状态见 `results/legacy-cleanup`。清单与限制见 [清理规划](docs/LEGACY_IMPLEMENTATION_CLEANUP.md)，长期目标见 [开发规划](docs/LONG_TERM_DEVELOPMENT_PLAN.md)。

## 当前入口

```bash
export PYTHONPATH="$PWD/python:$PWD"
python train.py preflight --config experiments/baselines/repro/configs/gpt2_30step.json --all
python train.py benchmark --config experiments/baselines/repro/configs/gpt2_30step.json --all
python train.py fit --config experiments/baselines/repro/configs/gpt2_30step.json --adapter ours
python train.py inspect --config experiments/baselines/repro/configs/gpt2_30step.json
```

每轮使用新的 results_root；框架、CANN、SPDK 与权限环境见执行环境文档。benchmark 会先生成共同 fixture，再为各方法启动独立源和恢复进程。独立恢复也可通过 verify-restart 显式执行。

支持：none、MindSpore 原生保存、Ours、DataStates/PCcheck ACL 语义移植、ByteCheckpoint/FastPersist Host upstream worker。语义移植与原生 upstream 运行分别标记。83:00.0 是获授权裸盘，84:00.0 是 /models 文件系统盘，两者不是同一物理设备。

## 库接口

```python
from npu_nvme import StrictCheckpoint
# 打开设备前，训练调用方提供完整组件和显式 expected_spec。
# save_state(..., expected_spec=spec) 返回有所有权的 handle。
# restore_full_state(factory, spec) 仅在校验与 ready 完成后返回 target。
```

冻结 FULL 当前保留两代、三个物理槽、一个待提交请求，chunk 不超过 1 MiB。观察超时不取消 I/O，不释放未证明安全的缓冲区。B2/C2 异步收敛、D2 格式、live/Delta/多 rank 能力仍分别等待后续验收。历史代码标签为 archive-exploratory-implementations-20260914。

## 验证

```bash
python tools/check_repository.py
python tools/run_gate.py --profile CLEANUP --out results/cleanup-software-new
```

软件通过不代替硬件通过；正式完成需清理综合验收。
