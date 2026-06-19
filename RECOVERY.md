# 服务器重启后恢复开发

## 环境准备
```bash
cd /home/user7/npu-nvme
git checkout delta
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH
PYTHON=/root/miniconda3/envs/ms_2.5/bin/python
```

## 开发状态
- **分支**: `delta` (从 Asyn 创建, 2026-06-19)
- **最后提交**: `655d26e` Documentation update
- **当前进度**: Step 1 ✅, Step 2 ✅, Step 2b ✅, Step 3 🔲
- **详细状态**: 见 `to_do/STATUS_SUMMARY.md`

## 启动命令参考
见 `to_do/CLAUDE_INSTRUCTIONS.md` 和 `to_do/STATUS_SUMMARY.md`

## 关键文件
- 源码: `python/direct_checkpoint.py`, `python/i3_delta_writer.py`, `src/npu_nvme.c`
- 实现计划: `to_do/IMPLEMENTATION_PLAN.md`
- 会话恢复: `to_do/CLAUDE_INSTRUCTIONS.md`
- Step 1: `experiments/baselines/benchmark/`
- Step 2: `experiments/baselines/step2_demo/`
- Step 2b: `experiments/baselines/step2b_recovery_validation/`
