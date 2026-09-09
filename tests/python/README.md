# Python 验证

目标 Linux/MindSpore 环境运行完整测试：

```bash
python -m pytest tests/python -q
```

普通开发机可运行协议、时间线、CLI dry-run 子集（需 numpy、pytest）：

```bash
python -m pytest tests/python -q \
  --ignore=tests/python/test_baseline_repro.py \
  --ignore=tests/python/test_checkpoint_admission.py \
  --ignore=tests/python/test_live_async_capability.py \
  --ignore=tests/python/test_r0_pipeline.py \
  --ignore=tests/python/test_s2_delta.py
```

以上五个模块仍保留，导入链依赖 MindSpore/ACL 或 Linux resource，需在目标环境执行，
不是删除或视为通过。完整硬件正确性另见 `tests/hardware/`，协议测试不替代持久化恢复。
