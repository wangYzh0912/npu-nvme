# Python 测试说明

当前测试仅覆盖不依赖 MindSpore、Ascend 或 SPDK 的协议逻辑，可在普通开发机运行：

```bash
python -m unittest discover -s tests/python -v
```

硬件相关路径仍需在目标 Ascend 服务器上执行 `scripts/verify_phaseA.sh` 和 C 回环测试。
