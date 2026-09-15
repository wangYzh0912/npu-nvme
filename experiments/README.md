# 当前实验入口

训练和方法比较只使用根目录 train.py。repro/entry 是内部实现，adapters 各保留一种方法。公共训练和 D1 oracle 位于 training/full_fixture.py。

Qwen 原生 TP4 训练/恢复、模型兼容和环境盘点继续服务 E0/EN。旧 Delta/FaF/IPC/campaign/PMU 实验在 archive-exploratory-implementations-20260914 标签，不在活动分支保留执行入口。raw 传输验证使用 tests/hardware/transport_matrix.py，故障与生命周期测试分别使用当前专用 harness。
