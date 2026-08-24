# WaitProbe / TriggerProbe 旧工程审阅与归档说明

## 归档位置

旧工程的完整文件、构建模板和历史实验保存在远端分支：

```text
codex/archive-wait-trigger-probe
```

归档基线提交为 `904d9bf99e6ff5a0243490bae96675349204abf3`。该分支仅用于历史追溯和实验复现，不合并回整理后的主分支。

## 原始设计

旧方案使用两个 MindSpore 自定义 AICPU 算子建立训练图与后台检查点之间的计数握手：

1. `TriggerProbe` 在检查点步写入设备触发缓冲，并递增期望完成计数；
2. C 侧监听线程读取触发值，完成 NPU 到 NVMe 的持久化；
3. `WaitProbe` 在 AICPU 核上轮询完成标志，直到实际完成计数追上期望计数；
4. MindSpore 图通过依赖关系把该等待点放入训练执行顺序。

## 审阅结论

该方案完成了早期设备侧握手探索，但不适合作为当前主线继续维护，原因如下。

### 1. WaitProbe 可能无限占用 AICPU 核

`WaitProbeCpuKernel::Compute` 使用无超时、无取消条件的自旋循环。如果 C 侧写入失败、监听线程退出、计数器丢失更新或设备状态异常，算子会永久停留在循环中。`volatile` 只能约束编译器访问，不能建立完整的跨执行单元同步协议。

### 2. TriggerProbe 缺少输入与并发保护

算子直接执行 `step % interval`，没有验证 `interval > 0`；触发缓冲和期望计数通过普通 volatile 读写更新，没有版本、原子操作或溢出处理。

### 3. GE / aclnn 适配没有形成可靠执行路径

为满足 GE 动态符号查找而增加的 aclnn wrapper 返回空 executor，函数本身不启动实际计算。历史记录和当前兼容层均说明，sink 模式下该路径无法稳定加载，因此后来被 step-counter poller 替代。

### 4. 安装方式修改 MindSpore 环境内部文件

`merge_ms_config.py` 使用固定绝对路径直接改写 MindSpore site-packages 内的 AICPU 配置。这种方式依赖特定用户、Python 环境和 MindSpore 目录结构，也缺少卸载与冲突恢复流程。

### 5. 工程包含大量重复的生成模板

WaitProbe、TriggerProbe 和 DeltaDetect 工程各自复制了 Ascend 自定义算子 CMake、makeself 和安装脚本。多数文件逐字节相同，增加了仓库体积、审阅噪声和版本漂移风险。

## 当前替代方案

当前主线使用 `ProbeTrainOneStepCell` 在图内递增 `step_counter`，由 C 层 Reactor 的周期 poller 读取该计数并触发预注册 I/O；持久化完成后通过 `probe_flag` 通知上层。该路径不再依赖 WaitProbe/TriggerProbe 自定义算子或修改 MindSpore 安装目录。

## 从主线移除的范围

- `wait_probe/`
- `kernels/trigger_probe/`
- `scripts/merge_ms_config.py`
- `python/_legacy_compat.py`
- 仅用于 WaitProbe/TriggerProbe 的编译与微基准脚本

旧结果图中如果继续使用 WaitProbe 名称，必须标明其属于历史方案，不能用于证明当前 step-counter Reactor 路径的性能。
