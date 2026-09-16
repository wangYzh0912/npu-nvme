# DeltaDetect AscendC 原型审阅与归档说明

## 归档位置

该原型的源码、CANN 生成模板和诊断脚本完整保存在已推送的远端原始快照分支：

```text
codex/pre-cleanup-archive
```

对应基线提交为 `904d9bf99e6ff5a0243490bae96675349204abf3`。原型不合并回整理后的开发主线。

## 审阅结论

该目录记录了 AscendC 自定义算子接入的早期探索，但当前实现不能证明“设备侧 Delta 检测”已经成立，也没有接入当前 `DeltaTrainCell` 或检查点恢复协议。

### 1. 内核没有计算 Delta

`delta_detect_kernel` 只有当前参数、参数偏移表和输出三个数据缓冲，没有上一版本参数或 reference 输入。核心计算仅对当前输入执行平方与求和，因此输出是当前参数的平方和，不是 `current - reference` 的 L2 差异。文件头所述设计目标与实际计算不一致。

### 2. Host、JSON 与 kernel ABI 不一致

Host `OpDef` 声明两个输入和一个输出，并通过 tiling data 下发参数；JSON 使用另一组输入名和额外属性；kernel 入口却直接声明三个标量参数。标准 AscendC tiling 指针、workspace 与入口参数的对应关系没有闭合，无法据此确认运行时会按预期传参。

### 3. 数值与并行策略不足以支撑模型级使用

算子固定 `blockDim=1`，逐参数串行遍历；FP16 输入先原位平方再归约，且结果实际上是 squared norm，却在测试中称为 norm。大参数上的溢出、尾块搬运约束、精度误差和吞吐均没有可靠验证。

### 4. 集成测试不能形成有效证据

测试脚本包含固定用户路径、固定设备号和多套互相冲突的 `ops.Custom` 调用方式。`test_dd_v2.py` 的注册函数在 `param_info` 定义前引用它；多处测试捕获异常后仍继续执行并打印完成信息，也没有以进程返回码表达失败。因此这些输出不能作为功能或性能结论。

### 5. 工程模板占比过高

原型包含完整复制的 CANN CMake、makeself 和安装模板，核心 Host/kernel 代码只占很小部分。继续放在主线会显著增加检索噪声和维护成本。

## 后续若重新启动该方向

1. 先定义唯一算子语义：输入必须显式包含 current/reference，输出明确是 L2 norm、squared norm 或块级 Top-K 指标。
2. 使用目标 CANN 版本的最小官方模板重新生成工程，保持 OpDef、JSON、tiling data 和 kernel ABI 一致。
3. 先做小张量 CPU 对照和异常返回码测试，再测多 block 并行、数值误差和真实模型规模。
4. 只有在输出能够进入统一 Delta frame 并被 `recover()` 消费后，才把该路径重新接入开发主线。
