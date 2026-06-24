# P0-U6: 方案可行性深度分析 & 下一步计划

## 一、之前方案为什么不行？

### 方案 A（原始方案）：sink=FALSE + Per-Step CPU Callback

这是在 sink=FALSE 下验证的 baseline，**它本身是能工作的**，但性能不可接受。

```
MindSpore (sink=FALSE):
  for step in 1..N:
    launch_graph(fwd→bwd→opt)     ← 每步独立编译/发射
    CPU callback fires             ← 回调正常触发
    callback 里调 trigger_probe()  ← CPU 驱动写盘
```

**为什么不行**：sink=FALSE 下每步有 ~1500ms 的图编译/发射开销，非 CKPT 步 1925ms。目标是把非 CKPT 步降到 ≤370ms。

---

### 方案 B（首次尝试）：sink=TRUE + CPU Callback 驱动

MindSpore 的 `dataset_sink_mode=True` 会把整个 epoch 的循环融合成一张大图，在 NPU 上自治运行，**中间不回 CPU**。

```
MindSpore (sink=TRUE):
  fused_graph:  ← 一张大图，NPU 自治
    for step in 1..N:
      fwd() → bwd() → opt()
  CPU callbacks: ← 只在图启动前/结束后触发！
    on_train_step_begin → 仅在最后一步触发
    on_train_step_end   → 仅在最后一步触发
```

我们用 `experiments/diag_sink_callback.py` 做了诊断验证，结果确凿：

> `on_train_step_begin` / `on_train_step_end` 在所有中间步上**完全不被触发**，只在 step=N 时触发一次。

这就意味着：**你无法在 step=10, 20, 30 时从 CPU 发指令触发 checkpoint**。CPU 在整个 epoch 中间是"失联"的。

---

### 方案 C（半 Device-Side）：图内 uint32 计数器自增

思路：既然 CPU 失联，就让图自己维护计数器，在 CKPT 步时触发。

```python
# ProbeTrainOneStepCell.construct()
self.step_counter = ms.Parameter(ms.Tensor([0], dtype=ms.uint32))  # ← 问题在这
ops.assign_add(self.step_counter, 1)  # Add 算子, uint32 输入
```

**为什么不行**：

Ascend 的 `Add` 算子底层调 `aclnnAddGetWorkspaceSize`，它**不支持 `DT_UINT32`**。这是 Ascend 算子库的硬限制，不是 MindSpore 的限制。`ops.assign_add` 在 uint32 tensor 上走的就是 `Add` 算子路径。

**我们亲手验证过**：把 dtype 改成 `ms.int32`，`Add` 就能过。但问题在于 `flag` 和 `expected` 已经是 uint32（因为 WaitProbe AICPU kernel 绑定 uint32，且 C 层 `signal_probe_flag()` 用 uint32 语义写 `flag += 1`），计数器如果是 int32，两个世界类型不一致。

**方案 C 的 CPU 子方案更不行**：

> "CPU 端通过 npu_nvme_set_probe_flag_value() 直写 expected Parameter"

这在 sink=TRUE 下等于回到方案 B 的问题——你在 step 中间根本拿不到 CPU 控制权。

---

## 二、为什么当前方案（AICPU Trigger Kernel）可行？

### 核心洞察：图内计数器 + AICPU 内核 = 完全摆脱 CPU

方案 B 的本质是把 "何时触发" 的判断逻辑从 CPU 搬到 **图内的 AICPU 核** 上执行：

```
┌──────────────────────────────────────────────────────────────────┐
│ Fused Graph (sink=TRUE, 整张图一次发射到 NPU)                      │
│                                                                    │
│  for each step (全部在图内, 不回 CPU):                               │
│                                                                    │
│    1. fwd(inputs) → loss                                           │
│    2. bwd(loss) → grads                                            │
│    3. opt(grads) → update weights                                  │
│                                                                    │
│    4. step_counter = assign_add(step_counter, 1)  ← int32, ✅      │
│                                                                    │
│    5. TriggerProbe(step, trigger_ptr, interval)    ← AICPU kernel  │
│       ┌─────────────────────────────────────────┐                  │
│       │ if (step % interval == 0):               │  ← 纯 C 逻辑    │
│       │     *trigger_ptr = step;  // 写 HBM      │  ← device 内存写 │
│       └─────────────────────────────────────────┘                  │
│                                                                    │
│    6. WaitProbe(flag, expected)  ← AICPU 自旋等 C 层放行            │
│       ┌─────────────────────────────────────────┐                  │
│       │ while (*flag < *expected):  // spin      │  ← 已有,已验证   │
│       │     ;                                     │                  │
│       └─────────────────────────────────────────┘                  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
         │                                      │
         │  TriggerProbe 写 device memory        │  WaitProbe 读 device memory
         │  (图内 AICPU → HBM)                   │  (HBM ← C 层 signal)
         ▼                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│ C 层 probe_listener_thread (独立线程, 100us 轮询)                   │
│                                                                    │
│  while(1):                                                         │
│    read_trigger_dev(trigger_ptr) → cur_trigger                      │
│    if cur_trigger > last_seen:                                     │
│      last_seen = cur_trigger                                       │
│      process_write_pipeline(tasks)  ← SPDK D→H→NVMe               │
│      signal_probe_flag()            ← flag += 1 (H→D)             │
│        → 释放 WaitProbe 自旋                                        │
└──────────────────────────────────────────────────────────────────┘
```

### 逐层解构可行性

#### 第 1 层：int32 计数器自增 — 已经可行

Ascend `Add` 支持 int32，`ops.assign_add(self.step_counter, self.one)` 在图内可以正确执行。不需要 uint32。

#### 第 2 层：AICPU kernel 写 device memory — 关键论证

**已确认的事实**：

1. **AICPU kernel 可以获取输入 tensor 的 device 指针**。WaitProbe kernel (`wait_probe_kernels.cc:21`) 已经在做：
   ```cpp
   volatile uint32_t* flag = static_cast<volatile uint32_t*>(flag_tensor->GetData());
   ```
   `GetData()` 返回的就是 HBM 上的物理地址。

2. **AICPU kernel 可以通过 uint64 参数接收任意 device 地址**。在 MindSpore `CustomRegOp` 中声明 `DataType.U64_Default` 类型的输入，将 `trigger_ptr` 以 uint64 scalar tensor 的形式传入内核，内核内部 cast 成指针即可写入。

3. **Ascend 的 HBM 是统一地址空间**。AICPU 核和 ACL runtime 的 `aclrtMemcpy` 操作的是同一个物理 HBM 地址空间。AICPU 核上 `*ptr = value` 写入的地址，C 层 listener 线程通过 `aclrtMemcpy(Device→Host)` 可以正确读到。

4. **WaitProbe kernel 验证了 AICPU 核上的 spin-wait 模式可行**。它已经在生产环境中正确运行——AICPU 核上 `while (*flag < *expected);` 不会导致死锁或超时。

**剩余不确定点及应对**：

| 不确定点 | 风险等级 | 应对 |
|----------|:--------:|------|
| AICPU kernel 内直接 `*(uint32_t*)raw_ptr = val` 是否合法 | 低 | 用 `aclrtMemcpy` 替代，或通过 output tensor 间接写 |
| `ops.assign_add` 在 sink=TRUE 下的执行语义 | 低 | 已验证 int32 Add 在图内正确，assign_add 是 assign+add 的组合 |
| TriggerProbe + WaitProbe 两个 AICPU kernel 在同一图中是否冲突 | 极低 | 两个独立 kernel，各自占用自己的 AICPU 核，不存在资源冲突 |

#### 第 3 层：C 层 listener 的 device-trigger 模式 — 已实现且自测通过

`probe_listener_thread()` (src/npu_nvme.c:462-558) 已经实现了完整的双模逻辑：

- **Device-trigger 模式**（`ctx->dev_trigger_ptr != NULL`）：100us 间隔通过 `aclrtMemcpy(D→H)` 轮询 trigger buffer，检测到新值则触发写盘
- **Host-trigger 模式**（fallback）：`probe_flags[0]` 自旋等待

触发后的 pipeline 完全复用已验证的 `process_write_pipeline()` + `signal_probe_flag()`。

`npu_nvme_set_trigger_ptr()` 和 `npu_nvme_read_trigger_dev()` 的 ctypes 绑定在 `direct_checkpoint.py:71-76,526-547` 已经就绪。

#### 第 4 层：与 WaitProbe 的协作 — 已验证

WaitProbe kernel 的语义是"自旋到 flag >= expected"。C 层的 `signal_probe_flag()` 做 `flag += 1`。触发序列是：

```
step=10: TriggerProbe 写 dev_trigger=10
         C listener 检测到 → process_write_pipeline → signal_probe_flag (flag:0→1)
         WaitProbe 自旋释放 (expected=1, flag=1) ✅

step=20: TriggerProbe 写 dev_trigger=20
         C listener 检测到 → process_write_pipeline → signal_probe_flag (flag:1→2)
         WaitProbe 自旋释放 (expected=2, flag=2) ✅
```

**注意**：expected 也需要在图内递增。TriggerProbe 触发后，下一步需要 `expected = assign_add(expected, 1)`，否则 WaitProbe 不等就直接过去了。但这又回到 uint32 Add 的问题...

**这实际上是当前方案需要补充的一个细节**：expected 本身也是 uint32 Parameter，它也需要在图内递增。但如果 TriggerProbe kernel 在内部做了 modulo 判断并写入 trigger，那么是否可以直接在 TriggerProbe 内部同时递增 expected？或者采用另一个方法：让 expected 和 trigger 协同——expected 可以是 int32（与 step_counter 同一类型），而 WaitProbe 内部的比较改为 int32 vs uint32 的宽松比较。

**更简单的方案**：不用 expected 自增。改为每次 CKPT 步后，TriggerProbe 直接将 dev_trigger 写入，C 层完成后 `signal_probe_flag`（flag++），而 WaitProbe 的 expected 始终是 `flag 的旧值 + 1`。但 expected 不进图，WaitProbe 就会在第 1 次触发后永久放行...

**最简可行方案**：让 `expected` 也是 int32（和 step_counter 一样），与 `flag`（uint32）分离。WaitProbe 内部做类型转换比较。但这需要改 WaitProbe kernel。

**或者更直接的方案**：在 TriggerProbe AICPU kernel 内部，不仅写 `dev_trigger`，还直接在当前图内更新 expected。因为 AICPU kernel 可以拿到 expected 的 device 指针，直接 `*(volatile uint32_t*)expected_ptr += 1`。

---

## 三、下一步开发计划

### Step 1: 创建 TriggerProbe AICPU Kernel 源码

新建文件 `kernels/trigger_probe/`，参照 `wait_probe/WaitProbeProject/` 的目录结构：

```
kernels/trigger_probe/
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh
├── cpukernel/
│   ├── CMakeLists.txt
│   ├── toolchain.cmake
│   ├── impl/
│   │   ├── trigger_probe_kernels.h
│   │   └── trigger_probe_kernels.cc   ← 核心实现
│   └── op_info_cfg/
│       └── aicpu_kernel/
│           └── trigger_probe.ini
├── op_proto/
│   ├── trigger_probe.h
│   └── trigger_probe.cc
└── framework/
    └── (同 WaitProbe)
```

核心 kernel 逻辑 (`trigger_probe_kernels.cc`)：

```cpp
uint32_t TriggerProbeCpuKernel::Compute(CpuKernelContext &ctx) {
    // Input 0: step (int32)
    Tensor *step_tensor = ctx.Input(0);
    int32_t step = *(int32_t*)step_tensor->GetData();

    // Input 1: interval (int32)
    Tensor *interval_tensor = ctx.Input(1);
    int32_t interval = *(int32_t*)interval_tensor->GetData();

    // Input 2: trigger_ptr (uint64) — device address
    Tensor *ptr_tensor = ctx.Input(2);
    uint64_t trigger_addr = *(uint64_t*)ptr_tensor->GetData();

    // Core logic: modulo check + device memory write
    if (step % interval == 0) {
        volatile uint32_t *trigger = (volatile uint32_t*)trigger_addr;
        *trigger = (uint32_t)step;  // write step number to trigger buffer
    }

    // Output 0: pass-through step
    Tensor *out = ctx.Output(0);
    *(int32_t*)out->GetData() = step;

    return 0;
}
```

### Step 2: 编译 TriggerProbe kernel

```bash
cd kernels/trigger_probe/ && bash build.sh
# 产出: build_out/lib/libtrigger_probe_kernel.so
```

### Step 3: 在 ProbeTrainOneStepCell 中集成

```python
class ProbeTrainOneStepCell(nn.Cell):
    def __init__(self, ...):
        # 新增: int32 step counter
        self.step_counter = ms.Parameter(
            ms.Tensor([0], dtype=ms.int32), requires_grad=False, name="step_counter")
        self.one_i32 = Tensor([1], dtype=ms.int32)
        self.interval = Tensor([CKPT_INTERVAL], dtype=ms.int32)

        # 新增: TriggerProbe AICPU kernel 注册
        trigger_op_info = CustomRegOp("TriggerProbe") \
            .input(0, "step") \
            .input(1, "interval") \
            .input(2, "trigger_ptr") \
            .output(0, "y") \
            .dtype_format(DataType.I32_Default, DataType.I32_Default,
                          DataType.U64_Default, DataType.I32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.trigger_probe = ops.Custom("TriggerProbe", ...)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)

        # 图内: step_counter += 1  (int32, ❌→✅)
        step = ops.assign_add(self.step_counter, self.one_i32)

        # 图内: TriggerProbe — 若 step 是 CKPT 步则写 dev_trigger
        _ = self.trigger_probe(step, self.interval, self.trigger_addr)

        # 图内: WaitProbe — 等待 C 层完成写盘
        wait_sig = self.wait_probe(self.flag, self.expected)

        safe_grads = self.hyper_map(ops.partial(bind_depend_op, wait_sig), grads)
        opt_res = self.optimizer(safe_grads)
        loss = self.depend(loss, opt_res)
        return loss
```

### Step 4: 修复 fire_and_forget.py

解决 `self.train_cell.step_counter` 不存在的问题：
- `set_trigger_ptr()` 需要传入 trigger buffer 地址（一个 `aclrtMalloc` 分配的 uint32 device buffer，不是 step_counter 本身）
- 需要在 `DirectCheckpoint` 中新增 `alloc_trigger_buffer()` 方法
- 将 trigger buffer 地址以 uint64 scalar tensor 的形式传入图内

### Step 5: 端到端测试

```python
# fire_and_forget.py 修复后:
TOTAL_STEPS = 30
CKPT_INTERVAL = 10

ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb],
               dataset_sink_mode=True)
```

验证标准：
- `final_flag >= 3`（3 个 CKPT 全部完成）
- 非 CKPT 步平均耗时 ≤ 370ms
- C 层 profiling 日志正常输出 timing 数据

### Step 6: 性能对比

| 指标 | sink=FALSE (baseline) | sink=TRUE (目标) |
|------|:---:|:---:|
| 非 CKPT 步 | 1925ms | ≤ 370ms |
| CKPT 步 | 2076ms | ≤ 420ms |
| flag_wait | 0.47ms | ≤ 1ms |
| CKPT 带宽 | 4380 MB/s | 4380 MB/s (不变) |

### 关于 expected 自增问题的解决方案

在上述 Step 3 中还隐藏一个问题：`self.expected`（uint32）也需要在图内自增，否则 WaitProbe 在第一次通过后就不再阻塞。有几个选项：

**选项 α**（推荐）：在 TriggerProbe AICPU kernel 内部同时处理 expected 自增。
- 增加 Input 3: expected_ptr (uint64)
- kernel 内部：`if (step % interval == 0) { *trigger = step; *(uint32_t*)expected_ptr += 1; }`
- 这样 WaitProbe 的 expected 同步递增，不需要图内 uint32 Add

**选项 β**：让 expected 也是 int32，WaitProbe 内部做类型转换。
- 但这需要修改已有的 WaitProbe kernel（已在用，有回归风险）

**选项 γ**：不用 expected 机制，TriggerProbe 直接写 `flag` 而不是 `dev_trigger`。
- 但这会让 WaitProbe 变成"触发即通过"，没有真正等 C 层写盘完成
- 失去了异步写盘 + 图内等待的安全保证

**推荐选项 α**，实现简单，不破坏现有 WaitProbe 语义。

---

## 四、风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| AICPU kernel 内 raw pointer 写入 crash | 低 | 高 | 降级为 output tensor + assign |
| TriggerProbe 注册/加载失败 | 低 | 中 | 参考 WaitProbe 已验证流程 |
| sink=TRUE 下图内 assign_add 语义异常 | 极低 | 高 | 已确认 int32 Add 在 Ascend 可行 |
| trigger buffer 地址在 sink=TRUE 下失效 | 低 | 高 | 用 aclrtMalloc 分配固定 buffer |
| C 层 poll 延迟导致 CKPT 步过长 | 低 | 低 | 100us poll 够快，可在 kernel 内加微等待 |
