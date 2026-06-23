## 二、I2: 基于设备内存轮询的异步同步机制

### 2.1 Motivation 论证链

### 2.1.1 问题背景：sink_size 与训练吞吐的权衡

MindSpore `GRAPH_MODE + sink=TRUE` 下，`sink_size` 控制连续多少个训练步被合并到一个 GE 图执行周期中。只有 epoch 边界（每 sink_size 步）才触发 `on_train_epoch_end` callback：

```
sink_size = 1:
  [Step 1] → callback → [Step 2] → callback → [Step 3] → callback → ...
  每步后 Host 可见，但图重编译/启动开销最大

sink_size = 10:
  [Step 1..10] → callback → [Step 11..20] → callback → ...
  训练吞吐最优，但 Host 每 10 步才能观测一次
```

**核心权衡**：

| sink_size | 训练吞吐 | Host 观测粒度 | 适合全量检查点 | 适合增量检查点 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | ⚠️ 较低（频繁图重编译） | 每步 | ✅ | ✅ |
| N (大) | ✅ 最优（减少图启动开销） | 每 N 步 | ✅（epoch 间同步写） | ❌（需要 per-step 感知） |
| FALSE | ❌ 极低（Python 回调开销） | 每步 | — | — |

**增量检查点面临的困境**：要获得最优训练吞吐，应该选较大的 sink_size；但要实现 per-step delta 写入，需要在每个训练步完成后触发。这两者直接矛盾。如果为了增量检查点而强制 `sink_size=1`，则牺牲了训练速度——而训练速度本应是系统优化的首要目标。

**本文的核心主张**：不应该为了检查点而妥协训练配置。我们的方案通过设备内存轮询解耦了"训练步的批量执行"和"检查点的触发粒度"，使得系统可以在任意 sink_size 下实现 per-step 增量检查点。

### 2.1.2 现有同步机制的局限性

在 NPU 生态中，可用于步边界同步的手段极为有限：

| 机制 | 可用性 | 限制 |
|------|:---:|------|
| Host callback (`on_train_epoch_end`) | ✅ | 仅在 epoch 边界触发；同步执行会阻塞训练 |
| ACL runtime callback (`aclrtCallback`) | ❌ | Ascend 不支持用户态设备回调 |
| CUDA-like Stream Callback | ❌ | Ascend 无等价机制 |
| AICPU custom kernel 注入 | ❌ | GE 以 `RTLD_LOCAL` dlopen 加载，符号隔离 |
| NPU→Host 中断 | ❌ | 不暴露给用户态 |

**结论**：在 NPU + MindSpore 生态中，**不存在标准的、低开销的 per-step 设备→Host 通知机制**。更关键的是，**Host 对 Device 的观测粒度被 sink_size 锁定**——要获得细粒度观测就必须牺牲吞吐。

### 2.1.3 本文方案：设备内存轮询异步同步

核心思想：不在图内插入阻塞点，不依赖框架 callback 的触发粒度。而是向图内**注入一个极轻量的信号**（4 字节 step_counter 自增），由 **Host 侧独立线程通过 HBM→Host DMA 轮询**检测步变化。这实现了两个关键的"解耦"：

**解耦 1：训练步执行 vs 检查点触发**
```
sink_size = 10（训练最优配置）:
  ┌─ GE 图连续执行 ─────────────────────────────────────────────┐
  │ Step 1 → Step 2 → ... → Step 10 → callback                 │
  │   ↑         ↑                        ↑                      │
  │   │    listener 检测           listener 检测                 │
  │   │    → delta write(N=1)      → delta write(N=2) ...       │
  │ step_counter 自增（每一步，4B HBM，1 个 AssignAdd）            │
  └──────────────────────────────────────────────────────────────┘
  
Host callback 只在 step 10 触发（用于 FULL ckpt），
但 listener 在每一步都能检测到步变化并触发 delta write。
```

**解耦 2：检查点触发 vs 训练阻塞**
```
  ┌─ GE 图 (Device) ─────────────┐    ┌─ Background Thread (Host) ──┐
  │                               │    │                              │
  │  forward → backward → opt    │    │  while (running):            │
  │                               │    │    aclrtMemcpy(             │
  │  step_counter += 1    ◄─────────────┤      step_counter,  ←──────┤
  │  (4 bytes, int32, HBM)       │    │      HBM→Host, 4B)          │
  │                               │    │    if step changed:         │
  └───────────────────────────────┘    │      trigger SPDK write     │
                                       │    usleep(10ms)             │
                                       └──────────────────────────────┘
训练线程不等待，I/O 与下一步计算并行。
```

**三个关键性质**：
1. **训练速度优先**：sink_size 的选择仅取决于训练吞吐最优，不受检查点需求约束
2. **无侵入**：图内仅增加 1 个 `AssignAdd` 算子（~132,000 个 kernel instances 中的 0.0007%）
3. **无阻塞**：轮询在独立线程，delta write 与下一步训练并行

### 2.1.4 Motivation 论证结构（论文 §3.3.1~§3.3.2 修订）

```
§3.3.1 问题分析
  ① sink_size 与训练吞吐的量化关系（E0 实验：sink_size sweep 曲线）
  ② 增量检查点的 per-step 需求与 sink_size 的矛盾
     — 为检查点选 sink_size=1 → 牺牲训练速度
     — 为训练选最优 sink_size → 失去 per-step 观测
  ③ NPU 生态中缺乏 per-step 设备→Host 通知原语
  ④ 本文目标：解耦训练配置与检查点触发粒度

§3.3.2 设备内存轮询机制设计
  ① step_counter 注入（图内 AssignAdd，~0.0007% 开销）
  ② 后台轮询线程（ACL context 绑定、10ms 轮询周期选择依据）
  ③ 双重解耦：训练执行 vs 检查点触发、检查点触发 vs 训练阻塞
  ④ 在最优 sink_size 下的 per-step delta write 验证（E2.5 实验）
```

---

### 2.2 I2 Evaluation 实验设计

#### E0: sink_size 与训练吞吐关系曲线（前置实验）

**问题**：sink_size 如何影响 GPT-2 XL 在 GRAPH_MODE 下的训练吞吐？最优 sink_size 是多少？

**为什么这是 I2 动机的基础**：
- 如果没有这个实验，I2 的 motivation 只是定性的（"sink_size=1 有开销"）
- 有了这条曲线，I2 的 motivation 变成定量的："在最优 sink_size=N 下，训练吞吐为 X steps/s；如果为了检查点而退回到 sink_size=1，吞吐下降 Y%。FaF 让你在保持 X steps/s 的同时还能 per-step 写 delta。"

**方法**：
```
统一配置: GRAPH_MODE, GPT-2 XL, device_id=1, batch_size=1, seq_len=1024
扫描 sink_size: 1, 2, 5, 10, 20, 50, 100 (或到单 epoch 内存上限)
每种配置运行 50 步（预热 5 步后开始计时）

测量:
  - 总 wall-clock 时间
  - 平均步时 = 总时间 / 有效步数
  - 吞吐 = steps / second
  - 步时稳定性 (std/mean)

注意:
  - sink_size 改变 epoch 的定义，callback 频率不同
  - 50 步指"有效训练步"（model.train 的 step 参数），不是 epoch 数
  - 内存限制：sink_size 越大，图内缓存的中间结果越多，需监控 HBM 占用
```

**预期**：
```
sink_size=1:    步时最高（每步后图重新启动）
sink_size=2-5:  步时快速下降
sink_size=10-20: 步时趋于平稳（图启动开销被充分摊销）
sink_size=50+:   步时基本不变（已达硬件上限）
```

**论文数据需求**：
- 曲线图：sink_size（x 轴，log scale）vs 平均步时（y 轴）
- 标注最优 sink_size（步时不再显著下降的点）
- 标注 sink_size=1 与最优点的吞吐差距（百分比）
- 表格：每个 sink_size 的 mean/std/n

**后续实验约定**：E0 确定最优 sink_size 后，所有后续实验（E2.1~E2.7, E3.x）统一使用该 sink_size。FaF 的 per-step delta write 在此 sink_size 下验证。

---

#### I2 实验矩阵总览

| 编号 | 实验 | 验证什么 | 状态 |
|:---:|------|------|:---:|
| **E2.1** | 轮询开销基准 | 后台线程不对训练造成显著干扰 | 🔲 全部重新采集 |
| **E2.2** | 检测延迟分布 | 轮询间隔与实际检测延迟的关系 | 🔲 |
| **E2.3** | 写盘完成时序 | Delta write 是否与下一步 optimizer 有时间重叠 | 🔲 |
| **E2.4** | 同步 vs 异步对比 | FaF 相比 callback 同步写盘的步时收益 | 🔲 |
| **E2.5** | sink_size 可扩展性 | FaF 在更大 sink_size 下是否仍然有效 | 🔲 |
| **E2.6** | 多步一致性 | 50+ 步无遗漏触发 | 🔲 |
| **E2.7** | 线程资源开销 | CPU 利用率、内存占用 | 🔲 |

> **数据采集原则**：所有实验数据一律重新采集，不依赖历史 benchmark 数据。统一使用 `sink=TRUE, GRAPH_MODE, sink_size=1` 作为标准配置。

---

### E2.1: 轮询开销基准

**问题**：后台轮询线程（每 10ms 一个 `aclrtMemcpy` HBM→Host 读 4 字节）是否对训练性能造成可测量的干扰？

**方法**：
```
条件 A (B1: 纯 MS): 无任何 probe，纯训练
条件 B (B2: 探针闲置): 后台线程启动并轮询 step_counter，但不触发 SPDK 写
条件 C (B3: 完整 FaF): 后台线程启动 + per-N-step SPDK FULL write
```

**历史数据备注**（不可用，需重采）：
- 历史 B1/B2/B3 = 412/436/435ms，但采集条件不明确（可能为 PYNATIVE_MODE）
- Step 1a 标准基线：GPT-2 XL GRAPH_MODE sink=1 → **468.3ms** (±25.8, COV 5.5%)

**方法**：
```
统一配置: sink=TRUE, GRAPH_MODE, sink_size=1, GPT-2 XL, device_id=1
条件 A (B1: 纯 MS):   无任何探针，纯训练（Step 1a 标准基线复用）
条件 B (B2: 探针闲置): 后台线程启动 + 轮询 step_counter，但不触发 SPDK 写
                        (设置 NPU_NVME_LISTENER_MODE=idle)
条件 C (B3: 完整 FaF): 后台线程启动 + per-10-step SPDK FULL write
                        (NPU_NVME_LISTENER_MODE=full, ckpt_interval=10)

每种条件 n≥30 步，预热 5 步后开始计时
C 层统一使用 get_time_us() 记录，Python 层使用 time.perf_counter() 记录
```

**关注点**：
- B2 vs B1 的差异：纯粹的后台线程开销（线程调度 + 4B DMA 读每 10ms）
- B3 vs B2 的差异：SPDK 写盘对训练步的干扰（HBM 带宽争抢、PCIe 争抢）
- 区分"轮询线程开销" vs "探针注册开销"（`npu_nvme_set_step_ptr` 的 HBM buffer 分配）

**论文数据需求**：
- 表格：B1/B2/B3 步时对比（mean ± std, n≥30）
- 预期：B2 开销 < 2%（4 字节 DMA 读每 10ms，远小于训练步 468ms）
- 结论："后台轮询线程对训练步时的影响在测量噪声范围内"

---

### E2.2: 检测延迟分布

**问题**：从 step_counter 在图内更新（训练步完成）到后台线程检测到变化，延迟是多少？是否稳定？

**为什么这很重要**：
- 延迟决定了 delta 写盘的"起跑线"
- 如果延迟 > (步完成时刻 → 下一步 optimizer 更新时刻)，则写盘会与下一步的计算重叠
- 论文需要给出这个延迟的可预测性

**方法**：
```
1. 修改 GE 图：在 step_counter += 1 之前记录时间戳 ts_step_done (aclrt 不可在图内调用)
   → 替代方案：在 C 层 listener 记录两个时间戳：
     ts_detect: 检测到 step_counter 变化的时刻
     
2. 在 GE 图出口（如果可能）或 callback 中记录 ts_actual_step_done

3. 延迟 = ts_detect - ts_actual_step_done

4. 重复 50 步，绘制延迟分布直方图
```

**时钟源**：统一使用 C 层 `get_time_us()`（基于 `gettimeofday`），避免 C/Python 跨语言时钟偏差。Python callback 中的时间戳通过 ctypes 调用 C 层的 `get_time_us()` 获取，或通过记录 `time.perf_counter()` 并在离线分析时做线性拟合校准。

**实际可操作方案**：
```
C 层 listener 每轮 poll 记录:
  ts_poll:    get_time_us() — 本次轮询开始时刻
  ts_detect:  get_time_us() — 检测到 step_counter 变化的时刻 (首次检测到 cur_step >= expected)

Python epoch_end callback 记录（通过 ctypes 调 C 层 get_time_us 或 perf_counter 校准后）:
  ts_cb:      callback 开始时刻

对于 step N，检测延迟的上下界:
  upper_bound = ts_detect(N) - ts_cb(N-1)  （从上个 callback 结束到检测到）
  lower_bound = ts_detect(N) - ts_cb(N)    （从检测到本次 callback 开始，取负）
  
更精确的在线方法:
  在 callback 中通过 npu_nvme_set_probe_flag_value 写入一个时间戳到 HBM
  listener 读到 step_counter 变化时同时读取该时间戳
  → latency = ts_detect - ts_timestamp_in_hbm
  但这需要额外的 HBM 通信

最简可行方案（推荐）:
  利用 sink_size=1 下 callback 在 step 结束后立即触发的特性:
  C 层记录 ts_detect
  Python callback 开始时通过 ctypes 调用 C 层 get_time_us_proxy() 获取 ts_cb
  latency_estimate = ts_detect - ts_cb
  （由于 callback 在 step 完成后极短时间触发，此差值近似于检测延迟）

**论文数据需求**：
- 直方图：50+ 步的检测延迟分布
- mean ± std / median / p95 / p99
- 预期：mean ≈ 5ms (轮询周期的一半)，p99 < 10ms (一个轮询周期)

---

### E2.3: 写盘完成时序

**问题**：Delta write 是否在下一个 optimizer 更新之前完成？

**这是整个 FaF 方案正确性最关键的一个验证**。

**混合态分析**：

需要区分两种场景：
1. **FULL checkpoint 的混合态**（已知问题）：FaF 写全量参数时，SPDK 写盘 (~700ms) 与下一步 optimizer 更新有时间重叠，导致最后 ~35% 参数可能是"混合态"（部分来自 step N，部分来自 step N+1 的 optimizer 更新）。这对全量检查点影响小（参数变化 ~1e-7），但对增量检查点不可接受——delta 必须记录确定步状态。

2. **Delta frame 的混合态**（待验证）：quant_buf 是 GE 图输出的静态 HBM buffer，在 step N 图执行结束时已固定，不会被 step N+1 的 optimizer 修改。因此 delta frame 的内容是确定性的。**但关键问题是**：C 层 listener 读取 quant_buf 时（`aclrtMemcpy HBM→DMA`），如果 step N+1 的 GE 图已经在执行并更新了 quant_buf（因为 quant_buf 被 Assign 到新的值），就会读到"step N 的部分 + step N+1 的部分"。

**预期**：delta write 很快（~159MB in 45ms, 3350 MB/s），而训练步间隔 ~468ms。listener 在检测到 step_counter 变化后立即读取 quant_buf，此时 step N+1 的图大概率还未启动（或至少还未执行到 Assign quant_buf）。因此 delta frame 应完全干净。但需要实验验证。

**方法**：
```
时间线采集（C 层 + Python 层双打点）：

  C 层 listener:
    ts_detect(N):   检测到 step N 完成
    ts_dma_start:   aclrtMemcpy(quant_buf, HBM→DMA) 开始
    ts_dma_end:     aclrtMemcpy 完成
    ts_spdk_start:  spdk_nvme_ns_cmd_write 提交
    ts_spdk_done:   SPDK callback 返回（nvme_write_complete_cb）

  Python callback (epoch_end):
    ts_callback_start: 回调开始
    ts_callback_end:   回调结束

分析：
  gap = ts_callback_start(N+1) - ts_spdk_done(N)
  if gap > 0: delta write 在下一步 callback 之前完成 ✅
  if gap < 0: 存在重叠 ⚠️
```

**实际可操作方案（分两层做）**：

**Level 1: 粗粒度验证（Python 打点）**
```
每步 callback 中记录:
  ts_cb_begin:     callback 开始时间
  ts_delta_flush:  触发 delta write (调用 npu_nvme_write_delta)
  ts_cb_end:       callback 结束时间
  
计算: delta_write_time = ts_spdk_done - ts_delta_flush
     next_step_gap = ts_cb_begin(N+1) - ts_spdk_done(N)
```

**Level 2: 细粒度验证（C 层 profiling）**
```
利用现有的 enable_profiling 机制：
  process_write_pipeline 内已经有 ts_submit / ts_npu_done / ts_spdk_done
  扩展: 增加 ts_opt_begin / ts_opt_end 来自 callback 传入（或从 step_counter 反推）

输出: time_delta.csv 每行包含完整的时序分解
```

**论文数据需求**：
- 时间线图（Gantt 风格）：5-10 个连续 step，展示 step_counter 变化、delta write 各阶段、callback 时间窗
- 统计：gap 的 min/max/mean/std
- 结论："delta write 平均在下一步 callback 前 XXms 完成，最坏情况下仍有 XXms 余量"

---

### E2.4: 同步 vs 异步写盘对比

**问题**：如果将 delta write 放在 callback 中同步执行，步时会增加多少？FaF 的异步方案带来了多少收益？

**方法**：
```
条件 A (同步 — sync_meta_io):  
  callback 内调用 npu_nvme_write_delta()（内部走 sync_meta_io，~1ms 延迟）
条件 B (同步 — write_batch):   
  callback 内调用 npu_nvme_write_batch()（走 pipeline DMA，~45ms 延迟）
条件 C (异步 / FaF):           
  callback 仅记录状态，C 层 listener 异步触发 write_delta

三种条件下各运行 30 步，对比:
  - 步时 (wall-clock per step)
  - 步时稳定性 (std/mean)
  - Delta write BW
```

**为什么比较两种同步模式**：
- `sync_meta_io` 延迟极小（~1ms），SPDK 写太快可能看不出 FaF 的异步优势
- `write_batch` 延迟较大（~45ms），更接近真实场景（大 delta frame 或多 chunk）
- 如果 `sync_meta_io` 的同步开销已经淹没在步时噪声中，说明对于小 delta frame，FaF 不是必需的——这是需要诚实面对的实验结果
- 如果 `write_batch` 的同步开销显著（~10% 步时），FaF 的收益才有说服力

**论文数据需求**：
- 对比表格：同步 vs 异步的步时、BW、稳定性
- 如果有多个 sink_size 设置，做二维对比

---

### E2.5: FaF 在最优 sink_size 下的 per-step 触发验证

**问题**：在 E0 确定的最优 sink_size（预期 > 1）下，FaF 能否实现 per-step delta write？这是 I2 核心主张的直接验证——"解耦训练配置与检查点触发粒度"。

**为什么这是核心验证**：
- E2.1/E2.7 验证了 FaF 没有副作用
- E2.3/E2.6 验证了 FaF 的正确性
- **E2.5 验证了 FaF 的存在意义**——如果没有这个实验，读者可以质疑："为什么不直接用 sink_size=1？"
- 证明了在最优训练配置下，per-step delta write 只能通过 FaF 实现

**方法**：
```
固定 FaF ckpt_interval=1（每步触发 delta write）
在 E0 确定的最优 sink_size 下（预期 10-50）

配置 A (对照组):  sink_size=1, callback 同步 delta write（当前可用方案）
配置 B (实验组):  最优 sink_size, FaF per-step delta write

运行 50 步，对比:
  - 吞吐 (steps/s) — A vs B 的差距就是 FaF 带来的训练速度收益
  - Delta write 触发准确率（B 中 listener 是否每个 step 都触发）
  - 步时稳定性
```

**论文数据需求**：
- 对比表格：配置 A vs B 的吞吐、触发准确率
- 关键数字："在最优 sink_size=N 下，FaF 实现了 per-step delta write，同时训练吞吐比 sink_size=1 方案高 X%"
- 这个 X% 是整篇论文的 I2 部分的"核心卖点"

---

### E2.6: 多步一致性

**问题**：FaF listener 在长时间运行中是否会遗漏触发？

**方法**：
```
运行 100 步训练 (sink=TRUE, GRAPH_MODE, sink_size=1)
C 层 listener 记录每次触发的 step 值
Python 层记录 callback 中的 step 值

验证:
  - listener 触发次数 == 步数 / ckpt_interval
  - 每次触发的 step 值连续且无遗漏
  - 无 false positive (step 未变但触发)
  - 无 false negative (step 已变但未触发)
```

**论文数据需求**：
- 触发日志表（step, ts_detect, triggered）
- "100 步中 0 遗漏触发"

---

### E2.7: 线程资源开销

**问题**：后台轮询线程的 CPU 和内存开销是否可接受？

**方法**：
```
在 B1 (纯 MS 无线程) 和 B2 (线程运行) 条件下:
  - 使用 top/htop 记录 CPU 利用率
  - 使用 /proc/PID/status 记录 VmRSS 内存
  
预期:
  - CPU: < 0.1% (10ms sleep + 微秒级 DMA 读)
  - 内存: < 10MB (step_poll_buf 4 bytes + probe_flag_host 4 bytes + 线程栈)
```

**论文数据需求**：
- 简短的一句话即可："后台线程消耗 < 0.1% CPU 和 < 10MB 内存"

---

