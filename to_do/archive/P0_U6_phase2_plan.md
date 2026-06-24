# P0-U6 Phase 2 规划 (Jun 7, 2026)

## 一、当前状态确认

### Phase 1 已完成

| 成果 | 状态 |
|------|:---:|
| v3 纯 C 层 Fire-and-Forget 架构 | ✅ |
| `aclrtMemcpy(D→H)` 跨 context 读 step_counter | ✅ |
| Listener 每 100μs 轮询，step % 10 == 0 触发 SPDK 写盘 | ✅ |
| 训练无阻塞（图内无 WaitProbe） | ✅ |
| 30步端到端验证：flag=3 >= expected=3 | ✅ |
| **纯训练速度: ~1.5s/step** (之前 7.9s 是错误的——包含了图编译) | ✅ |

### 性能数据纠正

```
总耗时 237.6s 的分解:
  模型加载 + dry-run:          ~82s
  sink=TRUE 图编译:            ~90s   ← 只在 epoch 开始前一次
  纯训练 29步 (step 2→30):    ~42s   → 1.45s/step
  注册/初始化等:               ~24s

SPDK 写盘 ~705ms，与训练完全并行
CKPT 步 ~1.5s，非 CKPT 步 ~1.5s
写盘对训练的阻塞: 0ms
```

## 二、数据一致性：增量 vs 完整检查点

### 现状: 混合态检查点

```
step 10 图结束  ─→ Listener 检测 (~80ms) ─→ SPDK 写盘开始 (705ms)
                                                   │
step 11 图执行 (1.5s) ─────────────────────→      │ 重叠 ~500ms
                                                   │
                                            SPDK 写盘结束
```

1425 个 tensor 中，前 ~65% 是 step 10 optimizer 之前的值，后 ~35% 是 step 11 中途的值。
**这本质上是一个混合态的检查点**。但由于相邻 step 参数变化在 1e-7~1e-6 量级，实际影响可忽略。

### 双层检查点架构

```
频率          类型        一致性要求    机制
────────────────────────────────────────────────
每10步        增量(delta)   宽松          当前 Fire-and-Forget (图内无阻塞)
每100步       完整(full)    严格          需图内 barrier (WaitProbe)
```

**增量检查点**: 用于快速恢复。恢复后从最近增量继续训练，因混合态导致的微小参数偏差通过 1-2 个额外训练步自然消除。

**完整检查点**: 用于长期存档和跨实验比较。需要在图内插入 barrier 确保一致性。WaitProbe 在这里仍然有价值——但我们现在不需要它，因为 30 步的 PoC 测试不需要完整检查点。

### 为什么之前的 WaitProbe 实验不是无意义的

| 实验 | 验证了什么 |
|------|-----------|
| v1 AICPU kernel 编译 | 确认了自定义 AICPU kernel 可以编译和加载（在非 sink 模式可用） |
| v1 GE dlsym 失败 | **关键发现**: sink=TRUE 下 GE RTLD_LOCAL 加载 libcust_opapi.so，外部 aclnn 符号不可达。这是不可逾越的架构限制 |
| v2 expected 别名 | 验证了 sink=TRUE 下 Parameter 的更新行为由 GE 内部管理 |
| v3 C 层 poll 成功 | 验证了 `aclrtMemcpy(D→H)` 可以跨 context 读取 MS Parameter 的 HBM 数据 |
| SPDK 写盘流水线 | 验证了 3.13GB/705ms/4.4GB/s 的硬件带宽上限 |
| channel_write 字节计数 | 提供了精确的 per-tensor 传输数据 |

## 三、下一步任务

### 任务 1: 干净的 sink=TRUE 性能基准重采集

**原因**: 之前的 "7.9s/step" 包含了图编译，实际训练仅 1.5s/step。需要用不带任何 checkpointing 开销的基线重新测量。

**方法**:
1. 创建 `experiments/sink_baseline.py`，与 fire_and_forget.py 相同模型但 `enable_probe=False`
2. 记录 epoch_begin→epoch_end 的 wall-clock 时间
3. 与 fire_and_forget.py 的纯训练时间对比

**预期**: 1.4-1.5s/step（有无 CKPT 都在此范围，因为 SPDK 写盘完全并行）

### 任务 2: 添加 per-step profiling hook

**原因**: sink=TRUE 下 callback 只在首尾触发，无法得到 per-step 时间。但我们可以从 Listener 的 step_counter 轮询日志中反推 per-step 间隔。

**方法**:
1. 在 Listener 中添加 `ts_first_seen[step] = get_time_us()` 记录每个 step 首次被检测到的微秒时间戳
2. 训练结束后 Python 读取这些时间戳，计算 per-step latenc y
3. 输出 CSV: `step, wall_ms, is_ckpt_step`

### 任务 3: 移除调试日志

**原因**: `faf_run10` 的 stderr 有 1246 行，其中 ~1240 行是 `listener poll#...` 调试输出。生产环境这不可接受。

**方法**:
1. 将 `listener poll#` 日志编译时通过 `#ifdef DIAGNOSTIC` 包裹
2. 保留 `TRIGGERED` 和 `write pipeline` 日志（稀疏，有意义）
3. 添加 `NPUNVME_CONTEXT` 环境变量控制诊断级别

### 任务 4: 设计完整 checkpoint 检查

**原因**: 虽然增量检查点不需要严格一致性，但我们应当验证混合态的误差确实在可接受范围内。

**方法**:
1. 在 step 10 的 callback 中（epoch_end），python 软件读取所有权重值保存为 ground truth
2. 比较 C 层写盘的检查点与 ground truth 的差异
3. 预期：差异 < 1e-6 per parameter（即相邻步间的梯度更新量）

### 任务 5 (远期): WaitProbe 在低频完整检查点中的复用

**原因**: 每 100 步的完整检查点需要在 optimizer 更新前 snapshot 所有参数。图内 barrier（WaitProbe）是保证一致性的正确方案。

**阻塞项**: GE RTLD_LOCAL 限制仍未解决。完整检查点的 barrier 可能需要采用以下替代方案之一：
- **方案 A**: 在 step_end callback 中（sink=FALSE 时可用）同步 save
- **方案 B**: 使用 MS 原生 `CheckpointConfig` + `ModelCheckpoint` 回调
- **方案 C**: 研究是否可以通过修改 GE 配置绕过 RTLD_LOCAL

## 四、执行优先级

| 优先级 | 任务 | 预计工作量 |
|--------|------|-----------|
| P0 | 任务 1: 干净基线 benchmark | 30min |
| P1 | 任务 2: per-step profiling | 1h |
| P1 | 任务 3: 移除调试日志 | 15min |
| P2 | 任务 4: 检查点一致性验证 | 1h |
| P3 | 任务 5: 完整检查点 barrier | 待定 |

## 五、环境

| 变量 | 值 |
|------|-----|
| sudo | `echo "CGCL_2025_#$" \| sudo -S` |
| Python | `/root/miniconda3/envs/ms_2.5/bin/python3` |
| 前置 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| GPT-2 XL | 3128 MB params, seq_len=1024 |
| NVMe | 0000:83:00.0, SPDK userspace driver |
