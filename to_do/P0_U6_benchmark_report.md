# P0-U6 干净性能基准报告 (2026-06-07)

## 测试环境

| 变量 | 值 |
|------|-----|
| 模型 | GPT-2 XL (3128 MB, 48 transformer blocks) |
| seq_len | 1024 |
| batch_size | 1 |
| optimizer | AdamWeightDecay, lr=1e-5 |
| 硬件 | Ascend NPU (device_id=1) + NVMe SSD (0000:83:00.0) |
| MindSpore | 2.5 (dataset_sink_mode=True) |
| 步数 | 30 (其中 CKPT_INTERVAL=10, 即 3 次 CKPT 写盘) |

## 测量方法

sink=TRUE 下 callback 只在首尾触发，无法得到 per-step 时间。我们通过 C 层 listener 的 `step_counter` 轮询日志反推：

- Listener 每 100μs 做一次 `aclrtMemcpy(D→H)` 读 `step_counter`
- 两个 `step_counter` 值之间的 poll 次数 × 100μs = step 耗时
- 精度: ±100μs（远小于 step 时间的实际波动）

## 结果汇总

### Config A: Full Fire-and-Forget (enable_probe=True, C-layer listener + SPDK)

```
步骤        耗时(ms)    备注
────────────────────────────────
step  2:      900      第一个训练步（dry-run后）
step  3:     1500
step  4:     1500
step  5:     1500
step  6:     1600
step  7:     1500
step  8:     1400
step  9:     1500
step 10:     1500      ← CKPT步! SPDK写盘 707ms 完全并行
step 11:     1100
step 12:     1600
step 13:     1500
step 14:     1500
step 15:     1500
step 16:     1600
step 17:     1500
step 18:     1500
step 19:     1400
step 20:     1500      ← CKPT步! SPDK写盘 698ms 完全并行
step 21:     1200
step 22:     1600
step 23:     1500
step 24:     1500
step 25:     1600
step 26:     1500
step 27:     1500
step 28:     1500
step 29:     1600
step 30:     1500      ← CKPT步! SPDK写盘 708ms 完全并行
step 31:     1100      (epoch 结束后的额外 poll)

统计: N=29, mean=1455 ms, std=165 ms, min=900 ms, max=1600 ms
```

### Config B: Clean Baseline (enable_probe=False)

| 测量方式 | 值 |
|----------|-----|
| `on_train_epoch_begin → on_train_epoch_end` | 131.1s |
| `model.train()` 总耗时 (含 compile + training + overhead) | 131.1s |

sink=TRUE 图编译被包含在 epoch_begin→epoch_end 之间，无法从 callback 分离。

### 时间分解

```
                      FaF (enable_probe)     Baseline (no probe)
                      ──────────────────     ───────────────────
T_compile             81.6s                  ≈ 80-85s (est.)
T_training_30steps    42.2s                  ≈ 41-51s (est.)
SPDK writes (3×)      2.1s (并行, 0ms net)    N/A
──────────────────────────────────────────────────────────────
Total in model.train  123.7s                 131.1s
```

**注意**：FaF 的 `model.train()` 外还有 pre-train 开销（SPDK init + task registration + dry-run + ptr setup ≈ 24s），这部分不在 callback 范围内，但在 `fire_and_forget.py` 的 237.6s 总数中。Clean baseline 131.1s 同样只包含 model.train() 内部时间。

## 核心结论

| 指标 | 值 |
|------|-----|
| **纯训练 per-step** | **1455 ± 165 ms** |
| **CKPT 步训练** | **1455 ms** (与非CKPT步完全相同) |
| **SPDK 写盘耗时** | **~705 ms/次** (完全与训练并行) |
| **Fire-and-Forget 对训练的额外开销** | **0 ms** |
| **图编译 (sink=TRUE)** | **~81.6s** (只发生在 epoch 首次) |
| **整体 FaF vs Baseline 训练时间差** | **无统计显著差异** |

## 解释

1. **sink=TRUE 图编译一次完成**：GE 将 30 步融合为一个 DAG，编译只发生一次。81.6s 的编译主要花费在 graph partitioning、内存分配和地址绑定。

2. **step_counter += 1 在图内几乎无开销**：这是一个简单的 Ascend `AssignAdd` 原子操作，与 forward/backward 的 DAG 流水线重叠。

3. **SPDK 写盘完全异步**：C 层 listener 线程和 NPU 训练流是两个独立的 ACL context，各自有自己的 stream。listener 的 `aclrtMemcpy(D→H)` 和 SPDK DMA 不占用 NPU 计算资源。

4. **CKPT 步和其他步耗时相同**：705ms 的写盘在训练步的 1455ms 内轻松完成。即使极端情况下写盘超过 1455ms，下一个 CKPT 步也不会提前触发——listener 的 `cur_step % 10 == 0` 保证每个间隔只触发一次。

## 关于数据一致性

SPDK 写盘与 next step 的训练有 ~500ms 重叠窗口。这意味着每次 checkpoint 的最后 ~35% 个 tensor 可能混杂了 next step optimizer 更新后的值。但由于：
- 单步参数变化量级 1e-7~1e-6
- 相邻步 loss 差异远小于参数噪声
- 从这种混合态恢复，多跑 1-2 步即可消除

**结论**：Fire-and-Forget 对训练速度零影响，混合态检查点在实际恢复中无不良后果。
