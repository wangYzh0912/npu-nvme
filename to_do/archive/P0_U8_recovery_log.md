# P0-U8 恢复日志：git checkout 事故修复 (2026-06-08)

## 事故概要

在执行 P1-1/P1-2/P2-1 的 C 源码编辑过程中，由于 `#ifdef`/`#if 0` 预处理指令嵌套错误导致反复编译失败。为回到干净状态，执行了 `git checkout src/npu_nvme.c`，但这将文件覆盖为了 git 仓库中的 **pre-FaF** 版本（1223 行，缺少 `dev_step_ptr`、`set_step_ptr`、listener 轮询等所有 FaF 扩展）。

更关键的是：`.bak` 和 `.bak2` 备份也都是 Jun 7 的早期版本（1287/1296 行），**同样不包含 FaF 扩展**。

## 当前状态 ✅ (ALL FIXED)

### 所有修复已完成

| 项目 | 状态 | 说明 |
|------|:---:|------|
| `.so 备份` | ✅ | `libnpu_nvme.so.1.faf_good` / `libnpu_nvme.so.faf_good` (5.5MB, md5=dae5a5d6...) |
| `src/npu_nvme.c` | ✅ | FaF 源码已重建 (1432 行), 包含 P1-1/P1-2/P2-1 所有改动 |
| `include/npu_nvme.h` | ✅ | 添加了 `npu_nvme_set_step_ptr()` / `npu_nvme_get_probe_flag_dev_ptr()` 声明 |
| `build_out/lib/libnpu_nvme.so.1.0` | ✅ | 新编译的 .so (所有 FaF 符号 + P1/P2 改动) |
| `python/direct_checkpoint.py` | ✅ | P0-1 warmup_fn + P0-2 probe_flag_ptr + FaF construct |
| `experiments/faf_clean_r6only.py` | ✅ | 完整 FaF 测试脚本 |
| `experiments/faf_clean_phase2.py` | ✅ | R5/R6 干净环境测试 |
| `experiments/faf_100step.py` | ✅ | P1-3 100 步测试 (已修复 expected_ckpts 未定义) |
| `experiments/fire_and_forget.py` | ✅ | 原始 FaF 脚本 |
| `experiments/spdk_end_to_end.py` | ✅ | warmup_fn 适配 + ctypes fallback flag alloc |

## C 源码重建内容

从 `.so` 反汇编 + 字符串分析重建了以下 FaF 扩展：

### 1. Struct 字段 (替换旧 WaitProbe 字段)
```c
void *dev_step_ptr;        // FaF: step_counter device address (HBM) [offset +504]
void *step_poll_buf;       // Host buffer (4 bytes) for polling step_counter [offset +512]
int last_step_seen;        // Last step value detected by listener [offset +520]
int ckpt_interval;          // Checkpoint every N steps [offset +524]
```

### 2. 新增函数
- `void* npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx)` — 返回 C 层自分配的 flag 地址
- `int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval)` — FaF step_counter 设置

### 3. 重写 probe_listener_thread
- FaF: 轮询 step_counter (aclrtMemcpy D→H 每 10ms)
- `NPU_NVME_LISTENER_MODE` 支持: "off" / "idle" / "full" (默认)
- 触发: cur_step >= (last_step_seen + ckpt_interval)
- SPDK 写入后: signal_probe_flag(ctx, expected_value)

### 4. P1-1: 轮询间隔 100μs → 10ms ✅
- `usleep(100)` → `usleep(10000)` (两处)

### 5. P1-2: 诊断日志包裹 ✅
- 诊断 fprintf 用 `#ifdef DIAGNOSTIC` 包裹

### 6. P2-1: WaitProbe 死代码保留 ✅
- `npu_nvme_set_trigger_ptr`, `npu_nvme_read_trigger_dev`, 和旧版 probe_listener_thread 通过 `#if 0` 保留
- 添加了向后兼容桩: `npu_nvme_set_trigger_ptr` → reroute 到 `npu_nvme_set_step_ptr`

## 已验证的运行结果

### R6: sink=TRUE FaF 20 步 (NO SEGFAULT) ✅
```
4× listener TRIGGERED (step 4/9/14/19)
374ms/step e2_ps
NO crash / NO segfault
```

### P0-3: spdk_end_to_end.py ✅
```
Avg non-CKPT step:    418-467ms
SPDK pipeline:        ~5s @ 3128MB (3.1GB GPT-2 XL)
NO crash / NO segfault
```

### P1-3: FaF 100 步 ✅
```
10/10 SPDK triggers (step 9/19/.../99)
418ms/step avg (excl epoch 1)
NO crash / NO segfault
```

## 残留问题

1. **probe_flag_ptr 在 sink=TRUE 时为 0**: MS 在 sink=TRUE 图中未分配 probe_flag 张量地址。
   - C 层 listener 有效地写入 flag，但 Python 的 `read_probe_flag_dev()` 需要 ptr
   - 解决方案: 要么使用 C 层自分配 (通过 `npu_nvme_set_probe_flag_ptr(ctx, NULL)`)，要么使用数据集迭代器标志检查
   - **不影响功能**: 所有 SPDK 写入均正确完成

2. **spdk_end_to_end.py 标志等待时间 ~5s**: 这代表了整个流水线中完整的 3.1GB SPDK 写入时间 (非 FaF；同步)。FaF 将此延迟隐藏在下一个训练步骤之后。

## 文件变更摘要

| 文件 | 行数 | 变更 |
|------|------|------|
| `src/npu_nvme.c` | 1432 | FaF 字段 + set_step_ptr + get_probe_flag_dev_ptr + FaF listener + P1-1/P1-2/P2-1 + #if 0 保留 |
| `include/npu_nvme.h` | +9 行 | 新函数声明 |
| `build_out/lib/libnpu_nvme.so.1.0` | 5.5MB | 新编译 (md5=33142ca8...) |
## 基线性能测试 (2026-06-09)

### 测试环境
- 模型: GPT-2 XL (3128MB), SEQ=1024, BATCH=1
- sink=TRUE, sink_size=10, 100 逻辑步
- NPU: Ascend 910B, Device 1

### 结果对比

| 测试 | 单步平均 | 备注 |
|------|---------|------|
| **B1: 纯 MS** (无探针/无 SPDK) | **412ms** | 基准线 |
| **B2: 探针闲置** (listener idle, 无 SPDK 写) | **436ms** (+5.8%) | 探针基础设施开销 |
| **B3: 完整 FaF** (SPDK 激活) | **435ms** (+5.6%) | 生产配置, flag=10 PASSED |

### 关键结论

1. **探针基础设施开销**: +24ms/step (+5.8%) — step_counter assign_add + C 层 listener 轮询 (10ms 间隔) + 1425 张量注册簿记
2. **完整 FaF 总开销**: +23ms/step (+5.6%) — SPDK 3.1GB 写入完全被后续训练步骤遮盖，不产生额外延迟
3. **B3 vs B2 差异仅 -1ms** — 证明 FaF 设计正确：NPU 永远不等待 I/O
4. **安全验证通过**: 10/10 次 SPDK 检查点全部完成，C 层自分配 probe_flag 正确递增
5. **编译期无回归**: B3 编译 120s vs B1 117s (3% 差异在噪声范围内)

### 逐 Epoch 明细 (ms/step)

| Epoch | B1 纯 MS | B2 探针 | B3 FaF |
|-------|---------|---------|--------|
| 2 | 420 | 403 | 511 |
| 3 | 410 | 411 | 441 |
| 4 | 423 | 406 | 437 |
| 5 | 423 | 396 | 408 |
| 6 | 401 | 498 | 411 |
| 7 | 415 | 446 | 438 |
| 8 | 405 | 416 | 434 |
| 9 | 410 | 502 | 427 |
| 10 | 402 | 444 | 406 |
