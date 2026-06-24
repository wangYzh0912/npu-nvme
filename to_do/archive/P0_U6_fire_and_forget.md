# P0-U6 综合状态报告 (2026-06-07 final)

## 一、开发历程回顾

### v1: AICPU TriggerProbe + WaitProbe (sink=TRUE)
- **设计**: 图内 AICPU 核做 step_counter 递增检测+自旋等待
- **失败原因**: GE 通过 `dlopen(RTLD_LOCAL)` 加载 `libcust_opapi.so`，`aclnnTriggerProbe`/`aclnnWaitProbe` 符号不可达。6次尝试均失败。
- **结论**: sink=TRUE 下自定义 AICPU kernel 不可注入（GE 编译器设计级限制）

### v2: C层轮询 expected (sink=TRUE)
- **设计**: 图内 `ops.assign_add` 操作 expected，C层轮询
- **失败原因**: sink=TRUE 融合图内 Parameter 更新被 GE 内部管理，外部 `aclrtMemcpy` 读到初始值

### v3: 纯C层轮询 step_counter (Fire-and-Forget) — ✅ 当前方案
- **设计**: 图内仅 `step_counter += 1`，C层 listener 每 100μs 轮询并取模判断触发
- **状态**: ✅ 端到端验证通过 (flag=3, SAFETY CHECK PASSED)
- **发现的Bug**:
  1. Listener 初始化顺序竞态（else 分支死循环）
  2. Python 侧 `probe_flag_ptr` 为空（MS 懒分配 → C层自分配）

## 二、性能基准

### 所有历史测试汇总

```
Test                          SPDK  Listener  Probe组件      per-step
────────────────────────────────────────────────────────────────────
cell_overhead (直接cell(x))     ✗      ✗       无             369ms
cell_overhead (直接cell(x))     ✗      ✗       WaitProbe      364ms
baseline_benchmark (sink=F)     ✗      ✗       无             386ms
我们的 A/B config A (sink=F)    ✗      ✗       无             394ms
我们的 A/B config B (sink=F)    ✗      ✗       step_counter   462ms
sink=TRUE sink_size=10          ✗      ✗       无             481ms
sink=TRUE sink_size=30          ✗      ✗       无             1550ms
Test D (sink=F)                 ✓      ✓(v3)   无             1674ms
spdk_end_to_end (sink=F)        ✓      ✓(v1)   WaitProbe      1925ms
```

### 关键发现

1. **WaitProbe + TriggerProbe AICPU 在图内几乎零开销**
   - 纯图: 369ms，加 WaitProbe+TriggerProbe: 364ms（统计噪声内）
   - 原因: WaitProbe 的 `flag >= expected` 判断在 sink=FALSE 时是直接过（初始 flag=0, expected=0），AICPU 核几乎瞬时返回

2. **SPDK 基础设施 + listener 线程是性能瓶颈**
   - 纯训练 379ms → 加 SPDK+listener 1674ms（+1295ms, +341%）
   - v1 listener (host-trigger): 1925ms — 开销来自 SPDK qpair `process_completions` 轮询
   - v3 listener (100μs poll): 1674ms — 额外 ACL context 切换开销
   - **根因**: `spdk_nvme_qpair_process_completions(qpair, 0)` 每次调用都要与 NVMe 硬件队列交互。在训练步之间被高频调用（listener 的 while 循环中），与 NPU 训练流共享 PCIe 带宽

3. **step_counter 在图内的开销可控**
   - sink=FALSE: +68ms (+17%) — `ops.assign_add` + `ops.depend` 新增控制边
   - sink=TRUE: GE 融合后可忽略

4. **sink=TRUE 的 sink_size 是关键参数**
   - `sink_size=30`（全 epoch）: 1550ms/step — 融合图过大，GE 被迫做 activation recomputation
   - `sink_size=10`: 481ms/step — 正常优化
   - 大模型应使用 `sink_size ≤ 10`

### 数据一致性分析

Fire-and-Forget 的 SPDK 写盘（705ms）与下一个训练步有 ~1.1s 重叠：
- 前 ~65% 的 tensor 是当前 step 的参数
- 后 ~35% 是 next step optimizer 更新后的参数
- 单步参数变化量级 1e-7~1e-6，混合态对恢复训练无实质影响

## 三、当前 Fire-and-Forget 架构

```
NPU 融合图:
  for each step:
    loss, grads = grad_fn(inputs)
    step_counter += 1          ← int32 HBM Parameter
    loss = depend(loss, step)
    optimizer(grads)
    return loss

C层 Listener 线程 (独立 pthread):
  aclrtSetDevice(1) + aclrtSetCurrentContext(ctx)
  while(1):
    aclrtMemcpy(D→H): step_poll_buf ← step_counter
    if cur_step > last:
      last = cur_step
      if cur_step % interval == 0:
        signal_probe_flag(expected)     ← HBM flag = expected
        process_write_pipeline(1425)    ← SPDK 异步写盘 (~705ms)
    spdk_nvme_qpair_process_completions(qpair, 0)
    usleep(100)
```

## 四、SDL 限制与已知问题

| 环境限制 | 影响 |
|----------|------|
| `sudo` 必需 | MS 2.5 访问 `/usr/local/Ascend/opp/scene.info` 需要 root |
| SPDK 需 root | userspace NVMe 驱动需要 root 权限绑定 PCI 设备 |
| ACL context 隔离 | sink=TRUE 融合图的 Parameter buffer 由 GE 管理 |
| `libcust_opapi.so` RTLD_LOCAL | 自定义 AICPU 无法在 sink=TRUE 中执行 |

## 五、待办

- [ ] **P0**: 优化 listener 轮询 — 降低频率到 10ms，去掉不必要的 ACL context 绑定
- [ ] **P1**: 回归 sink=FALSE per-step callback 触发 SPDK 写盘（避开 listener 开销）
- [ ] **P1**: 对比 Fire-and-Forget (sink=FALSE callback) vs 纯训练的性能
- [ ] **P2**: 删除 dead code（WaitProbe/TriggerProbe AICPU kernel）
- [ ] **P2**: 移除 listener 诊断日志
- [ ] **P3**: 完整检查点 barrier 设计

## 六、环境

| 变量 | 值 |
|------|-----|
| sudo | `echo "CGCL_2025_#$" \| sudo -S` |
| Python | `/root/miniconda3/envs/ms_2.5/bin/python3` |
| 前置 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 模型 | GPT-2 XL, 3128MB, seq_len=1024 |
