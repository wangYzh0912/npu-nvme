# 当前任务进展

## 已完成

- [x] **P0-U5 统一实验重跑** — 基线 A-F + SPDK 端到端 + Cell 开销隔离
- [x] **Sink 隔离分析** — 三角定位：图注入开销 ~0%、sink=FALSE 贡献 1508ms
- [x] **P0-U6 C 层架构简化** — 详见下方

### P0-U6 本轮完成 (Jun 7)

**核心修复：绕过 trigger_buf 懒分配 Bug**

用 `expected` 别名替代 `trigger_buf`：
- C 层 `signal_probe_flag(ctx, value)` — 直接设 flag=value（而非 flag+=1）
- C 层 listener 轮询 expected 设备地址（直接 aclrtMemcpy）
- Python 层 `trigger_probe(step, interval, expected, expected)` — expected 双用
- ACLNN_SUCCESS→0 修复

**C 库已重新编译安装** — `build_out/lib/libnpu_nvme.so.1.0`

## 当前阻塞

**aclnn host-side wrapper 缺失** — GE 在 sink=TRUE 下需要 `aclnnWaitProbe` / `aclnnTriggerProbe` 符号才能 launch AICPU 算子。

源码已写好：`kernels/trigger_probe/op_host/aclnn_wait_probe.cpp`
但尚未编译为 `libcust_opapi.so` 安装到 `build_out/opp/vendors/customize/op_api/lib/`

**上轮测试结果**：30 步完成但 `final_flag=0` — 零触发，且 `Dlsym aclnnWaitProbe failed` 警告。

## 下一步

- [ ] 编译 aclnn wrapper → 安装到 opp → 重跑 30 步测试（需手动 sudo 编译权限）
- [ ] 验证 `final_flag >= 3`
