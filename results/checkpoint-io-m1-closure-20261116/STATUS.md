# M1 closure 执行状态

快照时间：2026-11-16T19:26:44+08:00。该文件是时点快照，不是全套验收通过声明。

代码分支 codex/checkpoint-io-m1-closure-20261116，冻结提交 bfcabd3753148dafeea014c3d41c6a2a5bf2f67c。
相关软件检查357项通过；本轮未修改原生 C/ABI/候选二进制。

## 当前运行

修正后的完整队列已启动，PID 3138587。状态 running，阶段 e1_full_boundary。
四个训练 rank 已启动，修正版实际采样确认覆盖全部四个 rank 和物理 NPU4–7。
执行顺序 E1 → E4 → E2/E3；E2 预登记18次训练、6次 fresh restore。
任何门槛失败都会停止后续运行。硬件验收尚未完成，新的 P95、10%扰动及设备归因不能提前声称通过。

持续更新的持久状态：/home/user7/npu-nvme-evidence/aic-aiv-io-20261116-r1/result.json。
原始 tmpfs：/dev/shm/user7-aic-aiv-io-closure-20261116-r1。
执行契约：../../AIC_AIV_IO_M1_CLOSURE_20261116.md。

## 首轮被拒绝的原因与恢复

0bac9dfe 的 E1/T0 已完成32步训练，四 rank safe close 成功，supervisor 释放 lease。
但 MindSpore worker 使用独立 session，原进程组采样遗漏 rank，覆盖 gate 拒绝。
该次失败及证据保留在 pilot-rejected，不纳入正式 A/B 样本。

修正使用完整 PPID 子树与内核 start ticks；357项检查通过，live 采样确认四 rank 齐全。
重启监督程序核对旧进程退出、rank close、lease 释放、错误原因和旧源身份后，
才 fast-forward 与启动 -r1。没有修改运行中的冻结源，没有强制释放设备借用的 buffer。

上一轮最终 M1 归档：codex/checkpoint-io-results-20261111 的 bf9ec838。
262个 payload 加 manifest 共263个文件全部纳入 Git，并逐 SHA256 验证；不与新样本混合。
