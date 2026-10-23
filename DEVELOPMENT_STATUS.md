# 开发状态与交付范围

更新：2026-10-24。本文件由当前调用链和实际验证记录重建，替代工作区中的空文件。

## 已完成

- C ABI 升为 3，移除旧 FaF listener、delta bookkeeping、无 caller 的 LiveCapture
  和 cleanup_legacy，以及占位 C 测试。`src/npu_nvme.c` 保留 ABI 导出；实现位于
  CMake 编译的 runtime/reactor/request/DMA/read/write/metadata 等模块。
- D1 save/restore 从准入开始进入同一 operation registry，drain 按水位等待；close
  先停止准入，避免 freeze 尚未结束就关闭传输。
- Qwen FULL 保存模型、Adam、训练位置与 RNG 控制态；四 rank 经有界 Host bridge
  交给单一 SPDK owner 提交 D2。raw commit 后、文件目录发布前的中断可在 fresh mount
  后按 retained receipt 与四 rank journal 对账，恢复仍须通过完整校验和全 rank ready。
- 静态 Host/hugepages/region 预算与运行时实际几何、ACL 可用 HBM 准入已接入正式路径。
  逻辑 NPU ID 映射到物理卡锁键，已有跨用户 `/tmp` 锁文件不再通过 `O_CREAT` 打开。
- 修正 none 结果 gate，记录执行源码身份，验收工具使用流式摘要、输出文件系统空间准入
  及已完成只读用途模型 shard 的 SHA256 核验硬链接去重。
- 合入 `origin/master` 的 `64730754`：D2 validation purpose、bytes 边界修复、受控
  flush 故障注入与历史硬件证据。研究分支不合入本次主线提交。

## 验证与证据

ABI3 库：`/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/candidate/build/libnpu_nvme.so.3.0`，
SHA256 `070f87d5bc3a2a32723274c9319e5b08356f5f36163e3ad3594b81a12daf4189`。

| 范围 | 实际结果 |
| --- | --- |
| 原工作区 Python/CPU/ABI（含独立研究测试） | 424 passed；`/tmp/npu-nvme-full-closure-20261024-software.xml` |
| 隔离提交候选（合入上游后，排除研究实验） | 390 passed；`/tmp/npu-nvme-full-closure-20261024-integrated-software.xml` |
| 生产 C 源 + external stubs，ASan/UBSan | 45 passed；`/tmp/npu-nvme-full-closure-20261024-c.xml` |
| root 已有跨用户锁文件回归 | `native_owner_existing_file` PASS；无设备 I/O |
| 历史 ABI3 NPU4–7 targeted FULL | 2026-10-22，7 场景、5 比较通过；`...20261021-005/result.json` |
| 本轮 NPU4–7 targeted FULL | 7 场景按预期完成，5 次 all-rank oracle comparison 通过 |

本轮硬件证据根目录：
`/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261024-001`。
每个 run 的 execution_sources 保留测试时实际工作区身份；整理提交不会重写这些身份。
结果为 `status=pass`、driver 返回 0。none oracle 到 step24；Ours source8、显式 generation1
step4→8、latest step8→16 与 step16→24 均通过；发布中断在 raw generation27 提交后、文件
目录发布前注入，故障 worker 返回预期的 1 且 owner 安全关闭；fresh process 对账
`recovered=[1]`、`rejected=[]`，从 step4 恢复、续训并保存 raw generation28 到 step8，owner
关闭通过。五次比较检查四 rank 的模型/Adam SHA256 与 geometry、controls 精确相同，loss
容差 `rtol=1e-5, atol=1e-6`。这不是性能测试。
提交候选移除了未启用的研究 hook，并整合上游 D2 更新，因而不能宣称其 Git tree 与该
硬件执行树完全一致。C 实现及 ABI3 二进制相同；整理后的软件回归单独列出。

## 当前限制

FULL 定向正确性闭环不等于完整性能发布。保存同步阻塞训练；路径为 ACL D2H/H2D + Host
socket + SPDK，不是直接 NPU–NVMe P2P。Host/HBM 预算是容量估计，不是实测峰值。

完整四方法/重复计时矩阵、ByteCheckpoint 剩余链、真实 D1 源镜像迁移、跨进程
rank disconnect/owner 故障矩阵仍延期；不增加掉电安全声明。2026-09-18 的 ABI2
D2 flush 故障与 TP2/TP4 fixture 已通过，详见
[D2 验证记录](docs/plans/D2_FINAL_VALIDATION_20260918.md)，不能冒充本轮 ABI3 故障验收。
增量、guarded live、R0、Top-K 等继续作为独立研究保留，不要求在本轮落地。

当前流程和问题到方案的映射见 [FULL_CHECKPOINT_FLOW.md](FULL_CHECKPOINT_FLOW.md)；
实施前审查见 [SOURCE_IMPLEMENTATION_AUDIT.md](SOURCE_IMPLEMENTATION_AUDIT.md)；
命令见 [EXECUTION_ENVIRONMENT_AND_COMMANDS.md](EXECUTION_ENVIRONMENT_AND_COMMANDS.md)。
内部依赖已建立的契约，保留必要的 DMA 借用、request 引用、quarantine、reader pin
和恢复 ready 规则，不以重复检查或静默 fallback 替代所有权设计。
