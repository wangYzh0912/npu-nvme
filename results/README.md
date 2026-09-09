# 当前实验记录

保留原始 commit、配置与环境信息；本目录不表示整理提交上已经重跑硬件实验。
新运行写入 `experiments/output/<run-id>`。原机器权重、数据和裸盘内容不随 Git 分发，
需按 fixture/数据哈希和运行入口重新准备，不能仅凭结果 JSON 认定可直接恢复原代际。

| 目录 | 保留范围 |
|---|---|
| `io-minimal-20260904` | 最新 IO1–IO4，包含当前 XL live 严格续训失败与代表故障 |
| `incremental-observation-20260904` | INC1 修正 PMU、INC2 正式数据、INC3 三 seed formal_final_v2 |
| `baseline-gpt2-repro-20260906` | 各方法最新完成运行、prepare/preflight 与解释性汇总 |
| `full-state-recovery-net-20260907` | 三方法净恢复时间、独立正确性验证与环境分析 |
| `full-state-recovery-ours-root-20260907` | 最新 Ours 保存和 source oracle |
| `full-state-recovery-20260907` | 仅保留最新恢复结果引用的原 inventory/preflight，不保留旧计时 |

INC 总体能量覆盖不等于所有状态类别都可稀疏保存；图内负载等效性门禁和固定 Top-K
类别覆盖未通过。PMU 时钟未对齐，不推断精确空闲窗口。IO 的成功范围以对应恢复门禁为准，
不能把 XL 宽松容差诊断写成严格通过。多 rank 物理槽轮换不等于保留多个全局 manifest。
不同 SSD 的恢复计时只作端点参考，详见净恢复目录 README。

过往 WP/PPT/R0 campaign、失败重试副本和中间诊断已从 master 移除；原始历史可从
`f3c0861`（I/O/观测）和 `0fdcb48`（baseline/恢复）及其父提交读取。保留的 JSON 中
绝对路径和指向旧诊断的历史文字不自动重写为本机路径，也不伪造新的运行出处。
