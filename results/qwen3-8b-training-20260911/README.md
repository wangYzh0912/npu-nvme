# Qwen3-8B 四卡短程训练验收（2026-09-11）

结论：真实 Qwen3-8B 全参数短程训练及检查点结构检查通过；跨进程恢复、续训 oracle 对比尚未运行，不能视为完整恢复验收通过。

## 执行环境与结果

- 原始运行目录：`/models/npu_nvme_exp/user7-stack/qwen3-8b-full-restart-20260911-153007/`。
- MindSpore 2.7.1、MindFormers 1.7.0、本用户候选 CANN 8.3.RC1；系统驱动未修改。
- Ascend 910B3，设备 0–3，TP=4、DP=1、PP=1；BF16 计算，FP32 参数及 Adam 状态。
- 加载本地 `/models/Qwen3-8B` 的五个预训练权重分片，模型约 81.9 亿参数。
- 固定本地文本、序列长度 128、batch size 1；共 8 步（5 步预热、3 步正式测试）。此结果不代表生产规模或收敛质量验收。
- 四个 rank 的逐步 loss 完全一致，从 `0.7610580921173096` 降到 `0.5963893532752991`，无溢出跳步。
- `msrun --join=True` 返回 0，所有训练进程已退出。
- 四份 safetensors 检查点，每份 24,575,090,784 字节，总计 98,300,363,136 字节。每个 rank 包含 291 个模型参数张量及对应的 291 对 Adam m/v，`global_step=8`、`step_num=8`、`loss_scale=1`。

## 报告说明

- [acceptance.json](acceptance.json)：训练退出后对全部 rank 进行的汇总及检查点结构核查。
- `rank_0/` 至 `rank_3/`：原始逐 rank 报告，包含逐步 loss、环境标识、配置/数据摘要及本地检查点路径。
- 原始逐 rank 报告的 `checkpoint=not_run` 表示当时尚未由独立核查程序验收；后续结构检查结果以汇总报告为准。保留原始报告，不回写历史状态。
- 结构核查包括文件长度、模型与 Adam 状态覆盖、形状、Adam 精度及 step；不等价于张量数值恢复验证。RNG、数据游标及 fresh-process 恢复正确性尚未验收。
- JSON 中的绝对路径指实验机上的原始文件，不是远端仓库内可下载的检查点。

本提交仅发布报告和说明；模型、约 98.3 GB 检查点、编译产物及训练代码不包含在本报告提交中。训练入口修复涉及候选 Python 的 PATH、非 legacy Qwen3 上下文，以及分布式权重加载所需的 `sink_mode=True, sink_size=1`。
