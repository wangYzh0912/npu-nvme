# WP2 正式证据包（2026-08-25）

本目录是工作包二首轮执行的 curated evidence bundle。原始运行目录仍保留在本机
`experiments/output/wp2/`，本包只纳入结构化结果、配置、环境快照、样本时间线和
必要的失败复现，便于审阅和 Git 追踪。

## 覆盖范围

- `model_13b/`：GPT-2 13B 的 A1、A2、A3、A4、A5、A10 正式样本。
- `model_xl/`：GPT-2 XL 的 A1、A2、A3、A7 关键正式样本，以及 A8 正式协议样本。
- `synthetic/`：A3/A4/A5/A6/A9 门禁样本；`a6_formal/` 是 sync API 与 request-ring/FSM 对照，`a9_formal_slots_*` 是显式 host slot 生命周期结果。
- `failures/`：A4 depth=16 读提交缺陷复现，以及 npu-smi 解析误判复现。

模型 checkpoint-only 样本均带有 `hashes` 或逐参数摘要校验；A7 使用真实训练 cell
和图内 step counter；A8 使用 128 GiB 安全区中的 generation/CRC/active-slot 协议，
不修改 83.0.0 live superblock/metadata，也不触碰 84.0.0。

## 重要边界

13B A5 模型版已正式覆盖 1/4/16 MiB；64/256 KiB 仅有合成正式样本，原因是它们
分别产生约 400k/100k chunks，仍列为待补的极小 chunk 长实验。A6 的 API/控制面
门禁已 PASS，但不等价于模型全路径；A9 的 host slot 生命周期已 PASS，但 HBM
训练快照仍需单独验证，不能与模型 checkpoint 结论混合。

修复后的关键提交：`b36ea16`（npu-smi 进程表解析）、`b6ccb8a`（高 pipeline 深度
读提交重试）、`d2fb8fc`（读写 FSM 单调游标）。

每个证据子目录保留原始 `config.json`、`environment.json`、`result.json`、
`samples.jsonl` 和 `timeline.jsonl`（失败复现目录除外）。完整清单见
`MANIFEST.md`。
