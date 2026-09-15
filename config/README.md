# 当前配置

训练配置 schema 2 位于 experiments/baselines/repro/configs。方法参数放在 methods.<name>，实际源码路径和提交在运行时固定；禁止历史 project_root/project_commit 混用。

CLEANUP 是清理综合软件 profile；D1 和 EN-native 为独立能力 profile。历史 A/A-subset/B 随归档标签保留。raw_test_region 记录唯一授权裸盘；user_environments 明确区分旧环境和候选环境，不自动晋升。
