# 长期规划部署的核查证据

日期：2026-09-11。对应 [审查报告](../../docs/DEVELOPMENT_PROGRESS_REVIEW_20260911.md) 和 [长期计划 v1.3](../../docs/LONG_TERM_DEVELOPMENT_PLAN.md)。这是规划审查证据，不是未来 G/H profile 的验收包。

| 文件 | 内容 |
|---|---|
| workspace_snapshot.json | fetch 后分支关系、文档编辑前工作区状态/差异 hash、本地变更文件 hash、其他开发工作树状态 |
| master_cpu_tests.json | 干净 5b164b6 可移植子集命令、环境路径、退出码、原始 stdout/stderr；84 passed |
| local_cpu_tests.json | 本地新增三模块命令与输出；23 passed |
| local_cpu_tests_missing_pythonpath.json | 首轮缺 PYTHONPATH 的收集失败，保留诊断来源 |
| old_environment.json / candidate_environment.json | 环境 inspect 的命令/输出、包清单摘要、组件和库 hash、解析路径 |
| npu_smi.json / raw_driver.json / filesystem_driver.json / models_mount.json | 只读设备与挂载盘点 |
| qwen_readonly_review.json | 固定提交五份 JSON 与本机原文件核对、四 checkpoint 头 hash/标量/长度、模型配置摘要；payload hash 为 null，表示未计算 |
| qwen_readonly_review.py | 本轮只读核查程序副本；调用本地结构检查函数，不执行会回写原报告的 CLI |
| migration_checks.json | 临时→正式规划保留全部 118 个 G/H 编号（含子项）、37 个任务编号；原有运行代码/配置/测试未改动 |
| evidence_manifest.json | 本目录实际产物的大小/SHA-256；不包含 manifest 自身 |

主线 CPU 检查使用一次性独立工作树 `/tmp/npu-nvme-plan-review-20260911`；审查结束后移除，该目录不属于长期依赖。重放时从记录的完整 BASE 提交创建新工作树，保留相同解释器、PYTHONPATH 和五项排除。原日志不因清理工作树失效。

Qwen 原 checkpoint 读取需要 root。普通用户第一次尝试在 rank_0/meta.json 遇到 PermissionError，之后按项目密码文件模板提权，仅以读模式打开文件；密码未保存到结果。重放脚本需在含本地 Qwen 核查器的开发工作树执行，访问同一原始数据目录；它的输出应写到新审查目录。脚本使用 Python 3.11 的候选解释器，无 MindSpore 导入或设备初始化。

本轮未保存大模型/检查点副本，未读取密码内容到日志，未运行训练、裸盘写入、C 构建或硬件故障。inspect 证明路径解析，不等同于实际训练进程全部已加载库的审计。历史报告和本轮核查分开，文件头一致不能当作载荷或恢复正确。
