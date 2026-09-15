# 本地/远端开发进展与长期规划可行性审查

审查日期：2026-09-11。结论：临时规划的 FULL 优先、模块拆分、异步收敛、Native→Ours Qwen 恢复和精确增量路线有条件可行；已修订并固化为 [长期开发与验收计划](LONG_TERM_DEVELOPMENT_PLAN.md) v1.3。本报告记录一次审查，后续阶段状态只维护在长期计划 §8.4/§9。

## 1. 本地与远端不是同一开发起点

本轮成功执行 `git fetch origin --prune`。完整提交号、采集时刻、修改前状态、逐文件 hash 与其他工作树状态见 [workspace_snapshot.json](../results/development-plan-review-20260911/workspace_snapshot.json)。

| 对象 | 核查结果 | 后续处理 |
|---|---|---|
| origin/master | `5b164b6`，整理了当前运行代码、baseline 和有效实验；本地 master=`f3c0861`，落后 12 提交 | 从核验后的主线建立独立实施工作树，不从旧实验分支覆盖主线 |
| 当前 HEAD / upstream | 均为 `ef21f92`，分支 `exp/ppt-evidence-20260829` | 仅提交历史同步；仍有大量本地修改未推送 |
| 主线/实验分支关系 | 共同祖先 `ef01af9`；独有提交 37/2 | 证据分支两个独有提交为 `83e8e9c` 的环境文档、`ef21f92` 的六份 Qwen 报告/说明 |
| 已跟踪本地修改 | 文档修订前 11 文件，401 行新增、85 行删除 | 包含 README、P1/P2 和报告归属、C 轮询/PA IOVA、binding 库选择、G1/G2 harness；分别审查迁入 |
| 未跟踪本地成果 | 环境 profile/启动器/inventory、Qwen 训练/检查器/launcher/测试、结果与研究文档 | 代码和报告不是同一发布状态；按实际文件 hash 标识，不能把 ef21f92 当训练源码版本 |
| 其他开发工作树 | baseline-repro=`0fdcb48`、io-path-v1=`f3c0861`，本轮均干净 | 其后续功能和证据已形成主线来源，不重复实现；未修改这些工作树 |

当前工作树相对主线的 `direct_checkpoint.py` 与 `npu_nvme.c` 共表现为 95 行增、901 行删的差异，主要反映实验分支缺少后续实现，不能理解为一份可直接合入的“小修复”。本轮仅 fetch 远端引用，没有 merge、reset、移动本地 master 或 push。

## 2. 已有开发能力和证据边界

主线的 [当前结果说明](https://github.com/wangYzh0912/npu-nvme/blob/5b164b69ba8b4bdeb1f2eb7c454da0e3e9dc9e98/results/README.md) 保留 IO1–IO4、INC1–INC3、baseline 和完整态恢复记录。它们是后续工作的输入，不是新规划 G/H 门禁已通过的证明。

| 方向 | 已有进展 | 仍需解决 |
|---|---|---|
| FULL/异步 | 主线已有写 request API、frozen/live 实验入口、update fence、单 Reactor 和 DP/HCCL harness | CRC 写与异步读迁移、引用/关闭/隔离、库级 ready、统一提交和有界传输仍需按新合同验收 |
| 基线恢复 | Native、Host-adapted ByteCheckpoint、Ours 有历史独立完整态恢复/三步续训记录 | 历史 mean 7.0999/15.4111/4.1823 s 使用不同物理盘和不同状态字节；BASE timing 校验开关也不一致，不能直接成为新公平排名 |
| live | GPT-2 有历史成功记录；主线说明明确保留 XL live 严格续训失败 | 按 H05 修复并复测，不能将宽松容差诊断改写为严格通过 |
| 增量 | 有 R0/S2、状态分类观察及实验机制；INC2 汇总的 ±3% 等效性字段为 false | 精确 ACK/manifest/父链/回绕仍需加固；能量集中不保证各类 Adam 状态可稀疏保存，不承诺增量必有系统收益 |
| 新架构入口 | 现有 repro adapter 可复用 | BASE 尚无根 train.py、目标 npu_nvme 包及本规划 gate runner；不能把工具名字写进计划就视为已实现 |

本轮源码抽查确认了规划中的关键接入问题：`repro/cli.py` 聚合 source 状态会掩盖 restore 失败，`continue-on-failure` 可使失败 suite 返回 0；`chunk_helpers.py` 恢复分配按 shape/dtype，DMA 长度另取 size；现有多 rank harness 为 DP 且先保留全量 rank raw 字段，不能直接证明 TP4 的分片或有界内存。上述是 E2 源码确认，本轮未注入 DMA 故障或触发可达越界。

## 3. 执行环境与 Qwen 的实际进展

本轮 `npu-smi info` 成功：8 张 910B3，健康 OK，采集时未见运行进程；83 盘仍绑定 `uio_pci_generic`，84 盘为 `nvme`，`/models` 为 XFS。设备和网络可访问，历史环境报告的“无设备/网络不可达”只是旧会话事实。

old/candidate 的环境 `--inspect` 均通过，读取了 Python 包清单身份、CANN 组件/项目库 hash 和依赖解析路径。old 为 Python 3.9.25/CANN 8.0.RC3；candidate 为 Python 3.11.4/CANN 8.3.RC1，默认仍 old。当前候选 environment_id 与四 rank 历史报告一致，驱动版本文件 hash 与旧盘点相同。这不代替训练进程实际加载库记录、官方兼容性证据或双环境硬件回归。

Qwen 原始目录 `/models/npu_nvme_exp/user7-stack/qwen3-8b-full-restart-20260911-153007/` 仍在。本轮五份 JSON 与远端固定提交内容一致。受限 checkpoint 元数据和文件头通过 sudo 只读核查；未改写原文件，也未运行原结构汇总 CLI（该 CLI 会写 acceptance.json）。结果见 [qwen_readonly_review.json](../results/development-plan-review-20260911/qwen_readonly_review.json)。

- 历史训练：TP4/DP1/PP1、batch1、seq128、BF16 计算、FP32 参数/Adam；8 步 loss 四 rank 一致，无 overflow，首尾为 0.7610580921/0.5963893533。
- 本轮结构核查：四文件各 24,575,090,784 bytes，合计 98,300,363,136 bytes；每 rank 291 个模型张量和 291 对 m/v，step/global_step=8。五份原权重分片总长 16,381,516,776 bytes，模型配置摘要匹配。
- 头中可见 step/epoch/loss scale 等标量，未见独立 RNG/data cursor 字段；是否必需及如何补齐由 EN 的状态合同定义，不补造历史状态。
- 本轮未遍历全部 payload、未计算大文件/逐张量摘要、未执行恢复或续训；文件长度和头部 hash 不能代替上述验证。

本地训练源码、环境配置和 launcher 已经存在，所以 QW-01 应改为整理、参数化、锁定和选择性提交。两个 launcher 写死旧工作树路径，迁入新工作树后直接调用会跑错源码。本地训练把 LR total_steps、保存间隔和 stop_step 绑定，source oracle 实验必须拆开训练 horizon、保存点和停止点，不能只增加 `--steps` 就假定前八步轨迹不变。现有 checker 也不是完整容器校验器，QW-03 仍需偏移/重叠/长度/身份和载荷验证。

## 4. 可行性修订及长期计划的固定边界

| 原规划需要调整的部分 | 已落入 v1.3 的修订 |
|---|---|
| 仅基于远端报告判断 Qwen 材料不存在 | 分开本地未提交代码、原始文件只读验证、历史训练、尚未运行的恢复 |
| 把当前实验工作树当 BASE | 固定源码审查提交，明确从主线独立实施并逐组迁入本地成果 |
| 外部架构报告缺失 | 检索未找到的来源列为未核实；计划自包含，A0 从源码补反例，不凭摘要标 E1 |
| 源报告“本次 84 passed”无本机日志 | 干净 BASE 重跑并保存日志；本地新增三模块另跑；历史 6.91s 与本轮 2.76s 分开 |
| D2-rank 的真实依赖比 DAG 多 | 增加 B2→D2-rank、EN→D2-rank；D2-format 用明确子集，不循环要求未来 F1 真机完成 |
| B2 与 C2 的 ABI 删除时点矛盾 | B2 兼容 wrapper，C2a 退役独立同步 bulk，C2b 升 ABI 并删除旧阻塞 batch API |
| 单卡与 TP4/DP4 混用风险 | TP4 schema、owner、全 rank commit/ready 独立验收；10 GiB/rank 默认必须容量预检拒绝 |
| 工期与旧研究排期冲突 | 原人日表合计 49–84，单人约 10–17 个五日工作周；TP 协议、排队等另估，阶段入口重估 |
| 多份“唯一计划”并存 | 长期计划为唯一实施/状态入口；旧计划保留细分协议和历史事实，冲突按新计划；旧环境 E 编号显式映射 |

计划的部署只固定任务和验收合同，不执行全部研发工作。默认主链为 A0→A→B→D1→C1/B2→C2；Native Qwen E0/EN 可独立推进，Ours TP4 需要 C2+D2-rank+EN，live 与无损增量按各自门禁开放。裸盘测试区域/峰值容量尚需正式 preflight；本轮没有格式化、绑定变更、驱动安装或设备写入。

## 5. 本轮验证与未覆盖范围

| 检查 | 本轮结果 | 范围 |
|---|---|---|
| 干净 BASE 可移植子集 | 84 passed in 2.76s，退出 0 | 排除五模块；使用旧环境 Python，仅已有主机逻辑用例，无 C 构建或训练 |
| 本地环境/证据/Qwen 三模块 | 23 passed in 0.25s，退出 0 | PYTHONPATH=.:python；包含少量静态源码断言，不能证明模型或存储行为 |
| 首轮本地测试 | 缺 PYTHONPATH，收集失败，退出 2 | 保留原日志；按已有环境文档补路径后复跑通过，无运行代码改动 |
| 双环境/硬件盘点 | 两个 inspect、npu-smi、driver/mount 读取退出 0 | 路径/版本和设备可见性，未做设备 I/O |
| Qwen 原文件核对 | 报告一致、四 checkpoint 结构/step/长度可复查 | 非完整载荷校验，不是恢复验收 |

原始命令、输出、源码/文件摘要见 [证据目录](../results/development-plan-review-20260911/README.md)。未运行被排除的五个主线测试模块、目标机 Python 全集、C_IMPL、C 构建、硬件故障注入、Qwen 训练/恢复或正式性能矩阵；不以本轮检查宣布 A/EN 等阶段完成。
