# 固定驱动的用户环境升级与开发执行计划

> 自 2026-09-11 起，后续实施顺序、契约与阶段状态统一以
> [长期开发与验收计划](LONG_TERM_DEVELOPMENT_PLAN.md) 为准。
> 本文保留历史事实及细分协议/研究假设；下文旧“唯一计划”和排期不覆盖长期计划。
> 本机候选环境和 Qwen 训练入口现已存在；本页“未安装/无设备”是早先会话记录。
> 本页 E0–E3 与长期计划的 E0/EN/E1/E2 编号不同，映射与当前状态见长期计划 §0.7/§8.4。

制定及首次执行：2026-09-11。此计划承接已确认的用户决策；阶段完成状态以本页及
`results/environment-upgrade-20260911/` 为准，不以旧 PPT 实验计划的执行状态推定。

## 固定决策

- 驱动、固件、内核模块、系统 CANN、全局链接、其他用户配置和设备权限不变。
- 本用户的新环境使用 `/models/npu_nvme_exp/user7-stack/` 私有目录；Python、CANN、
  代码 checkout、构建和缓存分别存放。`/home` 空间不足，不默认在那里安装。
- 使用 MindSpore/MindFormers；从支持 Qwen3-8B 全参数训练的发行版反查 CANN 与
  驱动兼容范围。不因最新推荐组合要求较新驱动就否定所有较早组合。
- 官方资料没有覆盖、也没有明确冲突时，允许最小隔离验证；明确不兼容或安装器拒绝时
  不强制绕过。官方支持状态和本机实测结果分别记录。
- Qwen3-8B 作为多卡完整训练基线：先四卡 TP=4/DP=1/PP=1，确认容量确实不足后才
  考虑八卡。两卡先做预算评估，不增加首轮正式矩阵；不在单卡采用 Host 卸载替代。
- 环境切换必须通过 Qwen3-8B 全参数训练、完整状态持久化、新进程恢复和续训。
  配置解析、推理或权重 I/O 通过均不构成环境升级完成。
- 若固定驱动下没有通过的组合，旧环境继续 FULL 主线；Qwen3 记录阻塞，不暂停全部研发。

## 阶段与验收

### E0：固化环境与兼容性核查

核对实际加载驱动、固件、910B3、OS/架构、设备可见性、Python 包锁、CANN 各组件、
ACL/HCCL/项目库解析路径、SPDK/DPDK 构建和旧 fixture。分别保留发行目录版本
`8.0.RC3` 与 runtime 内部版本 `7.5.0.1.129`；`compatible_version_fw` 是固件字段，
不能当作 CANN 支持范围。

建立逐发行候选表：Qwen3 全参数训练支持、MindFormers、MindSpore、Python、CANN、
toolkit/算子包、最低驱动、推荐驱动、固件和 910B 支持。查官方发行文档、CANN 兼容
说明和安装包元数据，记录来源、日期、包哈希。先明确兼容者，再资料不完整者；同等
条件优先较低 CANN 所在的匹配维护组合。资料不可访问为 `blocked`，不是不兼容。

安装空间包括下载、解包、安装、构建、缓存及 20% 余量。旧安装和模型不自动清理。
安装前检查非 root 自定义目录行为，禁止系统路径变更及强制跳过版本检查。

### E1：私有试装与基础回归

候选使用独立 Python、版本目录和 checkout；重新构建 ACL/SPDK 桥。启动器显式选择
旧或候选环境，清除继承的 Python/CANN 搜索路径和 shell hooks；检查依赖缺失、stub
和混用其他 CANN。保留 Python 包清单与 CANN/驱动/库的环境标识。

按序验证 ACL 初始化、内存、四条 Host/NPU I/O、异步 event、两/四卡 HCCL、元数据
回退、错误传播、超时，以及 GPT-2 完整训练状态保存→退出→新进程恢复→续训。
设备需空闲；沿用已核实的裸盘测试区域、单 SPDK owner 和现有权限，不格式化或重绑定。

### E2：Qwen3-8B 四卡完整恢复

固定完整预训练权重 revision/转换哈希、真实文本/tokenizer/batches、BF16 模型、
完整 AdamW、global batch=1、128 个有效训练位置、dropout=0、seed=41。
采用官方支持的实际状态分片，枚举参数和 Adam 槽位，不能把数据并行当作容量分片。

测每 rank 唯一状态、复制状态、梯度、激活、工作区和快照峰值，保留约 15% HBM 余量。
四卡不足时先排查分片未生效，再考虑八卡；协调器不能额外占用第九张卡。

Native 与 Ours 均执行：预热 5 步并固化 fixture，更新 3 步保存一代；模型、Adam m/v、
实际 master weights、step、scheduler、loss scale、RNG、数据游标、别名和分片策略齐全。
所有 rank 达到 PERSISTED_READY 才发布 COMMIT。生成连续 3 步 oracle 后退出全部源
进程；新进程恢复所有分片、逐字段字节校验、新建 HCCL 组，再续训 3 步。

恢复使用相同卡数/分片，不在首轮实现跨卡数重分片。默认续训容差为 rtol=1e-5、
atol=1e-6；先做无故障跨进程对照。对照失败或恢复偏差未解释时停止验收，不临时放宽。
大分片 manifest 使用版本化 payload 与根引用/hash，不突破固定 400 KiB 元数据槽；
保持旧格式可读，不隐式迁移旧盘。

### E3：切换和回退

顺序执行旧→新→旧 GPT-2 回归，记录实际加载库与相同 fixture 结果；核对系统文件、
链接和默认配置未变。E1/E2/E3 全部通过，才实现并启用经过验收记录校验的项目默认
环境切换。当前启动器仅允许显式候选运行，默认仍为 old。

### A–G：环境阶段之后的研发顺序

1. **A 基线整合**：核验远端最新 master，审查恢复分支及当前未提交修复；保留原工作区，
   建立单独集成 checkout，记录提交依赖和 CPU 测试结果。E1/E2 只提前整合验收必需部分。
2. **B 证据修正**：主实验与 smoke/嵌套负载分开，按实验身份统计；degraded 不算 pass；
   汇总只生成本轮证据支持的结论，不把最新一条异构配置的吞吐当作全矩阵代表。
3. **C XL live 正确性**：旧/新环境同 fixture 比较 serial/frozen/live，定位状态、RNG、
   游标、优化器完成顺序、快照与更新 fence。三 seeds 和慢盘通过原严格 oracle 后，
   才开放 XL live 性能；未解决则保留 frozen 支持范围。
4. **D 性能**：P2 精确分层、P3 异步对照、P4 训练干扰、P5 内存按序；计入忙轮询 CPU。
   P2 残差目标≤10%，未闭合不画精确百分比。trace 使用私有 instance，不操作全局 tracefs。
5. **E 现代模型与恢复比较**：Qwen3-1.7B 单卡完整恢复，Qwen3-8B 多卡基线；复用共享
   runner/state bridge，Native/ByteCheckpoint Host-adapted/Ours 同环境、fixture、step、
   状态范围和分片方案比较。只有通过的模型进入对应正式矩阵。
6. **F 多卡可靠性与协调器**：先分解 Host/IPC 复制、队列及串行化，再做单因素优化。
   覆盖 rank 部分写入退出、提交前协调器退出、提交后源退出、manifest 损坏、慢盘、
   缓冲耗尽和两次物理槽位回绕。最新全局 manifest 的恢复不冒充多历史代际恢复。
7. **G 增量候选**：CPU 轨迹回放比较分类预算、max-age、残差及全块量化；先无损替换块
   R0，再有损。候选通过后才接真正 FULL+Delta、持久化 ACK、回绕、损坏和新进程续训。

暂不实现多 Reactor；只有测量证明单 Reactor 饱和并限制设备吞吐，才另立工作包。

### 固定正式实验规格

- P3：serial/queue/async × chunk 1/4/16 MiB × depth 1/4 × 正常/1 s 持久化延迟；
  每组 10 warmup+30 formal；明确延迟注入位置，不称作 SSD 服务时间。
- P4：none/serial/frozen_async，live 限独立验收通过模型；三 seeds、每组至少 30 个
  checkpoint，主 interval=10、压力 interval=1；显式记录 admission wait 和快照成本。
- 恢复：每方法每模型三 seeds 正确性；seed 41 各五个独立计时进程，另做验证。
  state-ready 不含构造/实验额外 hash；多卡以最慢 rank 就绪及必要全局同步为结束。
- 增量观察：三 seeds、120 步，早中晚各 20 个观测，adjacent/persisted-reference
  分开；model/Adam-m/Adam-v 分类，PMU 未对齐共同时间轴时只报告资源均值。
- 有损候选：计入所有状态、对齐和周期 FULL 后写量<20%，NRMSE≤5e-3、恢复 loss
  偏差≤1%、无超龄块；未通过转全块量化或 FULL 优化，保留负结果。

## 已实现的入口与使用方法

从仓库根目录执行（不修改默认 shell）：

```bash
python scripts/run_user_environment.py --profile old --inspect
python scripts/run_user_environment.py --profile old -- python \
  experiments/benchmarks/environment_upgrade_inventory.py --check-network \
  --models /models/Qwen3-8B --output results/<new-run>/inventory.json
python experiments/benchmarks/summarize_p1_p9.py \
  --root results/formal-20260909-full \
  --output results/<new-run>/P1_P9_REVIEWED.md \
  --json-output results/<new-run>/evidence_summary.json
```

`config/user_environments.json` 保存 old 和 candidate 的独立路径；candidate 当前为 null。
候选完成实际安装后，填写其 python/toolkit/version_root/set_env/library 绝对路径，
均必须解析到 private_root 内；set_env 仅记录哈希，启动器不执行可能拷贝 custom op
的厂商脚本。旧 MindSpore 2.5 的动态符号加载依赖 `latest` 入口，因此 old 沿用已有
入口并校验实际库文件属于明确的 `8.0.RC3` version_root，不修改系统链接。
实际 CANN/Python/框架回归仍必须通过，`--inspect` 只证明静态路径与动态库依赖检查。

`NPU_NVME_LIBRARY_PATH` 指定项目库，必须绝对路径；默认沿用当前 build_out。
证据包增加 environment_id、experiment_id、run_role、parent_run_id；旧证据通过原有
config 和目录父子关系读取，不回写旧原始数据。P2 分层不可观测时退出码 2，编排器
记为 degraded 并停止该批次，不继续以“全通过”运行。

## 本次完成状态与限制

- 已实现：环境盘点、明确环境启动入口、库路径覆盖与校验、分阶段模型探测状态、
  证据身份和父子归属、P1/P2 报表修正、P2 私有 tracefs 与 degraded 传播。
- 未完成：候选组合核实/安装、硬件回归、Qwen3 模型工厂与分片状态桥、完整训练恢复、
  默认环境晋升、远端主线整合、XL live 修复及后续新硬件矩阵。
- 当前会话没有 NPU/UIO 设备，官方网络不可达，firmware/Qwen 配置读取被拒绝；
  允许写入的目录仅工作区和 /tmp，尚不能安装到选定的 /models 私有目录。
- 因此兼容性为 `not_determined`，不是 `incompatible`；不会选择未经核实的软件包。
  必须在可访问上述资源的目标机执行上下文中继续，驱动不可修改的约束继续有效。

预计工作量：E0 1–3 日，E1 2–4 日，E2/E3 4–8 日；基线与 XL 诊断 3–5 日，性能
及恢复矩阵 5–8 日，多卡与增量后续 2–4 周。编译、设备排队和外部访问等待另计。
