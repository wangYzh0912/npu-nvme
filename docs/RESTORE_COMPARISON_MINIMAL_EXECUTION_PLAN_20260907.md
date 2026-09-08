# 完整训练状态恢复对比：最小执行方案

> 2026-09-08 状态更新：本地已缓存的 `origin/codex/baseline-gpt2-repro@0fdcb48`
> 包含三方案计时与独立验证结果，原 attach 阻塞已解决。本文保留原始方案，
> 最新执行状态见[开发进度](DEVELOPMENT_STATUS_AND_ROADMAP_20260908.md)及
> [恢复结果快照](development-evidence/20260908/FULL_STATE_RECOVERY_RESULTS_20260907.md)。

日期：2026-09-07。状态：方案规划，尚未实施计时改造或运行实验；不修改 PPT。

## 一、目标与范围

用最少的新增测试回答：在同一 NPU 上恢复同等规模的完整 GPT-2 训练状态，本系统相对于框架原生和近期论文适配方案，恢复延迟如何，主要差异发生在哪里？

本轮只研究完整状态恢复，不同时扩展增量恢复、多卡恢复、故障注入、流水参数扫描或长时间压测。结果应能形成一张三方案恢复延迟对比图；如果分段可可靠获取，再补一张小型开销分解图。

主指标不是“加载函数返回”，而是“模型参数、优化器与控制状态恢复到可继续训练的状态”。恢复成功与恢复速度分别验证。

## 二、已核实的现状

远端最新相关提交为 `origin/codex/baseline-gpt2-repro` 的 `b0548a5d426f6952f848623143171a8e253306c2`。最新抓取未发现新增恢复计时结果。

| 方案 | 当前恢复路径 | 已有记录 | 本轮取舍 |
|---|---|---|---|
| 本系统 `ours` | `DirectCheckpoint.load_state`，直接恢复目标模型/优化器 | 已有其他实验的恢复验证与权重读回；统一 runner 的 SPDK attach 曾失败 | 必测 |
| `mindspore_native_save` | 真正 `ms.load_checkpoint`，再经公共状态桥装回 | 独立进程 byte-exact / 3 步续训通过，无耗时 | 必测 |
| `bytecheckpoint_host` | 上游 `DDPLoadPlanner` / `load_state_dict` / extra-state 流程，再经共享内存和 NPU 状态桥 | 同上，无耗时 | 必测，作为论文适配参照 |
| `fastpersist_host` | `torch.load(..., map_location="cpu", weights_only=True)`，再桥接 | 同上，无耗时 | 可选，不进入最小矩阵 |
| `datastates_acl` / `pccheck_acl` | 相同 `load_raw_snapshot` 恢复路径 | 同上，无耗时 | 暂不测，不能增加独立机制信息 |

已有成功标志不能直接转化为恢复性能数据。现有 `restore.json` 未记录统一起止时间；日志也不足以重建可靠的完整恢复耗时。

旧 WP1 中的权重读回可以展示规模趋势，但不包含同等范围的 Adam/控制状态，不与本轮完整恢复做直接速度比。已有原生保存 API 的 1.70 s、保存入口总计 6.10 s 都不是恢复时间。

## 三、最小实验矩阵与预算

### 3.1 固定配置

| 项目 | 约定 |
|---|---|
| 模型 | 既有 GPT-2 配置，不新增模型 |
| 初始状态/数据 | 同一 prepare fixture、同一 batches 文件及其 SHA256 |
| 训练配置 | seed 41，batch 1，输入 129 tokens，dropout 0 |
| 保存内容 | 模型参数、Adam m/v、已有 schema 中的训练步数、数据游标、loss scale/RNG 等实际控制字段；约 1.485 GB，以 manifest 精确字节数为准 |
| 源状态模式 | 冻结保护，保存完成后源进程退出 |
| NPU/绑定 | 沿用已核验设备与 CPU/NUMA 配置；各方法串行执行 |
| 本系统 I/O | chunk 4 MiB、depth 4；不做本轮参数消融 |
| 被测方法 | Ours / MindSpore Native / ByteCheckpoint Host-adapted |
| 性能样本 | 每方法 5 个独立恢复进程，共 15 次 |
| 正确性样本 | 每方法额外 1 个独立验证进程，续训 3 步，共 3 次 |

“同一 seed”不是同一 fixture 的充分证明。核对初始状态和数据文件哈希；不同训练运行产生的末态允许存在已解释的数值非确定性，但状态类别、shape/dtype、字节规模和训练步数必须一致，不能把明显不同工作负载混为同配结果。

### 3.2 检查点复用，避免不必要训练

按以下顺序决策：

1. 核对真实文件/裸盘代际仍在，不能只看 Git 中的 JSON 记录。
2. 核对模型配置、fixture/batches 哈希、schema、步数、提交状态及每方法的 source oracle。
3. 若三种方法已有符合条件的检查点，直接复用，新增训练为 0。
4. 若文件已删除或范围不同，复用同一 prepare 初始状态，每方法训练 3 步、保存 1 代，并生成之后 3 步的无故障 oracle。无需每方法再训练 30 步或保存 10 代。
5. 若只有一个方法缺失，只有在能重建相同 fixture 和目标步数时才单独补齐；否则三方法统一生成上述小检查点。

第 4 种情况下，共 9 个源训练正式 step，外加生成 oracle 的续训。既有 prepare 含 5 步预热；若已有有效 fixture，不必重新执行 prepare。图编译成本可能主导进程运行时间，不能以 step 数承诺固定分钟数。

## 四、检查点与环境的前置检查

### 4.1 建立不可变的输入清单

新增 `checkpoint_inventory.json`，每方法至少记录：

- 实际 runner HEAD、核心库版本、未提交代码差异，以及上游 lock 文件。
- adapter 名称和适配层级，禁止把 Host-adapted 标成 GPU 原版。
- checkpoint 绝对路径或裸盘设备/代际/槽位标识。
- `generation`、`step`、`state_bytes`、schema 摘要、fixture/batches 哈希。
- 模型、优化器及控制字段清单，参数别名/重复状态处理约定。
- 文件存在性、大小和已提交状态，source oracle 路径。
- 如果某信息缺失，显式标记，不从文件名推断已验证状态。

本系统当前 `restore(generation, destination)` 实际向 `load_state` 传入目标 step。需核对请求的 generation 与实际选中的 step/元数据一致；不能请求一代却恢复到另一个最近代际。

### 4.2 设备条件

先用 `findmnt`、`lsblk`、`lspci`、设备序列号及现有配置核对端点。`same_physical_storage_verified` 当前为 false，不能默认 `/models` 与裸盘位于同一 NVMe。

优先使用相同物理介质、明确隔离的测试空间、相同后台负载。若安全条件下不能同盘，先保留恢复原始结果，标题写“当前存储配置下的恢复参考”，不计算严格的系统速度比。

本轮不自动格式化、重新分区、覆盖旧槽、重绑定驱动或清空全系统缓存。涉及设备接管与裸盘写入的动作应在准确核对设备和测试区后另获授权；配置里的历史 `raw_test_authorized=true` 不代表可以扩大写入范围。

### 4.3 本系统阻塞项

统一 runner 曾在 `0000:83:00.0` 的 `uio_pci_generic` / PA IOVA 路径 attach 失败。现有 `ours.preflight` 主要检查授权字段和库文件存在，并不证明 attach 成功。

因此需分别记录：静态 preflight 结果、运行时初始化结果、实际读取目标代际结果。若 attach 仍失败，停止本系统计时，不回退为文件路径。可继续完成另外两项和计时改造，但不能宣称三方实验已完成。

## 五、统一计时定义

### 5.1 主指标：状态就绪恢复延迟

```text
构造模型、优化器、目标状态容器；初始化必要 adapter/worker
                    ↓
NPU synchronize；记录 restore_begin
                    ↓
选择完整代际 → 读取元数据 → 读盘 / 反序列化
                    ↓
必要的格式转换 / IPC / Host 桥接 → 参数与优化器装回 NPU
                    ↓
恢复步数、游标及控制状态；NPU synchronize
                    ↓
记录 state_ready
```

主指标 `T_restore = state_ready - restore_begin`。计时使用单进程 `time.monotonic_ns()`；父进程完整墙钟区间是最终比较依据，不跨进程混用不同来源时间戳。

该指标明确以模型和恢复服务已初始化为前提，不是从进程启动到训练恢复的完整 RTO。adapter/worker 初始化耗时单独记录为 `setup_ms`，不可遗漏后又宣称冷启动收益。若某方法存在惰性初始化，应在所有方法一致的初始化约定下处理，不能只把最慢方法的启动成本移动出去。

### 5.2 最小分段

| 字段 | 含义 |
|---|---|
| `restore_total_ms` | 公共父进程 restore_begin → state_ready，主指标 |
| `read_decode_ms` | 元数据/数据读取和反序列化；能分开再细分 |
| `bridge_apply_ms` | 必要 IPC、状态转换、H2D、参数绑定、控制状态恢复 |
| `setup_ms` | 模型及 adapter/worker 初始化，单列 |
| `validation_ms` | 额外实验验证，单列，不计入性能主值 |

本系统可将读取直接流水到 HBM，并不天然存在可加总的“纯读盘 + 纯 H2D”串行阶段。若底层尚无分段事件，本轮只给 `direct_restore_ms` 与总值，不通过差额虚构分解。异步重叠阶段也不可简单相加成总时间。

ByteCheckpoint 的 worker 时长只作为诊断子段；必须包含父进程中的共享内存处理和最终 NPU 装回后，才能与本系统总值比较。

### 5.3 先不测的指标

首步训练用于验证恢复可用性，不进入最小性能矩阵。首步时延容易混入图编译；只有主对照完成且确有展示需求，再增加独立首步/完整 RTO 测试。不为追求图表数量顺带扩大范围。

## 六、计时代码改造清单

以下是待实现内容，不是声称仓库已有这些参数。

### 6.1 公共 runner 与 CLI

修改 `experiments/baselines/repro/runner.py::restore`：

1. 增加性能模式与验证模式的明确分支，原有默认验证行为不变。
2. 在 `_model` 和 adapter 初始化外层分别记录 setup。
3. 在 `adapter.restore`、`apply_snapshot`/`restore_training_controls` 周围记录公共阶段。
4. state_ready 前调用 `ms.hal.synchronize()`，避免只测异步提交。
5. 为性能模式创建独立结果结构，不把未做验证的性能轮写成 byte-exact 通过。

修改 `cli.py`，建议新增：

- `--restore-mode timing|verify`，默认 verify。
- `--output <独立 JSON 路径>`，禁止覆写已有实验结果。
- `--rep-id <编号>`，便于关联 5 个独立进程。

现有 `restore` 默认写 `<run-dir>/restore.json`。未实现新输出机制前，不能直接循环命令覆盖已有证据。

### 6.2 MindSpore Native

保持实际 `ms.load_checkpoint` 与完整状态装回。分别标记原生 load 和公共状态桥；必要的 shape/dtype/字段检查保留。

当前适配器额外执行 `snapshot.digest()` 和整文件 SHA256。区分算法所需的生产校验与实验重复 oracle：生产路径必需的校验保留并计入；仅实验额外校验放到独立 verify 轮，默认功能不受影响。不能用“总时间减一个平均哈希时间”替代实际隔离测量。

当前加载后 `asnumpy()`、Host 数组及再装回是适配路径的一部分。如保留，则如实计入并标注“原生加载 + 当前 NPU 状态桥”；本轮不为追求更好数字同时重构为另一套高性能加载算法。

### 6.3 ByteCheckpoint Host-adapted

涉及 `adapters/worker_semantic.py` 与 `workers/bytecheckpoint_worker.py`：

- 保留 `DDPLoadPlanner`、`load_state_dict`、extra-state 读取和 future 完成等待。
- worker 返回可选阶段耗时；父进程总计时仍为主指标。
- 保留必要共享内存复制和 NPU 装回开销，不把它们算到计时外。
- 将额外 canonical digest / snapshot digest 的实验验证与性能轮分开；不得静默去掉生产完整性要求。

### 6.4 公共 `apply_snapshot`

当前 `state_bridge.py::apply_snapshot` 为检查目标 dtype/shape，会调用 `destination.value().asnumpy()`，这可能额外把目标参数从 NPU 搬回 Host。

本轮先显式计入 `bridge_apply_ms`，避免假装它只是 H2D；并将这项标为适配开销。若决定改用参数元数据完成 dtype/shape 校验，应在所有使用该桥的方法中一致修改、保留别名和单元素 shape 兼容语义，并重新做一次验证。不要为三方测试扩成“改前/改后”六组矩阵；只锁定一个版本作为正式版本，记录修改。

### 6.5 本系统正确性验证补强

当前 runner 对 adapter 返回 controls 而非 Snapshot 的分支，将 `restored_digest` 直接取自已选检查点摘要。该相等比较本身不是恢复后读取目标状态的独立证据。

因此 verify 轮必须从恢复后的真实模型/优化器与控制状态重新捕获摘要，与 source oracle 比较，不能仅沿用 `byte_exact` 标志作为验收。性能轮不做这一全量捕获，验证轮才做。此要求适用于所有方法，不能只对其中一个方法放宽。

## 七、实际执行顺序

### 阶段 A：输入与环境审计

- 锁定 runner、核心库和上游依赖；记录 git diff，不覆盖用户修改。
- 检查真实检查点文件/代际，建立 inventory。
- 检查存储映射与本系统 attach 前置条件。
- 确认复用还是每方法 3-step/1 代重建。

产出：`environment.json`、`checkpoint_inventory.json`、代码与依赖版本记录。

### 阶段 B：最小计时实现与正确性门槛

- 实现第六节计时和独立输出，不改变原始默认验证流程。
- 三方法各恢复验证一次：实际状态摘要、全部控制字段、3 步续训。
- 复用现有 loss 容差 `rtol=1e-5, atol=1e-6`；数值结果记录完整，不只输出 pass。
- 单个方法失败立即定位；在语义修正前不采其性能值。

产出：每方法独立 `verify.json`，验证错误详情或明确通过结果。

### 阶段 C：15 次恢复计时

- 每方法 5 次，每次独立进程，按固定、可记录的顺序串行。
- 优先按轮次轮换方法，减小温度/后台负载漂移；如果裸盘与文件访问要求不同设备占用状态，按安全可执行的分组顺序测量，并记录顺序，不反复自动重绑定驱动。
- 每次记录设备、文件/代际、缓存策略、setup、恢复总值和可得分段。
- 不使用旧进程中已经装回的模型再次测量，避免退化为仅内存路径。

缓存策略：本轮默认不主动清系统页缓存，明确标注“未主动驱逐缓存”；第一条观测单独保留，后续四条也保留。独立进程并不保证冷读，不能把首条命名为冷缓存。主图可报全部 5 次中位数与范围，附表列首条和后续中位数；如果冷热差异主导结果，必须解释缓存条件，必要时只追加一项缓存敏感性诊断，不隐去首条。

产出：15 个互不覆盖的性能 JSON。

### 阶段 D：汇总与停止

- 校验 15 条样本均指向目标完整代际，状态范围和字节数一致。
- 报告每方法 5 条原始值、中位数、均值、min–max。
- 如果同盘、缓存及状态条件成立，计算 `T_baseline / T_ours`；否则不计算严格加速比。
- 检查分段与父进程总时间的关系；未归类部分明确标出，不能硬凑成 100%。

完成三方法验证和 15 条性能记录后即停止，不追加不同模型、深度、卡数和高频保存矩阵。

只有出现日志证明的异常（例如 attach 错误、设备竞争、错误代际）才重测；异常样本保留并说明。若无异常但波动大，最多每方法追加 2 次，仍只报告趋势，不为了获得优势不断增加样本或删除不利结果。

## 八、命令模板与新旧功能边界

在已确认的远端实验 checkout 中使用原有 MindSpore 环境。新配置和结果目录应独立，不能覆盖 20260906 原始结果。

现有可用 CLI 形式：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=.:python
export LD_LIBRARY_PATH="$PWD/build_out/lib:$PWD/build:$LD_LIBRARY_PATH"

/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli preflight \
  --config <恢复实验配置.json> --adapter <ours或mindspore_native_save或bytecheckpoint_host>
```

注意：preflight 会写结果，并可能触发设备探测；执行前已完成设备权限检查。本系统静态 ready 不替代运行时 attach 检查。

仅当需重建检查点时使用：

```bash
/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli run \
  --config <恢复实验配置.json> --adapter <方法> --steps 3 --checkpoint-every 3
```

该命令会运行现有源训练与恢复验证；正式恢复性能仍由新 timing 模式采集。不能把命令总墙钟作为恢复时间。

**以下命令需要先完成计划中的 CLI 改造，当前不能直接运行：**

```bash
/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli restore \
  --config <恢复实验配置.json> --adapter <方法> --run-dir <该方法源运行目录> \
  --generation latest-committed --restore-mode verify --continue-steps 3 \
  --output <新结果目录>/<方法>/verify.json

/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli restore \
  --config <恢复实验配置.json> --adapter <方法> --run-dir <该方法源运行目录> \
  --generation latest-committed --restore-mode timing --continue-steps 0 \
  --rep-id 1 --output <新结果目录>/<方法>/rep_01.json
```

timing 命令以新进程重复 5 次，编号和输出路径依次改变。输入 source.json 和 checkpoints 保持只读；代际固定后优先使用明确代际编号，避免测试期间 latest 指向改变。

## 九、输出结构与数据格式

建议新目录：`results/restore-comparison-20260907/`；若实际执行日期不同，使用实际日期。

```text
restore-comparison-<date>/
  config.json
  environment.json
  checkpoint_inventory.json
  code_diff.patch
  ours/                  # 另有 mindspore_native_save/、bytecheckpoint_host/
    verify.json
    rep_01.json ... rep_05.json
    stdout/ stderr/
  summary.csv
  summary.json
  findings.md
```

每条性能记录最少包含：

```json
{
  "adapter": "bytecheckpoint_host",
  "mode": "timing",
  "rep_id": 1,
  "generation": 10,
  "step": 35,
  "state_bytes": "实际字节数，输出时用整数",
  "restore_total_ms": "实测值，输出时用数值",
  "setup_ms": "实测值，输出时用数值",
  "read_decode_ms": null,
  "bridge_apply_ms": null,
  "cache_policy": "no_explicit_eviction",
  "same_physical_storage_verified": false,
  "verification_record": "verify.json",
  "status": "completed"
}
```

示例字段不代表已有数据；无法测得的分段保留 null，而不是 0。性能轮的 `completed` 不等于 byte-exact 已验证，通过独立验证记录关联。

## 十、验收标准与汇报形式

### 必须满足

1. 三方法均真实调用指定恢复路径，没有替代或静默回退。
2. 完整状态 schema、目标步数、载荷规模及 fixture 条件可追溯。
3. 每方法至少 1 次实际恢复后状态摘要验证和 3 步续训通过。
4. 每方法 5 次独立进程性能记录，公共端点包含 NPU 同步及控制状态恢复。
5. 额外实验 oracle 与生产必要校验已区分，未通过删去必要语义换取速度。
6. 存储、缓存、适配层级与初始化边界明确披露。
7. 所有失败和异常原始记录保留。

### 最终展示

主图标题建议：“GPT-2 完整状态恢复：原生框架与论文适配对照”。三柱：本系统 / MindSpore Native + 状态桥 / ByteCheckpoint Host-adapted。单位秒，柱高中位数，叠加 5 个原始点或 min–max。

只有在分段互斥且可比时才用堆叠图；否则保持总时间对比，旁边用小表解释桥接路径。不能将本系统重叠流水阶段与对方串行阶段强行按同色堆叠。

允许的结论包括：本系统恢复更快、开销相近、或桥接主导某个适配方案。若本系统没有优势，也如实展示并结合保存性能解释权衡。不给短样本 P99、全规模最优或 GPU 原版算法排名结论。

## 十一、风险与停止条件

| 问题 | 处理 |
|---|---|
| 本系统 attach 失败 | 停止本系统性能采集，给明确阻塞；不自动接管/格式化磁盘 |
| 原检查点文件不在 | 每方法 3-step/1 代统一重建，不扩大到 30-step 全矩阵 |
| 检查点步数/状态范围不一致 | 不拼图；补齐匹配输入 |
| 不能安全同盘 | 报不同配置的参考耗时，不给严格加速比 |
| extra oracle 无法简单分离 | 先报告“含验证恢复入口”及独立诊断，不冒充净恢复性能 |
| ByteCheckpoint 上游环境失效 | 优先修复既有锁定环境；若超出小范围，则先完成本系统/原生两项并显式保留缺项 |
| 性能波动大 | 保留全部值，最多追加 2 次/方法；不无限测试或挑值 |
| 校验失败 | 先修正确性，不对失败路径报告有效恢复吞吐 |

## 十二、可追溯来源

- [统一 runner / adapters / workers](https://github.com/wangYzh0912/npu-nvme/tree/b0548a5d426f6952f848623143171a8e253306c2/experiments/baselines/repro)
- [已完成的适配与原生恢复记录](https://github.com/wangYzh0912/npu-nvme/tree/b0548a5d426f6952f848623143171a8e253306c2/results/baseline-gpt2-repro-20260906)
- [历史 I/O 与完整状态实验总结](https://github.com/wangYzh0912/npu-nvme/blob/f3c086157d2ce65d8b686cd938874b1b3541ec5d/docs/IO_MINIMAL_RESULTS_20260904.md)

本方案的核心缩减：只做三条有代表性的恢复路径，尽量复用检查点，新增 15 次性能恢复和 3 次正确性恢复；没有必要为了恢复对比重新做所有保存实验。
