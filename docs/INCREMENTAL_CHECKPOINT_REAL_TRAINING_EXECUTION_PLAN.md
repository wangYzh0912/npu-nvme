# 真实训练增量检查点落地与验证计划

> 固化日期：2026-08-27  
> 实施基线：`exp/full-state-correctness` / `ed0e7a5`  
> 目标设备：Ascend 910B3，Huawei ES3000 V6 NVMe `0000:83:00.0`  
> Python 环境：`conda activate ms_2.5`

## 1. 目标、术语与结论边界

本计划的目标是完成以下真实训练闭环，而不只是证明训练过程中能够产生 Delta：

```text
训练 step 完成
  → NPU 图内变化检测与编码
  → generation 独占的 HBM frame slot
  → Reactor/SPDK 裸盘写入
  → A/B 元数据提交
  → 持久化 ACK
  → 推进 P_persisted
  → 进程退出并重新初始化
  → FULL + Delta 顺序恢复
  → 恢复完整训练态并继续训练
```

增量策略统一使用 `S2-R0`、`S2-R1`、`S2-R2` 命名，避免与项目中“多 rank
整改项 R1”混淆：

| 策略 | 语义 | 本阶段作用 |
|---|---|---|
| S2-R0 | 相对最后 ACK 的 `P_persisted`，保存所有变化块的原始 dtype 当前值 | 建立无损 replacement oracle 和端到端正确性 |
| S2-R1 | Top-K/阈值选择，保存 per-block INT8 当前值与 scale | 测量基础数据缩减与量化误差 |
| S2-R2 | R1 + residual feedback + 最大块年龄 | 控制长期饿死和链式误差 |

只有 S2-R0 真实训练端到端门禁通过，才能表述“增量检查点原型可恢复”；只有
S2-R1/R2 同时满足物理写量、训练开销、恢复误差和稳定性门槛，才能表述“基于
NPU 图内筛选的增量写入技术有效”。CPU 合成轨迹、普通文件回放、单次 HBM/SPDK
回环均不能单独支撑上述结论。

## 2. 当前项目基础与缺口

截至实施基线，可直接复用的能力如下：

| 能力 | 现有实现/证据 | 当前边界 |
|---|---|---|
| 完整训练态 FULL | `DirectCheckpoint.save_state/load_state`；C1 单卡和 C2 两 rank | C2 是独立训练进程，尚非 HCCL 梯度同步 |
| 控制状态编码 | `python/training_state.py` | 已覆盖 Python/NumPy/MindSpore RNG、数组和游标 |
| S2 replacement oracle | `python/s2_delta.py` | CPU 语义已通过，尚未连接真实 NPU 图输出 |
| 自描述 S2 frame | `python/delta_protocol.py` v3 | 需要扩展完整状态类别、encoding 和 FULL lineage |
| HBM slot 生命周期 | `python/frame_lifecycle.py`；A9/I7 | 已证明冻结 slot 和背压，未连接 Delta 图输出 |
| FULL/Delta 分区 | `python/disk_layout.py` V2 | 83:00.0 已格式化；新布局不得临时占用未声明 gap |
| 裸盘 ring 与故障选择 | `python/raw_ring.py`；I6 100 代、6.25 次回绕、11 类故障 | 当前 frame 是测试负载，不是完整真实训练 Delta |
| CPU/NPU 算子等价 | I2：norm、Top-K、index、value、scale、INT8 | 输入规模和状态类别仍有限 |
| 真实训练轨迹 | GPT-2/GPT-2 XL 3 seeds × 10；XL 100-step 数值链 | 尚未保存每个采样点的完整可离线重放轨迹 |
| 策略扫描 | 36 组合 CPU 合成 R1/R2 | 不是 MindFormers 权重+Adam 真实轨迹 |

已有结果支持“各子系统具备实现基础”，但真实图输出、完整训练态 Delta、ACK 后参考态
提交和跨进程续训仍未连成同一条链。因此禁止直接恢复旧 `DeltaTrainCell` 线程原型并
将其结果作为正式结论；旧实现只用于定位算子和接口。

## 3. 固定的状态、分块和版本契约

### 3.1 完整状态范围

FULL 和 Delta 使用同一份稳定 manifest，正式结果以完整可继续训练状态为准：

- `model/...`：去重后的模型参数；
- `optimizer/...`：Adam 一阶矩、二阶矩、global step 及其他可变槽位；
- `control/global_step`、`control/loss_scale`；
- Python、NumPy、MindSpore 可获取的 RNG 状态；
- `control/data_cursor`：epoch、sample、batch digest 或下一批数据标识；
- schema version、模型配置摘要、world size、rank、训练 step。

模型参数在 optimizer 中的别名必须按对象身份去重。学习率等不可变超参数进入配置
摘要，不重复作为大状态写出；若框架将其作为可变 Parameter，则作为 singleton state
保存。每次恢复必须严格检查字段集合、dtype、有效 shape、字节数和 checksum，不允许
静默缺字段。

### 3.2 稳定 manifest 与参数内分块

新增统一 manifest 层，不能在训练过程中依赖框架遍历顺序临时编号。每条记录至少包含：

```text
stable_state_id = SHA256(namespace + canonical_name + dtype + shape)
category        = model | adam_m | adam_v | optimizer_other | control
rank_id
parameter_local_block_id
element_offset
effective_elements
storage_dtype
encoding
```

块只允许位于单个参数内部；尾块显式记录 `effective_elements`；小参数独立记录或固定
打包，不能跨参数拼接后再切块。manifest 自身生成 digest，并同时写入 FULL metadata、
Delta frame 和恢复报告。训练期间字段集合、shape 或 dtype 改变必须终止本次链并要求
新 FULL。

### 3.3 S2 与 ACK 不变量

S2-R0 固定采用 replacement 语义：比较 `P_current - P_persisted`，frame 保存选中块的
当前值，恢复时覆盖对应块。以下不变量不得配置关闭：

1. `observe()` 只读取稳定训练快照，不修改 `P_persisted`；
2. 数据完成和 inactive metadata 持久化后，generation 才可见；
3. 只有匹配 generation、base generation、manifest digest 和 frame checksum 的 ACK
   才能推进参考态；
4. 写失败、CRC 失败、超时、错误 ACK 或进程退出均不推进参考态；
5. generation 单调，恢复链不接受缺失、重复、乱序或跨 FULL base 的 frame；
6. control state 与 tensor payload 属于同一 generation，不能形成混合 step；
7. 周期 FULL 成功提交后才重置 Delta base 和 ring 回收边界。

首个真实 S2-R0 版本固定 `max_unacked_delta=1`。即上一代未 ACK 时，下一个 checkpoint
触发必须阻塞或明确返回 busy，不能基于旧参考态生成多个无从排序提交的后代。该限制
会降低吞吐，但可消除谱系歧义。正确性通过后再增加有序 speculative queue：后代基于
前一代冻结值，ACK 严格按序；任一代失败时其全部后代失效并从最后 ACK 重新 observe。

### 3.4 数值比较规则

- FULL 和 S2-R0 在“加载完成、续训前”要求 tensor/control payload 字节一致；
- batch digest、global step、data cursor 和 RNG 解码值必须完全一致；
- NPU fresh-process 续训可能存在框架/算子非确定性。正式报告同时给出严格阈值
  `rtol=1e-5, atol=1e-6` 和诊断阈值 `rtol=1e-4, atol=1e-5`；
- S2-R0 的续训偏差不得超过同模型、同进程边界的 SPDK FULL restart 基线包络。诊断
  阈值通过不能覆盖“严格阈值失败”的事实；
- S2-R1/R2 另报告逐类别 relative L2、NRMSE、最大绝对误差和固定验证集 loss/perplexity。

## 4. HBM、帧、裸盘与提交协议

### 4.1 数据路径与生命周期

实现 `S2DeltaTrainCell`（名称可按代码风格调整），将模型和 Adam 状态按 manifest
展平为参数内 block view。R0 使用原始 dtype `P_persisted`；R1/R2 才引入 scale、INT8、
residual 和 age。NPU 输出至少包含 generation、base generation、step、state ID、block
ID、参数内 offset、有效长度、dtype/encoding、scale、payload 和 control-state descriptor。

slot 状态机固定为：

```text
FREE → NPU_GENERATING → READY → IO_SUBMITTED → DATA_PERSISTED
     → METADATA_COMMITTED → ACKED → FREE
                    ↘ FAILED → FREE（不推进参考态）
```

至少提供两个 HBM slot，但 R0 correctness 期间仍只允许一个未 ACK generation。slot
只在 ACL event、NVMe completion 和 metadata commit 全部完成后释放。请求环满、slot
耗尽和 cleanup 有在途请求时必须返回可观测状态，不允许 latest-wins 或静默丢弃。

### 4.2 自描述 frame

Delta frame v4 作为后续实现目标；v1/v2/v3 只读兼容，不再扩展：

```text
FrameHeader
  magic / frame_version / strategy / flags
  schema_version / manifest_digest / world_size / rank_id
  base_full_generation / base_delta_generation / current_generation / step
  category_bitmap / block_count / small_state_count
  logical_payload_bytes / aligned_io_bytes / frame_crc / manifest_crc

BlockRecord[]
  stable_state_id / parameter_local_block_id
  element_offset / effective_elements
  source_dtype / encoding / scale / payload_offset / payload_bytes / checksum

SmallState[]
  stable_state_id / codec / payload_offset / payload_bytes / checksum
```

固定 frame header、records 和 CRC 可由 Host/Reactor 补齐；变化检测、选择和大 payload
不得通过“整模型先复制到 Host”实现。4 KiB padding 不计入逻辑 payload，但必须计入
实际 SPDK bytes 和写入比例。

### 4.3 FULL、Delta ring 与原子提交

- FULL 复用 `DirectCheckpoint.save_state()`，并在 Delta metadata 中记录 FULL generation、
  step、manifest digest 和完整状态 checksum；
- Delta 仅写入 V2 superblock 声明的 Delta partition，不使用临时安全 gap 作为正式布局；
- 每个 slot 先写 payload/envelope，等待所有 SPDK completion，再执行 NVMe flush；
- 写 inactive A/B metadata，校验并 flush 后才提升 active generation；
- metadata commit 成功后才向训练侧发 ACK；
- ring 回收只能删除早于最新保留 FULL 且不属于任何可恢复链的 frame；
- 若当前 C/SPDK 层尚无 flush/FUA 能力，该项是实现门禁，不能以普通 completion 代替
  掉电持久化结论。

恢复流程只选择 CRC/manifest/lineage 均有效的最新 FULL，随后应用连续 Delta 后缀。
若最新 frame 不完整，可回退上一完整 generation；若链中间缺失，则回退到该缺口之前
或最新 FULL，不能跳过缺口继续恢复。

## 5. 分阶段落地门禁

每个门禁只有 `PASS`、`FAIL`、`BLOCKED` 三种状态。失败运行保留，但不得进入成功样本
均值。已有实验只能关闭其明确覆盖的子项。

### G0：环境和连续训练基线

模型按 GPT-2 → GPT-2 XL → GPT-2 13B 四卡顺序晋级。固定 seed、数据顺序、batch、
sequence length、optimizer、混合精度、loss scale、NPU/NUMA 和软件 commit，执行无
checkpoint 训练并记录每步 loss、batch digest、step time、有限性和每 10 步状态摘要。

配置：GPT-2 100 steps/1 seed；GPT-2 XL 100 steps/3 seeds。现有 XL 100-step finite
轨迹可作为入口证据，但需按本计划统一结果 schema 补齐 batch digest 和环境记录。

出口：loss 和完整状态 finite；同 seed 数据顺序一致；连续训练参考轨迹可复现。

### G1：真实轨迹 CPU S2-R0 oracle

从真实训练采集 model + Adam + controls。GPT-2 每 step 采样 100 steps；GPT-2 XL 至少
100 steps、默认 `sample_every=5`，3 seeds。用 `python/s2_delta.py` 验证参数内分块、
observe/ACK、失败 ACK、FULL+Delta 回放和逐数组误差，同时输出 R1/R2 所需的变化能量、
块年龄和状态类别分布。

出口：R0 对所有采样点逐字节恢复；generation/manifest 连续；小参数和尾块无遗漏；
失败/乱序 ACK 不改变参考态。通过后冻结轨迹和 oracle 版本。

当前状态：合成 100-step 与真实训练数值入口已通过；“完整真实轨迹离线回放”未完成。

### G2：CPU/NPU 图内等价

在全零、单块、多参数、小参数、尾块、FP16/FP32、Adam m/v、极小 scale、大动态范围、
tie score 和 NaN/Inf 负向输入上逐项比较 norm、selection、state/block ID、offset、有效
长度、replacement value、scale、INT8、residual、age 和 valid count。

出口：R0 index/length/value 完全一致且无跨参数块；R1/R2 index 一致，INT8 最大差不
超过 1 bin，scale/residual/age 满足预注册容差；NPU 输出可由 CPU v4 parser 解析。

当前状态：I2 基础算子通过；完整 manifest、Adam 分类、residual/age 和 v4 parser 未完成。

### G3：HBM frame 与 ACK 生命周期

将真实图输出接入 2/4-slot HBM pool 和 Reactor writer。覆盖 100 ms/1 s/5 s 慢写、ACK
延迟、SPDK/CRC 失败、请求环满、连续触发、cleanup 在途、generation 乱序、重复 ACK 和
错误 ACK。

出口：冻结 HBM 与 NVMe 回读 hash 一致；未 ACK 不覆盖；失败不推进参考态；无死锁、
泄漏和静默丢失；背压策略与计数一致。

当前状态：A9/I7 已通过真实 HBM slot、回读和慢盘背压；图输出到 frame 及 ACK 提交
`P_persisted` 未完成。

### G4：裸盘 frame、回绕与恢复协议

在 `83:00.0` V2 Delta partition 完成 Host-SPDK 和 HBM-SPDK 的非对齐、尾块、
multi-segment 回环，并运行至少 100 generations、两次以上回绕。故障覆盖 payload、
header、manifest、step、base generation、重复、缺失、乱序、A/B metadata 和进程重启。

出口：只选择完整连续链；损坏样本按协议拒绝或回退；所有成功 generation 可恢复。

当前状态：I5 正式三用例矩阵通过；I6 100 generations、6.25 次回绕和 11 类故障通过。
在接入 v4 完整状态 frame 后必须复验，现有结果不能替代 G5。

### G5：S2-R0 真实训练端到端

1. step 0 保存完整训练态 FULL；
2. 每个 checkpoint step 从稳定训练态生成 R0 frame；
3. HBM slot → Reactor/SPDK → payload/metadata commit → ACK；
4. ACK 后提交该 generation 的 `P_persisted`；
5. 训练至少 100 个 Delta，ring 回绕至少两次；
6. 在预定故障点强制终止全部训练/SPDK 进程；
7. fresh process 加载 FULL，顺序重放 Delta，恢复 controls；
8. 恢复后继续至少 10 steps，与连续训练和 SPDK FULL restart 基线比较。

故障点至少包括生成前、READY 后、payload 半写、payload 完成但 metadata 未提交、只写
metadata A、commit 后 ACK 前、ACK 后 reference commit 前、ring 回绕和周期 FULL。

出口：加载瞬间 R0 完整状态逐字节一致；batch/RNG/cursor 一致；续训不超过 FULL
restart 基线包络；至少 100 Delta、两次回绕和 10 类故障全部通过。GPT-2 先通过，
再执行 GPT-2 XL 3 seeds。

当前状态：未完成。这是项目能够宣称“真实训练增量检查点闭环”的最小出口。

### G6：S2-R1/R2 真实轨迹候选筛选

先在 G1 冻结轨迹上做 CPU 扫描，避免 NPU 参数笛卡尔积：

| 参数 | 候选值 |
|---|---|
| block size | 65,536 / 262,144 / 524,288 elements |
| selection fraction | 1% / 5% / 10% / 20% |
| max block age | 4 / 8 / 16 |
| FULL interval | 20 / 50 / 100 / 200 steps |
| encoding | FP16 / INT8 |
| strategy | S2-R1 / S2-R2 |

每组报告选中块、逻辑 payload、frame/index/scale、4 KiB 对齐实际写量、最大年龄、逐步及
最终 NRMSE、状态类别误差、恢复 loss、FULL 摊销写入比。只保留最低写入、最低误差和
Pareto 折中三组；存在饿死块的固定 Top-K 不晋级。

当前合成扫描只作工具验证，必须以真实模型+Adam 轨迹重跑。

### G7：S2-R1/R2 NPU 正式实验

GPT-2 XL 每候选 3 seeds、至少 500 training steps、至少 30 个正式 checkpoint，前
5–10 个 checkpoint 作为 warmup；每个候选分别运行正常和 crash/recovery 实验。最优
配置再运行 1,000–5,000 steps，覆盖训练早/中/晚阶段、慢盘和 FULL interval 扫描。

出口门槛：

- 单步逐参数 NRMSE 中位数 ≤ `5e-3`；
- 100-step 链 NRMSE 中位数 ≤ `1e-2`；
- 固定批次恢复 loss 相对偏差 ≤ `1%`；
- 无块超过最大年龄；
- 周期 FULL 摊销后实际物理写入 < FULL-only 的 `20%`；
- 稳态训练 step overhead ≤ `10%`；
- 所有已提交 generation 可恢复，无静默丢失或错误 ACK。

物理写入达标但 step overhead 超标时，只能得出“减少写入但当前实现成本过高”。

### G8：四卡 GPT-2 13B 多 rank 扩展

仅在 GPT-2 XL G5/G7 通过后执行。每 rank 独立 manifest/checksum 和 Delta shard；全 rank
PREPARED 后才提交 global generation；任一 rank 失败不得发布；恢复时所有 rank 必须
属于同一 step/FULL base，随后 HCCL 训练的 batch 顺序和 loss 连续。

先执行 S2-R0 20 steps/1 seed，再将单卡最优 R2 扩展到 100 steps/1–3 seeds。现有
13B 四卡 1-step smoke 和 C2 两 rank FULL 协议只证明入口能力，不关闭该门禁。

## 6. 正式实验矩阵

| ID | 模型 | 方案 | steps / seeds | 恢复 | 目的 |
|---|---|---:|---:|---|---|
| T0 | GPT-2 | 无 checkpoint | 100 / 1 | 否 | 调试基线 |
| T1 | GPT-2 | S2-R0 | 100 / 1 | 是 | 首条真实 E2E |
| T2 | GPT-2 XL | 无 checkpoint | 500 / 3 | 否 | 正式训练基线 |
| T3 | GPT-2 XL | SPDK FULL | 500 / 3 | 是 | FULL restart 和性能基线 |
| T4 | GPT-2 XL | S2-R0 | 100 / 3 | 是 | 无损正确性 |
| T5 | GPT-2 XL | S2-R1 候选 | 500 / 3 | 是 | 固定选择策略 |
| T6 | GPT-2 XL | S2-R2 最低写入 | 500 / 3 | 是 | 写入下界 |
| T7 | GPT-2 XL | S2-R2 最低误差 | 500 / 3 | 是 | 误差下界 |
| T8 | GPT-2 XL | S2-R2 Pareto | 500 / 3 | 是 | 综合候选 |
| T9 | GPT-2 XL | 最优策略 | 1,000+ / 3 | 是 | 长程稳定性 |
| T10 | GPT-2 13B 4卡 | R0/最优 R2 | 20–100 / 1–3 | 是 | 多 rank 扩展 |

T4 前不执行 T5–T10；某策略正确性失败时不采纳该运行的性能数字。

## 7. 指标、统计与证据文件

训练侧记录 baseline/Delta graph/snapshot/trigger/slot wait/checkpoint step/non-checkpoint
step；存储侧记录逻辑 payload、SPDK submit、对齐写量、metadata、周期 FULL、持久化时间、
ring 占用；资源侧记录 HBM/Host DRAM、NPU 利用率、Reactor CPU、ACL copy 和 SPDK
completion；恢复侧记录 RTO、重放长度、各状态类别误差、loss/perplexity 和续训轨迹。

每组报告 mean、median、stdev、Student-t 95% CI 和 P95；正式样本数 `n >= 30` 才报告
P99。失败样本单独保留，不进入成功均值。

每次运行使用不可覆盖的 `RUN_ID`：

```text
results/incremental-real-training/<RUN_ID>/
  config.json              environment.json
  commit.json              manifest.json
  samples.jsonl            timeline.jsonl
  state_digest.jsonl       delta_metrics.jsonl
  recovery.json            failures.jsonl
  result.json
```

`result.json` 至少包含 status、model、seed、steps、state scope、FULL/Delta 数、逻辑及
实际字节、摊销写入比、graph/step overhead、恢复时间、NRMSE、loss deviation、最大块
年龄、fault matrix 和逐门禁布尔结果。原始日志必须记录 run ID、rank、generation、step、
request ID 和实际 NPU/PCI 地址。

## 8. 代码落地顺序与提交边界

1. `incremental manifest`：统一完整训练态清单、稳定 ID 和参数内分块；
2. `real trace`：保存可离线回放的模型+Adam+controls 轨迹；
3. `S2DeltaTrainCell`：原始 dtype `P_persisted`、R0 observe；
4. `optimizer coverage`：补齐 Adam、小参数、尾块和有效长度；
5. `frame v4`：统一 NPU 输出、自描述 frame 和 control state；
6. `lifecycle`：HBM slot、持久化 handle、ACK 后 reference commit；
7. `raw ring`：V2 Delta partition、flush、A/B metadata 和恢复选择；
8. `R0 E2E`：GPT-2/XL crash、重放和 continuation 门禁；
9. `R1/R2`：INT8、residual、age 和真实轨迹筛选；
10. `formal experiments`：NPU 性能、长程、慢盘、周期 FULL 和多 rank。

每个提交只改变一个主要语义层。frame、manifest、ACK 或布局发生变化时，必须重跑受影响
的 CPU 单测、G2 等价、G4 裸盘和 G5 恢复门禁。实验产物先留在实验分支，核心实现和门禁
复核通过后再合并 `master`。

## 9. 环境和设备安全规则

- 裸盘实验仅允许 `0000:83:00.0`；运行前用 `lspci` 和挂载检查再次确认；
- 不允许写入 `84:00.0` 或 `/models` 所在设备；
- 格式化前单独记录 superblock/分区并明确确认，普通复验不得格式化；
- NPU 实验前执行 `npu-smi info`，仅使用无进程占用的卡；
- SPDK 命令前阅读 `third_party/spdk/README.md`，确认 build、hugepage、UIO 和 root 环境；
- root 命令从 `.sudo_pw` 经 stdin 传递，日志不得打印密码；
- Python 使用 `ms_2.5`；正式运行记录 MindSpore、MindFormers、CANN、SPDK、固件、kernel
  和 git/submodule commit；
- `/home` 空间不足时优先流式 hash 和分片证据，不为 XL/13B 同时保留多份完整 Host oracle。

## 10. 推进周期与决策

建议按门禁而非自然周结束任务：

| 阶段 | 工作 | 出口 |
|---|---|---|
| P0 | manifest、真实轨迹、CPU oracle | G0/G1 |
| P1 | R0 图内等价、HBM/ACK、frame v4 | G2/G3 |
| P2 | raw ring 复验、GPT-2/XL R0 E2E | G4/G5 |
| P3 | 真实轨迹筛选、3 个 NPU 候选 | G6/G7 |
| P4 | 长程、慢盘、FULL interval、13B | G7/G8 |

最终决策：

- **Go**：R0 E2E 正确，R1/R2 实际写入 <20%，step overhead ≤10%，误差/loss、最大
  年龄、崩溃、回绕和慢盘全部达标；
- **Pivot**：写入明显下降但图内开销、编译或 HBM 不可接受，改为 NPU 仅生成摘要、
  Host 选择/压缩，或只对特定大状态增量；
- **Stop**：ACK/恢复语义无法闭合，FULL 摊销后写量无收益，误差不可控或长期成本高于
  FULL。此时固化负结果，项目主线回到用户态 I/O、异步流水和后台控制。

下一次实施从 P0 开始：先生成统一完整状态 manifest 和真实轨迹格式，再扩展 NPU 图；
不得直接以旧 `DeltaTrainCell` 开始正式 R1/R2 性能实验。
