# 源码问题与方案逐项确认

> 以下是本轮清理**实施前**的审查快照，保留缺陷复现与决策依据，不能作为当前代码状态。
> D1 水位、FaF、none gate、资源准入、dirty source identity 和 raw/catalog 对账已在源码中处理。
> 当前调用链、清理清单、ABI3 环境和验证边界见 [FULL_CHECKPOINT_FLOW.md](FULL_CHECKPOINT_FLOW.md)。
> FaF listener 与旧 live 模块已删除，下方旧路径仅用于历史定位；复现脚本不再适用于 ABI3。

审查日期：2026-10-20。检查当前工作区源码、CMake 实际编译清单、相关测试和历史验收记录。源码含当前未提交修改；本审查不将工作区结果等同于干净发布版本。本次没有重新执行 NPU/NVMe 硬件实验。

**结论**：主体 FULL、安全生命周期、异步传输、TP4 owner/bridge、格式与证据机制的代码骨架已按规划拆分并接入对应路径。它们不都具有同等验收程度：有些仅通过 CPU/C mock，有些已有历史硬件验收，有些格式或增量能力仍在研发。审查还复现了两个不属于“尚在探究”范围的问题：严格 FULL runtime 在 close/drain 时可能提前关闭仍被一个已准入保存操作使用的传输；FaF 兼容入口存在零任务假完成和重复注册 use-after-free。另有结果 gate 的 `none` 分支接受缺少训练证据的 `pass`。

## 重构后的源码位置

`src/npu_nvme.c` 只有 ABI 版本导出并非实现丢失。CMake 把实现拆入 [runtime.c](src/runtime.c)、[reactor.c](src/reactor.c)、[request.c](src/request.c)、[validation.c](src/validation.c)、[dma.c](src/dma.c)、[write_pipeline.c](src/write_pipeline.c)、[read_pipeline.c](src/read_pipeline.c)、[metadata_io.c](src/metadata_io.c)、[step_listener.c](src/step_listener.c) 和 [metrics.c](src/metrics.c)。私人跨模块声明在 [implementation.h](src/internal/implementation.h)，公共 ABI 仍在 [npu_nvme.h](include/npu_nvme.h)。实际生产源清单见 [CMakeLists.txt](CMakeLists.txt#L134)。

Python 也采用职责拆分：[checkpoint.py](python/npu_nvme/checkpoint.py) 是兼容门面；严格 FULL 编排、提交、恢复分别在 [strict_checkpoint.py](python/npu_nvme/strict_checkpoint.py)、`runtime/d1_runtime.py`、`runtime/d1_commit.py`、`runtime/d1_restore.py`；捕获、状态适配、C ABI 与存储分别在 `framework/`、`storage/`。TP4 的 rank 会话、owner 和 region format 在 `d2/session.py`、`d2/rank.py`、`d2/format.py`。不能以门面或原文件行数判断功能是否实现。

## 问题与方案映射

“代码完成”只表示当前树里有相应实现；“验收”需同时看运行范围。`软件验证` 指本次 CPU/Python 或 stub C，`历史硬件` 指仓库已有的硬件记录，`待验收` 不等于已通过。

| 规划问题 | 源码确认 | 判断 |
| --- | --- | --- |
| P01 准入/启动失败时锁与资源误释放 | [scheduler.py](python/npu_nvme/runtime/scheduler.py) 的有界 admission/lease；`d1_runtime.py` 在线程启动失败时撤销 reservation 与 busy；`tests/d1/test_runtime.py` 覆盖 start failure、close 唤醒。 | **已实现；软件验证通过。** 原则是每个 acquisition 由同一 owner 释放。 |
| P02 close、排队请求与 request UAF | [request.c](src/request.c) 的 caller/reactor 引用计数；[reactor.c](src/reactor.c) 在停止时取消未启动项；C host-safety harness 直接编译这些生产源并覆盖 early release、late completion、queued close。 | **已实现；C_IMPL 44 项本次通过。** |
| P03 未证停 DMA 被复用 | [dma.c](src/dma.c)、[read_pipeline.c](src/read_pipeline.c)、[runtime.c](src/runtime.c) 在 stop 无法证明时隔离 slot/context；请求 timeout 不取消设备操作；[storage/full_transport.py](python/npu_nvme/storage/full_transport.py) 保留借用 buffer。 | **软件安全规则已实现；硬件验收不完整。** H02 子集不证明任意故障下 DMA 已停止或可安全恢复。 |
| P04 FaF 注册、步进和提交 | [step_listener.c](src/step_listener.c) 仍导出 `set_step_ptr/register_tasks` 并在 combined role 下运行旧 FaF。重复注册先释放 `old_tasks`；零任务周期可完成空请求；未见它使用新的 commit coordinator 或 namespace ownership contract。 | **未落实，且是当前可达的兼容 ABI。** 见下方复现；虽然 StrictCheckpoint 将入口拒绝，不能据此说 C ABI 已禁用。 |
| P05 唯一提交者、幂等与错误归属 | [d1_commit.py](python/npu_nvme/runtime/d1_commit.py) 管理单 rank FULL reservation/publish/resolve；D2 [TrainingOwner](python/npu_nvme/d2/training_session.py) 顺序处理 TP rank operation，最终由 [Region.commit](python/npu_nvme/d2/format.py) 写 anchor；`PersistentR0` 也通过 Region 提交。 | **按格式/owner 的单 writer 路径已实现；不是一个跨 D1、文件 catalog、D2/R0 的通用 coordinator。** D1 的 resolve 仅活进程有效；跨重启幂等索引归 D2。增量整合仍属未验收研发。 |
| P06 长度/分配/DMA 分叉 | [validation.c](src/validation.c) 限 item、总字节、地址和 namespace；[chunks.py](python/npu_nvme/storage/chunks.py)、[d1_schema.py](python/npu_nvme/runtime/d1_schema.py)、[full_state.py](python/npu_nvme/framework/full_state.py) 校验 shape×dtype 与实际长度。 | **核心 FULL/传输边界已实现。** 仍须对新增 backend/chunk owner 使用同一权威长度，不应靠重复校验弥补不同来源。 |
| P07 V2 header/anchor 与历史引用 | V2 [storage/format.py](python/npu_nvme/storage/format.py) 的 metadata CRC 仅覆盖压缩 body；头中的 generation/flags 不在 CRC 覆盖内。D2 [format.py](python/npu_nvme/d2/format.py) 用固定头和 payload 一起校验、双 anchor 与 extent union。 | **D1/V2 的已知边界未修；D2 有代码与 CrashDisk CPU 测试，但 format/migration 硬件验收仍 partial。** 不能把 D2 草案当正式格式接受。 |
| P08 部分恢复后可训练 | [d1_restore.py](python/npu_nvme/runtime/d1_restore.py) 先校验再创建新 unready target，逐块校验、控制态读回后才 ready；`full_state.py` 访问器要求 ready。Qwen worker 在恢复与 TP barrier 完成前不开始 Trainer。 | **软件事务已实现。** Qwen Native/Ours 有相应 fresh-process 历史验收，严格 D1 的完整硬件范围仍有自身门禁边界。 |
| P09 live 源与 checksum 排序 | Strict FULL 明确报告 live unsupported；但 [framework/live.py](python/npu_nvme/framework/live.py)、`runtime/handle.py` 与 C FaF listener 源码仍存在。 | **产品 strict 路径已隔离，底层兼容能力没有整体退役。** 不能声称 live/FaF 全部关闭或通过；FaF 当前缺陷见 P04。 |
| P10 溢出、容量、metadata 范围 | [validation.c](src/validation.c) 用除法保护 delta size×count，校验 namespace 范围；[metadata_io.c](src/metadata_io.c) 限制大小和 extent；Python schema 对 geometry 有界。 | **已实现；相关 C 安全用例通过。** |
| P11 增量帧、manifest、ACK 与 reference advance | [incremental.py](python/npu_nvme/d2/incremental.py)、[r0_session.py](python/r0_session.py)、`framework/byte_capture.py` 实现 native-dtype bitmap/帧和 durable receipt 后推进；D2 共享 extent/anchor 由 format 管理。 | **CPU/Fake 能力已相当完整；F0/F1/Qwen full-state 硬件验收仍延期。** 该功能未接入已发布 Qwen 普通训练配置，不把代码存在算作产品支持。 |
| P12 结果 gate 和 CLI 不得假成功 | [schemas/evidence.py](python/npu_nvme/schemas/evidence.py)、[tools/run_gate.py](tools/run_gate.py)、训练 campaign 传播子进程退出码；`tests/regression/test_gate_evidence.py` 覆盖坏 evidence。 | **证据 manifest gate 已实现；仍有两个缺口。** `types.validate_result_gate()` 的 `mode=none` 只拒绝 `restore_verified=True`，会接受 `{status: pass, mode: none}` 等无训练证据结果；`config/gates/` 不存在，按名称调用的 run_gate profile 无法解析。实际入口有更强规则，不能因此把宽松通用 helper 当可靠 gate。 |
| P13 admission deadline、close 唤醒和 drain 水位 | Python scheduler/D1 admission 为 deadline + condition；C 在 close 时停止 admission，bounded wait/quarantine。`tests/d1/test_runtime.py` 覆盖一类等待者被唤醒。 | **常规准入已实现；严格 runtime 的 drain/close 水位仍有并发缺口。** 见下方真实 RestoreTarget CPU 复现。 |
| P14 大 batch 与每 tick 工作量 | C 限 65,536 descriptors、64 GiB batch、pending 数；[runtime.c](src/runtime.c) 暴露预算；读写 FSM 限 submit、copy、checksum 和 quantum。 | **代码已实现，host-safety 对应项通过。** 不声称在所有真实设备负载下服务时延都已硬件验收。 |
| P15 比较条件、公平性与计时 | [training_validation.py](python/npu_nvme/runtime/training_validation.py) 比 continuation state/control/loss；[training_benchmark.py](python/npu_nvme/runtime/training_benchmark.py) 分组记录 restore-ready timing；旧 contracts 标明 Native/Ours/ByteCheckpoint 比较边界。 | **比较规则已实现；当前发布 campaign 明确 deferred，timings 为空。** 不能从代码或 C1 旧 workload 推断新 Qwen 四方法 matrix 已完成。 |

## 两个复现的生命周期问题

### 严格 FULL 的 close/drain 提前结束

在 [d1_runtime.py](python/npu_nvme/runtime/d1_runtime.py#L54) 中，save 在取得 reservation 后先执行 `prepare/freeze`，直到这些步骤结束才创建并登记 `FullHandle`。`drain()` 只快照 `_handles`，且仅在快照为空时额外检查 `_busy`；与此同时 `begin_restore()` 没有与 save admission 共用 busy token。若 close 进入 drain 时 restore handle 在运行、另一 save 正处于 freeze，则 restore 结束后 drain 会退出，`close()` 关闭传输，而 save 仍持 busy 并持有尚未完成的 freeze。

复现入口：[reproduce_drain.py](audits/source-20261020/reproduce_drain.py)。本次输出：`real_restore_ready=true`、`transport_closed=true`、`accepted_save_still_busy=true`、`capture_still_blocked=true`。这是受控 CPU 时序复现，没有设备 I/O。影响严格 D1/StrictCheckpoint 并发 save + restore + close；已发布 Qwen TP4 使用独立 `CheckpointController`/D2 owner worker，没有据此推断 Qwen 主路径已触发同一故障。

建议修法是把“已获准、尚未登记 handle”的操作也纳入一个明确的 operation lease/watermark，close 先线性化关闭 admission，再排空同一 owner 接受的全部操作。不要添加更多 `_closing2`、`restore_pending` 等可推导标志。

### FaF 零任务完成和重注册 UAF

[step_listener.c](src/step_listener.c#L99) 的周期路径不要求至少注册一项，也不检查 step counter memcpy 结果；无任务时 `write_fsm_tick()` 可将空请求标完成，poller 仍可走成功 flag 分支。[register_tasks](src/step_listener.c#L150) 在第二次替换时无条件释放 `old_tasks`，而写 FSM 可能继续借用第一代数组。

复现入口：[reproduce_faf.c](audits/source-20261020/reproduce_faf.c)。用生产 C 源和现有外部调用 stub 编译：`zero-tasks` 模式确认空 FaF 被当成功；`reregister` 模式在未完成请求时连续注册两次，AddressSanitizer 报 heap-use-after-free。两者不需要 NPU/NVMe 硬件。

该能力应优先从公开 product ABI 中退休：combined 角色不再暴露旧 step listener，确认无受支持 caller 后把它移入归档。若确需保留研究实验，应至少由单一 owner 固定注册生命周期、拒绝零任务、核实源读取状态、以同步持久提交 receipt 替代“DMA 完成 flag”，并单独验收。只在每处分支增加 null/状态检查无法修复数组借用契约。

### 通用 none 结果误判成功

[types.py](python/npu_nvme/types.py#L74) 的 `validate_result_gate()` 对非 none 要求 persistence/restore 字段，但 `mode=none` 分支只限制不得声称 restore verified 就立即返回。本次确认以下输入被它接受：`{"status":"pass","mode":"none"}`、`{"status":"pass","mode":"none","restore_status":"pass"}`。统一 Qwen compare 和 `cli/contracts.py:outcome()` 有更严格验证，故问题限定于该导出通用 helper/潜在 caller；修复时要求 none 结果具备训练/trajectory 完成证据，且 restore 状态明确为 `not_applicable`。

## 其他完成度不一致

| 规划/实现领域 | 实际状态 |
| --- | --- |
| RF-02 重构 | CMake 已编译拆分后的源码；C_IMPL 本次 44 项 ASan/UBSan 通过。主文件变短是正常结构变化。 |
| IO-01/02/03 与 C2 | D2H/H2D、NVMe read/write、metadata/flush、Host owner/copy rank 和每 tick budgets 均有代码；blocking FULL wrapper 通过 submit/wait 使用它们。旧 blocking bulk API 的 C 符号已移除。不过长期计划仍将 IO-01..03/IO-04 记为未完成，H07 分向验收并非齐全；应将“代码迁移完成”和“能力验收完成”分开登记。 |
| QW-06 资源预算 | 有 schema/公式和 `tests/c2/test_resources.py`；`train.py preflight` 未调用 `qwen_resource_plan.plan()`，也未在 admission 前比较聚合 Host/HBM 峰值/Headroom。D2 `BlockingState` 只对单个 Host tensor 做限制。计划仍标未完成是合理的。 |
| QW-07 Ours TP4 | 已接入 Qwen training adapter：四 rank 通过单 owner socket bridge 传 chunk，TP schema/checksum/receipt/global publish/fresh restore 都有实现与历史硬件记录。用户发布范围仍固定 TP4。长期计划中的 checkbox 仍未勾选；其多轮验收和完整性能矩阵也尚未完成。 |
| D2 format/migration | 新格式、immutable pages、anchors、retention/pins、V2 migration 和故障模型测试存在；机器状态仍记 partial，未完成硬件格式/迁移矩阵。D2 format/session 已由当前 Qwen Ours 实际使用，不能整体当作未接入的研究代码；注释中的 Draft 与当前调用范围需要同步，使用中也不等于完整格式验收通过。 |
| G05/FaF | 生产库仍导出 `npu_nvme_register_tasks` 和 `npu_nvme_set_step_ptr`。本次清点运行库的 nm 结果证实符号存在；StrictCheckpoint Python facade 禁用不等同于 C ABI 禁用。 |
| G14 profile runner | `schemas/evidence.py` 与 `tools/run_gate.py` 在树中，但 `config/gates/`、`tests/gates/` 不存在；profile runner 的名称快捷入口不可用。测试有真实 profile fixture，不代表缺少的正式 A profile 可跑。 |
| C smoke 测试 | [test_npu_nvme.c](tests/c/test_npu_nvme.c#L94) `test_constants()` 两项断言不读实际宏/ABI；设备读写回环在 `:166` 仍是 TBD。README 对 TBD 有明确警告，但空壳保留会让 smoke 的 PASS 数量失真。 |
| dirty source identity | `tools/run_campaign.source_snapshot()` 仅枚举 `git ls-files` 跟踪文件；Qwen 未跟踪的可导入 Python 源可能不进入 source digest。图 Top-K launcher 只 hash 显式 config、launcher、`graph_*.py`，不等于整棵 dirty worktree 冻结。正式 acceptance 前应要求 clean tracked tree + no untracked executable source，或对实际导入闭包整体 hash。 |

## 验证记录与限制

- 本次重跑 `tests/python`：185 passed，4.88 s。
- 本次重跑选定的 `tests/d1`、`tests/d2`、资源/FULL/campaign/evidence 用例：161 passed，19.36 s。
- 本次重跑 `tests/regression/test_c_host_safety.py`：44 passed，ASan/UBSan。
- FaF 特殊复现：零任务 flag 假成功；重复注册在 ASan 下触发 heap-use-after-free。
- Strict D1 特殊复现：生产 `StrictRestoreSession` 的真 restore target 变为 ready 后，已接受的 save 仍在 freeze 阶段，close 已退出并关闭 transport。
- 代码审查读取了历史 Qwen 394-case JUnit 和图 Top-K 19 配置/76 rank evidence；这是旧 run 记录，不是本次硬件验收。
- 没有重新构建或运行真实 ACL/SPDK/NPU 测试，没有验证现有硬件状态；D2、Qwen 性能和故障矩阵按计划保留未验收状态。

两个生命周期问题的最小复现源码位于 `audits/source-20261020/`；FaF 复现包含零任务和重注册两种模式。通用 `none` gate 的反例输入已列在上文，尚未另存为回归测试。
