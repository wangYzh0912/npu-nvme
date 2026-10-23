# FULL Checkpoint 当前流程与清理结果

更新：2026-10-24。本文对应 FULL 实现及定向硬件验收，不代表完整性能发布。
实施前逐项审查保留在 [SOURCE_IMPLEMENTATION_AUDIT.md](SOURCE_IMPLEMENTATION_AUDIT.md)。

## 当前结论

本轮已完成源码清理与软件契约闭环：D1 已准入操作纳入统一 drain；Qwen raw commit
与公开目录之间增加可验证对账；资源预算接入静态及运行时准入；旧 FaF、delta bookkeeping、
未使用的 LiveCapture/cleanup_legacy 已退役；通用 `none` gate 已收紧。

**NPU4–7 targeted FULL 硬件闭环已通过；全量性能 campaign 仍未执行。** 新 ABI3 环境已构建并通过加载与符号检查；NPU4–7 的 Qwen none24
oracle 和 NPU7 copy-only 双向 ACL/DMA 探针通过。获授权后，raw83 (`0000:83:00.0`) 已绑定
`uio_pci_generic`，受保护的 `0000:84:00.0` 仍绑定 `nvme`，`/models` 挂载未变。
历史 Ours source8 曾在 rank-local native copy context 初始化时以 `-13` 退出。随后用
`fs.protected_regular=1` 复现：root 对已存在且归普通用户所有的 `/tmp` 锁文件执行带
`O_CREAT` 的 `open` 返回 `EACCES`；只读写打开同一文件则成功。该 errno 来自 native-owner
锁的 open 路径，现先打开已有文件，仅 `ENOENT` 时 `O_EXCL` 创建，遇到创建竞争再打开；
未放宽内核保护或删除共享锁。另行确认的锁键问题是逻辑 NPU 序号未映射到物理卡，可能导致不同可见设备
争用同一锁；现已按 `ASCEND_RT_VISIBLE_DEVICES` 将逻辑序号映射为物理 NPU ID，ACL 仍使用
逻辑 ID。映射用例、ABI3 构建/加载通过；本次 4–7 硬件验收已验证 native open、copy、restore、save
和 close。
第二次 acceptance 因外部 GDN-AIME 正式验证占满 NPU0–7 而被 admission 正确阻断，未启动任务。
历史 ABI2 通过记录不能替代当前 ABI3 的 Ours 硬件验收；本次 ABI3 验收现已补齐，详见下文。

## 方法与范围

- `none`：无持久化的训练 oracle，不能声称 restore 或 persistence 成功。
- `mindspore_native_save`：显式 rank-local FULL safetensors。
- `bytecheckpoint_host`：独立 Host worker 适配器。
- `ours`：阻塞 FULL；四 rank 保留 ACL/NPU 上下文，经有界 Host socket bridge 发送给单一 SPDK owner。

FULL 包含模型、Adam 和训练位置/RNG 控制态。Qwen 在保存边界暂停训练更新，不新增完整
HBM 快照，也不是 NPU–NVMe 直接 P2P。摘要仍逐 tensor 调用 `asnumpy()`。
D1 另用 frozen snapshot；guarded live、R0、增量和 Top-K 仍属独立研究，不因本轮清理而删除或转正。

## 保存流程

```text
train.py -> config/source/environment identity -> hardware lock + lease
  -> static resource preflight (Host/hugepages/region)
  -> fresh SPDK owner mounts registered D2 region, exports retained inventory
  -> catalog reconciliation, then optional restore selection + reader pin
  -> TP4 workers open their private native pools
  -> runtime admission: actual tensor geometry + available Host/HBM
  -> before initial/step-end tensor manifest or large copies
  -> periodic FULL boundary: stable local state + controls
  -> rank 0 reserves filesystem generation
  -> each rank fsyncs prepared.json (state/controls/identity/raw request ID)
  -> all-rank preparation barrier
  -> ACL D2H -> bounded socket -> single TrainingOwner -> D2 payload writes
  -> chunk SHA/schema/topology checks + source-safe ACKs
  -> payload flush -> immutable metadata pages + flush -> alternate anchor + flush
  -> raw commit receipt -> each rank complete.json -> all-rank barrier
  -> rank 0 publishes checkpoint.json + filesystem prune
  -> all ranks observe publication + barrier -> training continues
```

`source_safe` 仅允许复用相应传输源，不代表整代持久化。optimizer step、filesystem generation、
raw generation 是三个不同标识。新运行保留历史 `source-identity.json` 字段，并额外记录包括
未跟踪源码的 `execution_sources`，执行结束再次比对；新证据复用也检查这个身份。

## 中断后的对账

raw media receipt 是提交权威，filesystem catalog 是可补全的公开索引，未改变 D2 磁盘格式。
owner fresh mount 后提供**仍保留**代的 receipt、step、topology 和 controls 摘要。
supervisor 在启动 rank、选择恢复代之前持 catalog 锁扫描未发布 reservation：

1. 必须存在四 rank 完整、校验通过的 prepared sidecar，lineage/identity/step/rank 一致。
2. 四份 raw request ID 必须一致，并匹配 retained receipt、identity、step、TP4 topology、controls 摘要。
3. 先校验所有已有 completion、文件摘要和整组 publication，再补写缺失的 completion，最后原子发布。
4. 缺失 completion 以 `recovered=true` 标记，未知 save/backend timing 为 `null`，不伪造性能样本。

操作可重复，已经有效发布的目录不重写。缺 rank、历史无 journal、raw 记录已退休/结果未知、
损坏或冲突都写入 `reconciliation.json` 的拒绝记录，保留原文件，不冒充 abort 或成功。
对账不读取全部 payload；真正恢复仍必须经过 chunk/tensor/control 校验和全 rank ready。
该机制不等于通用分布式事务、自动修复任意损坏或掉电安全证明。

## 新进程恢复

fresh owner mount + reconciliation -> explicit/latest 选择及 reader pin -> 身份、文件及 raw retention
检查 -> fresh rank runtime admission -> verified read/H2D 或 Host apply -> 完整 tensor/control readback
-> all-rank prepared proof -> owner global ready -> rank barrier -> supervisor 释放 pin
-> worker 观察 pin-release -> Trainer 继续训练。

explicit 不静默换代；latest 可跳过损坏/不再保留的公开代，记录拒绝原因。
失败不会放行恢复目标。未知 DMA、无法证明安全关闭时保留映射和 lease，不自动杀进程或删 lease。

## 清理与所有权

| 原问题 | 当前方法 |
| --- | --- |
| D1 freeze 阶段尚无 handle，close 提前结束 | admission 时登记 operation token；save/restore 同一 registry，drain 快照该水位，关闭 admission 后排空 |
| FaF 空任务假成功、重注册 UAF | 从 C/header/bindings/build 同步移除 listener/probe/register API；SONAME/ABI 升为 3，不保留危险兼容分支 |
| 旧 delta 区域 bookkeeping | 删除未支持的公开 API 和上下文字段；保留通用 I/O 校验、D2/R0 研究实现 |
| 旧编排与虚假通过测试 | 删除无 caller 的 LiveCapture、cleanup_legacy 和恒真/TBD C 测试；薄 aliases、baseline 适配器保留 |
| 预算仅存在于工具 | 正式 preflight 比较聚合 Host、hugepages、region；runtime 再比较实际几何和 ACL 可用 HBM |
| `none` helper 允许空 pass | 必须明确训练通过、restore 不适用且未持久化；不能将训练 oracle 当成恢复通过 |

资源预算包含 owner + 四 rank 原生池、metadata pool、socket、chunk 临时副本、描述符、controls、
逐 tensor Host 副本及 HBM 临时张量上界。默认保留 Host 16 GiB、每 rank HBM 1 GiB；
runtime 不重复计费已经分配的 native pools。估算不冒充实测峰值；预算不足直接失败，不换后端。

保留 request refs、quarantine、reader pin、source/transport-safe 和 unready target。
这些是异步生命周期责任，不是本轮要求减少的冗余防御。内部依赖已建立的契约，外部输入及
磁盘/IPC/原生边界由其 owner 校验，不新增多层 fallback 或可推导状态标志。

## 环境与验证

默认配置已切到独立新环境：
`/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json`。
新库位于该环境的 `candidate/build/libnpu_nvme.so.3.0`，SHA256：
`070f87d5bc3a2a32723274c9319e5b08356f5f36163e3ad3594b81a12daf4189`。
构建使用已记录的修补版 DPDK ring archive；未修补的 archive 会留下 `common_ring_get_count`
未定义符号，不能仅凭链接完成判断环境可加载。
旧 ABI2 环境和历史验收文件保持原样；当前 loader 明确拒绝 ABI2。不要用新 loader 搭配
旧实验硬编码的 `.so.2` 路径，也不自动迁移环境身份不同的旧训练 lineage。

软件与 C 检查、ABI3 构建/加载均已通过；具体命令见
[执行说明](EXECUTION_ENVIRONMENT_AND_COMMANDS.md)。本轮证据：

2026-10-24 独立复验同样通过 7 场景、5 比较，driver 返回 0；发布中断 raw27，自动恢复并
保存 raw28/step8，owner 安全关闭，无手工重试。证据目录为
`/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261024-001`。
提交候选软件/ABI 390 项、C sanitizer 45 项通过；与硬件冻结树的差异范围见
`DEVELOPMENT_STATUS.md`。下列 005 及更早记录保留其历史范围。

- `/tmp/npu-nvme-full-closure-npu-map-python.xml`：417 项 Python/CPU 软件回归，通过。
- 历史 C harness 为 43 项；加入物理卡映射及已有锁文件用例后，本轮 45 项 ASan/UBSan 通过。
- root 执行 `native_owner_existing_file` 通过，覆盖普通用户持有的已有 `/tmp` 文件；不含设备验收。
- `/tmp/npu-nvme-full-closure-final-python.xml`：最终 30 项通过，覆盖验收 driver、去重/空间准入、ABI3、准备顺序与真实 D2 对账契约。
- `/tmp/npu-nvme-full-closure-npu-map-abi3.xml`：3 项 ABI3 符号、SONAME、无设备初始化的版本检查，通过。
- `tests/python/test_training_reconciliation.py`：内存介质上真实 D2 commit/mount 与中断目录对账。
- `tests/python/test_full_closure_contracts.py`、`tests/d1/test_runtime.py`：准入、准备先于 payload、close 时序。
- 4–7 映射只读 preflight：所选卡空闲、ABI3、容量准入通过；软件回归 417 passed，ABI3 测试 3 passed。
- NPU7 使用 TP4 完整可见列表 `4,5,6,7` 和 rank3 逻辑设备的 copy-only probe 通过：8 B、4 MiB D2H/H2D 均字节一致，executor 正常 close；证据为 `/models/npu_nvme_exp/user7-stack/full-closure-placement-npu7-mapped-20261020-001/result.json`。
- 历史首次 Ours source8 的 owner mount 成功，随后 rank-local native open 返回 `-13`，无 owner checkpoint/payload 操作，run 失败并已退出。物理卡映射修复后的旧候选 `full-closure-abi3-20261021-002` 已构建/加载，但尚未包含已有文件 open 修复。记录见 `/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261020-002/result.json`。
- 历史 acceptance `/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261021-003/result.json` 被 admission 标记为 `blocked`：另一项 GDN-AIME 正式验证占满 NPU0–7，未启动 FULL 作业。当时的离线 Python 回归 55 项、C ASan/UBSan harness 44 项、新候选 ABI/符号检查 3 项通过。

定向硬件入口 `tools/full_checkpoint_acceptance.py --npu-ids 4,5,6,7` 的当前完整证据为
`/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261021-005/result.json`：
主 driver 返回 0，最终 `status=pass`，7 个场景、5 次 oracle comparison 全部按预期完成。

| 场景 | 结果 |
| --- | --- |
| none24 oracle | final_step=24 |
| Ours source8 | 保存 step4/8；oracle 比较通过 |
| 显式 generation1 恢复 | step4→8；oracle 比较通过 |
| latest 首次新进程恢复 | step8→16，保存 step12/16；oracle 比较通过 |
| latest 第二次新进程恢复 | step16→24，保存 step20/24；oracle 比较通过 |
| 发布中断 | raw generation19/step4 已提交，catalog 尚未发布；注入 worker 失败，owner 安全关闭 |
| 自动新进程恢复 | `recovered=[1]`、`rejected=[]`；选中 catalog1/step4，续训至 step8、保存 raw20，owner 安全关闭；第五次 oracle 比较通过 |

状态比较覆盖模型和 Adam 的 SHA256/geometry、controls 精确一致；loss 使用
`rtol=1e-5, atol=1e-6`。中断 run 的返回码 1 和局部 `validation_status=fail` 是预期故障，
由 driver 校验注入证明、未发布目录及安全关闭后接续 recovery，不应将其改写为训练成功。
本轮不依赖手工重试。测试结束后 NPU0–7 均健康且无运行进程，raw83 仍为 `uio_pci_generic`，
受保护 raw84 仍为 `nvme`，`/models` 仍挂载 `/dev/nvme1n1`。

历史 `...20261021-004` 主 driver 的 `fail` 保留：fault worker 收尾、首次 recovery 启动超时及
重复转换模型占满磁盘曾阻塞流程；该轮手工 `ours-publication-recovery-retry3` 最终恢复 raw11、
保存 raw12 并通过比较，不能将其当成主 driver 整轮通过。本轮使用独立的 005 证据替代该缺口。
已获授权的 SHA256 核验硬链接释放历史重复空间约 81 GiB；005 自动去重结束后 `/models`
剩余约 59 GiB。只链接已完成 run 的只读用途模型 shard 和名称映射，不链接 checkpoint。
工具随后将整文件哈希改为有界分块读取，并将空间准入收紧为**输出文件系统**容纳两份转换缓存
加 8 GiB；这些工具细化由上述软件回归覆盖，硬件证据仍保留执行时的冻结源码身份。

fault worker 只注入目录发布边界失败，不添加生产故障开关、不模拟突然断电。
仅使用 NPU 4–7、raw83 `[256,1280) GiB`，不格式化。

本次提交另整合上游 `64730754` 的 D2 更新：2026-09-18 ABI2 的受控 flush 故障矩阵
和 TP2/TP4 fresh-process fixture 已通过，详见 `docs/plans/D2_FINAL_VALIDATION_20260918.md`。
该证据不能当成本轮 ABI3 的任意故障或掉电证明。剩余为真实 D1 源镜像迁移、跨进程
rank disconnect/owner 故障矩阵，以及四方法完整性能矩阵、ByteCheckpoint 延期项。
研究任务保持各自登记；本轮最终软件及硬件复验见 `DEVELOPMENT_STATUS.md`。
