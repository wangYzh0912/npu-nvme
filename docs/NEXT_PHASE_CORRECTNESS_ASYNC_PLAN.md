# 下一阶段：完整训练态正确性与真实异步 I/O

## 1. 当前结论与证据边界

现有实验已经证明：同一块 `83:00.0` SSD 上，256 MiB buffered ext4
写耗时约 279 ms，SPDK 写耗时约 62.3 ms，前者约慢 4.48 倍；而
`O_DIRECT` 约 66 ms，与 SPDK 只相差约 1.06 倍。因此当前证据支持的
严格表述是“页缓存、文件系统及其 buffered 路径开销很大”，不能泛化为
“所有内核代码都很慢”。GPT2 13B 的 26.2 GB FULL 写入约 6.05 s，约
4.3 GB/s，说明当前大模型数据路径可以逼近 SSD 的持续带宽。

真实 GPT2-XL HBM slot 实验中，1/2/4 slot 的前台等待均值分别约为
4279 ms、135 ms 和 0.056 ms；注入 5 s 慢盘后，单 slot 等待均值增至
约 8934 ms。这支持“足够的冻结槽位能把持久化延迟移出训练关键路径”。
raw ring 100 代、A/B 元数据、模型参数跨进程恢复和两 rank 两阶段提交
协议也已分别通过门禁。

以下问题尚未被现有结果证明：

- G1 仅保存模型参数，未覆盖优化器槽位、loss scale、RNG 和数据游标；
- G4 使用小张量验证两 rank 协议，不是实际分布式训练态恢复；
- C 后端虽创建了 ACL stream/event，读写仍调用同步 `aclrtMemcpy`；
- 增量检查点的低写入比例来自合成或离线回放，尚未证明真实续训精度。

因此下一阶段主线依次为：完整训练态正确性、多 rank 原子恢复、真正的
`aclrtMemcpyAsync` 重叠；增量检查点只并行采集真实轨迹并做离线筛选。

## 2. 完整训练态接口与磁盘语义

在 `DirectCheckpoint` 上增加：

```python
save_state(components, control_state, step, commit_meta=True)
load_state(components, step=None) -> control_state
```

`components` 是命名映射，首版支持 `model` 和 `optimizer`；已有
`save(model)`/`load(model)` 保持兼容。`control_state` 是版本化的标量、
数组或字节字段，至少保存 global step、loss scale、Python/NumPy/框架
可获取的 RNG 状态、数据 epoch 和样本游标。

每个持久化条目使用 `model/...`、`optimizer/...`、`control/...` 稳定命名，
记录 dtype、shape、有效字节数、偏移、placement 和 SHA-256。checkpoint
级元数据记录 schema version、generation、训练 step、world size、rank
manifest 和总校验。模型及优化器 NPU Parameter 在 API 返回前冻结；只有
全部 payload 完成并校验后才通过 A/B 元数据发布 generation。加载必须严格
检查名称、shape、dtype、范围和校验和，不允许静默跳过字段。

## 3. 正确性门禁

### C0：格式、接口与失败语义

- 覆盖命名空间、元数据往返、旧接口兼容、重复字段、shape/dtype 不匹配；
- 覆盖 payload、单份元数据和未完成 generation 损坏；
- 失败 generation 不得覆盖上一份有效 checkpoint。

### C1：单卡完整续训

先以小模型调试，再以 GPT2-XL 形成正式结果。固定初始化和数据顺序：基准
进程连续训练 20 step；保存进程训练 10 step 后保存完整训练态并退出；新
进程加载后继续训练 step 11--20。加载瞬间模型、优化器、global step、
loss scale、RNG 和游标必须逐项一致。续训逐 step 比较 loss、数据顺序及
状态摘要；状态哈希以字节一致为目标，loss 使用
`rtol=1e-5, atol=1e-6`。另注入 payload 损坏、元数据损坏、控制字段缺失和
未完成写入，要求明确失败或回退上一 generation。

### C2：两卡实际训练恢复

用 GPT2-XL 和真实优化器执行两 rank 训练、保存、完全退出、重新拉起和
续训。首版坚持单 SPDK owner：rank 先发送 manifest，再分 chunk 把本 rank
状态交给 coordinator；coordinator 写入 rank 独立区域并校验，全部 rank
进入 PREPARED 后才发布全局 COMMIT。重启只接受全局已提交 generation。

故障矩阵覆盖一个 rank 在 prepare 前、传输中、prepare 后及 commit 前
退出。任何 rank 失败都不得暴露半成品，上一 generation 必须仍可恢复。
该 host/Unix-socket 转发路径仅用于正确性，不作为最终性能结果。

#### C2 当前执行记录（2026-08-27）

已完成 correctness-first 的两 rank 版本：GPT-2 小模型、rank 0/1 各运行
真实 MindFormers 训练 cell 与 Adam，保存完整 model/optimizer/control state；
rank 通过 Unix socket 将 manifest 和 4 MiB 分片发送给单一 coordinator，
coordinator 在 `0000:83:00.0` 上完成校验、分片写入和一次全局 metadata commit。
两个 fresh restore 进程随后分别加载自己的 shard，并完成 1 个 continuation
step；两 rank 的 loss 均为 `10.875582695007324`，与保存进程 continuation
完全一致，结果为 C2 PASS。

本轮同时修复了两个问题：SPDK shared-memory primary 约束导致的并发 restore
冲突改为串行 restore；MindSpore fresh process 对单元素 optimizer 参数可能
产生 `[]`/`[1]` 形状差异，加入仅限单元素且 dtype/size 一致的兼容规则。
本实现是两真实进程的多 rank 状态提交/恢复验证，尚未启用 HCCL 梯度同步，
因此不能替代 C3 的四卡实际分布式训练和故障矩阵。

### C3：四卡与 13B 规模门禁

先用四卡 GPT2-XL 完成正常恢复及两卡相同的故障矩阵。通过后运行四卡
GPT2 13B：至少完成一个训练 step、保存完整训练态、退出、重新加载和一个
续训 step。检查各 rank 分片、优化器状态、全局 step、数据游标和提交
generation，并记录每 rank 数据量、冻结、传输、提交和恢复时间。

## 4. 真实异步与增量探索

正确性门禁通过后，把写路径和读路径的同步 `aclrtMemcpy` 替换为
`aclrtMemcpyAsync`。写状态机落实为
`HBM_READY -> NPU_COPYING -> NPU_DONE -> NVME_SUBMITTED -> DONE`；读路径
在 NVMe 完成后进入异步 H2D，通过 ACL event 轮询完成。热路径不得执行
整流同步，slot 只有在 ACL event 和 NVMe completion 均完成后才能复用。
同步基线与异步版本比较总耗时、前台等待和 D2H/H2D--NVMe 重叠率，并
重新运行 C1--C3。

并行采集 GPT2-XL 和 13B 的真实权重、优化器及 loss 轨迹。R1/R2 首先只做
离线回放；只有写入比例、最大状态年龄、恢复误差和续训 loss 同时满足门禁，
才进入 NPU 增量编码实现。

## 5. 环境、证据与合并规则

- raw SSD 实验只允许使用 `0000:83:00.0`；运行前确认 PCI 地址、挂载和
  SPDK 绑定，禁止格式化或写入其他设备；
- NPU 实验前运行 `npu-smi info`，只使用无进程占用的卡，并记录实际卡号；
- Python 环境使用 `conda activate ms_2.5`；SPDK 命令前按仓库 README
  完成构建和配置，需要时从 `.sudo_pw` 读取密码切换 root；
- 每个门禁保存命令、环境、commit、设备信息、原始日志和机器可读汇总；
- 计划、完整训练态实现、单卡结果和多卡结果分别提交；全部门禁通过并复核
  后才合并 `master`。

本阶段验收标准是：完整训练态可跨进程恢复并产生一致续训轨迹；失败
checkpoint 不可见且上一 generation 可恢复；两卡和四卡均不存在部分提交。
