# M1 补测执行契约（2026-11-16）

本轮补齐全保存边界、观测扰动、资源/生命周期及设备捕获。旧 M0/M1 数据保持独立，
不把旧筛查结果当作新门槛的通过证据，不合并旧、新 P95。入口为
tools/checkpoint_io_closure.py，经 train.py fit、Qwen callback、CheckpointController、
TrainingCheckpoint、RankSession、单 NVMe owner 到既有 ABI3 传输。
原生 C 源、ABI、候选二进制和 raw 格式不变。

## 固定条件与执行顺序

- Qwen3-8B，TP4/DP1/PP1，物理 NPU 4–7，seq128，microbatch1，seed42；
  LR horizon128，保存间隔4，4 MiB chunk，depth4，retention3。
- 只用既有登记 raw83 的 [256,1280) GiB 范围，绝不格式化。
- 新执行分支 codex/checkpoint-io-m1-closure-20261116 从 266bb35f 派生。
  源身份、候选二进制哈希、全部 E2 顺序在首个训练进程前登记；运行中禁止修改源或 HEAD。
- 默认原始目录 /dev/shm/user7-aic-aiv-io-closure-20261116；
  持久证据 /home/user7/npu-nvme-evidence/aic-aiv-io-20261116。
  每个进程前检查设备/lease、96 GiB tmpfs 和 8 GiB 持久空间。
- 顺序 E1 → E4 → E2/E3。E4 先行可尽早暴露本机 profiler 支持问题；
  它始终是独立进程，绝不参与正式性能统计。任一门槛失败即停止后续运行。

## E1：保存边界与正确性

T0/T1/T2 各自新进程到32步，再从 T2 step28 新进程恢复至32步。
比较 loss、最终全状态、控制态与全部新进程初始状态；对 T1/T2 的
4 ranks × 8 保存边界逐 chunk 比较顺序、tensor/offset、长度和 SHA256。

完整边界从 admission 前开始，到最终集体完成后返回。分别记录 admission/collectives、
framework synchronize、manifest read/hash、controls、sidecars、reservation/setup、
capture/transport、complete/publication/collective。详细模式在 manifest owning loop 聚合
read/hash 纳秒、tensor 数、逻辑字节，不逐 tensor 打印。

覆盖门槛同时检查 owning-call 阶段及细分关键路径：transfer 的笼统总跨度不能掩盖未知成本。
transfer 内只计 setup、prepared journal/barrier、protocol admission、每 chunk 的源读取至释放、
最终 commit wait 的关联区间；与其他阶段取并集。每个 rank、每次保存的未解释比例必须 ≤1%。
必需事件整对缺失、缺 rank/resume、缺 chunk release 或缺 flush/publication 都拒绝通过。
嵌套 owner/native 区间可以重叠，不能直接相加或当作设备服务时间。

## E2：预登记 A/B 扰动验收

A：必要边界、正确性和安全关闭记录；B：额外 chunk 事件、manifest 聚合细节、
native/API 统计、Host/HBM 采样。相同源与二进制。T0/T1/T2 × A/B ×3
独立训练进程到64步，共18次。每进程正式区间28..64，10个保存/对齐窗口，
每方法/模式合计30窗口才报告 nearest-rank P95。每进程10窗口的 P95 保持为空。

三轮方法顺序依次为 T0/T1/T2、T1/T2/T0、T2/T0/T1；各方法 A/B 次序按轮交替，
具体 schedule 写入 result.json。每次 T2 后立即执行对应 A/B 的 step60→64 新进程恢复，
以免下一 T2 轮换掉当前 raw generation；同轮 T0 就绪后再比较。共6次恢复。

9 对独立进程的 aligned-window median、正式 elapsed 的 B/A 均须 ≤1.10；
3组方法 pooled P95 的 B/A 也均须 ≤1.10。所有预登记样本都保留，无筛掉慢样本或补跑替换。

## E3：资源、计数器与安全关闭

B 在 supervisor 中以名义 1 Hz 采 /proc 的进程 RSS/内核 VmHWM 和 npu-smi 的
物理 NPU4–7 HBM。保存真实时间戳和最大采样间隔。采样最大值不是精确分配峰值；
HBM 是设备总占用，资源 admission ledger 则是单独的预算，不能互相替代。

快照复用既有 NPUNVMEStats。request_ring_depth/peak **只指 queued writes**，
不代表 copy/read/active 或全 native pending。额外统计 owning Python API 的提交、完成、
拒绝/EBUSY、submit/wait 纳秒和 copy 字节；记录 socket buffer、owner lock wait、
bridge ACK 区间。ACK 含对端处理与传输，不是纯 socket 等待。
全 native pending 高水位和 wake-notification 统计未由现有 ABI 暴露，明确列为 unavailable；
polling/query/retry/CPU counters 保持原生口径，不用占位数伪造。

每 rank 记录 startup、initial validation、Trainer 初始化、finetune、first/last step、
final validation、close。supervisor 另记 owner/worker launch、children exited 与 lease release。
正式吞吐与 supervisor 启动至 owner/rank 安全关闭的端到端吞吐分别报告；
后者包含 preflight、owner mount、compile、校验和 drain，token 数为实际执行步数×128。

通过必须同时有：rank/owner close 成功，成功提交与完成相等，NVMe/DMA/queued writes
均为0，无 retained buffers、子进程或硬件 lease。采样/统计错误传播为失败；
观测失败不跳过正常安全关闭，观测超时更不授权释放仍被设备借用的 buffer。

## E4：独立设备捕获与依赖限制

T0/T1/T2 各一次到32步。使用当前已安装 MindSpore Profiler 的 Level1、
PipeUtilization PMU，完整覆盖28..32回调（重点是28/32的完整保存窗口），
采 AscendCL、NPU tasks、MAC/vector/MTE。保存 host monotonic/realtime anchors、
原始 profiler 时钟元数据、导出的非空 device-task CSV 和正时长 trace。
只有请求 capability 或空导出不能通过。Profiler 数据、分析时间全部排除在 E2 外。

本组 PMU 不含独立 HBM bandwidth 指标；晚启动 GE allocation trace 也不声称支持。
导出中是否实际出现 PMU 列单独报告，不能用请求的选项代替测量。
原始时钟保留，未验证跨时钟映射前不推断设备引擎重叠。

可自动核实的归因边界是当前同步路径的必要依赖：framework synchronize 后捕获源，
capture/commit/publication 返回之前不能启动下一优化步。逐 rank 核对保存落在
optimizer-end→resume，且29步在28步保存结束后开始；同时检查 profiler 覆盖目标窗口。
这是当前控制流与源生命周期契约的限制，不是硬件吞吐下界、引擎全空闲或 NVMe 服务时间结论。
更细的 PMU 瓶颈分析保留原始导出供审查，不以 T2−T1 或公共时间桶冒充设备证据。

## 运行与证据

先运行：PYTHONPATH=python:. /usr/bin/python3 tools/checkpoint_io_closure.py preflight

冻结并推送执行分支后，以现有 raw-owner 权限运行：
PYTHONPATH=python:. /usr/bin/python3 tools/checkpoint_io_closure.py queue

result.json 持续记录 active_run、阶段、错误和各 gate。每个完成进程归档 JSON、
压缩事件/日志；E4 另外归档非 JSON profiler 文件和 SHA256 inventory。
转换权重逐文件验 SHA 后复用 hardlink，旧结果与模型不删除。
不自动进行 M2 调优、缩减 barrier、AIC/AIV co-run 或 KV offload。

## 冻结前验证

2026-11-16：tests/python、tests/d1、tests/d2、tests/c2 共355项通过，
包括缺失阶段/事件、关键路径覆盖、18进程 A/B/P95、native/API 计数、
观察者失败后的安全关闭、设备导出和新进程状态哈希相关检查。
其中 native ABI/symbol 检查只加载候选库，不初始化设备；不是硬件验收。
本轮未修改 C 源或二进制，因此未重复运行 C sanitizer。

启动前只读 preflight 通过：NPU4–7 空闲，tmpfs 约277.6 GiB、
持久目录约9.6 GiB 可用。硬件是否通过只以本轮 result.json 的完成门槛为准。
