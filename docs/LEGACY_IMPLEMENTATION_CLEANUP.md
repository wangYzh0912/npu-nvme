# 旧实现全链路清理

入口：`ed43b2e`；历史标签：`archive-exploratory-implementations-20260914`。清理工作树：`/models/npu_nvme_exp/user7-stack/checkouts/legacy-cleanup`。原始工作树未改动，32 份唯一工作文件另存档；路径、身份和哈希见 `results/legacy-cleanup/entry.json`。

## 已批准的策略

保留未来能力目标，删除旧探索实现；每种 baseline 保留一套；删除旧 API/导入/配置，不留转发或报错壳。活动实现的完整删除路径及入口版本 SHA256 见 `results/legacy-cleanup/removed-files.json`，C 接口去留见 `native-retirement.json`。文件移动后的有效实现继续被当前测试调用，历史通过数量不作为新验收门槛。

| 清单编号 | 执行内容 |
| --- | --- |
| P01–P06 | 删除动态 facade、根目录兼容别名、eager FFI 入口、旧 save/load/live 壳和固定参数；严格保存必填 expected_spec；统一 drain |
| P07–P09 | 删除旧 Handle、scheduler、leases、cleanup 和 LegacyBatchTransport；Busy 错误迁入当前 errors 模块 |
| P10–P12 | 删除 LiveCapture、probe 训练与 split live cells；保留普通标量检查训练；capture 仅保留完整态冻结路径，必须框架同步 |
| P13–P16 | 删除旧 profiler/noop/export 壳、旧工厂和无效配额文件；receipts 归核心，仍被 baseline 使用的状态机迁入 baseline 子系统 |
| N01–N06 | 删除 listener/poller/FaF FSM 分支、Delta bookkeeping、旧 CRC、死字段和重复公共声明；隐藏项目内部符号 |
| N07 | ABI 2，公开版本查询；必需 FFI 符号和 ABI 在打开设备前校验；ACL 文件 baseline 不加载 SPDK |
| N08–N10 | 删除旧分槽 helper，保留 V2 序列化几何；格式化必须 flush；检查工具复用严格目录校验并返回正确错误码 |
| E01–E05 | 删除 Delta/S2/R0/frame/ring/multirank 探索算法和专用测试；Qwen 原生 TP4/EN 保留 |
| E06–E13 | 单卡与 baseline 共用 train.py 和训练 fixture；删除旧 live/IO3/IO4/INC2/S3/campaign/PMU 入口；raw 传输收敛到 transport_matrix |
| E14–E17 | 删除 mindspore_sync/原始字节伪原生 adapter；迁出 two_phase_common 有效 ACL 操作；移除历史源码锁和报告推断 |
| T01–T05 | 将安全测试迁到当前运行时；接口退役检查变为不存在；ring 真正边界测试并入 sanitizer harness；旧硬件执行器归档 |
| T06–T08 | 历史 A/B profiles 归档；CLEANUP profile、完整依赖检查、失败传播与身份绑定；文档接入唯一入口 |
| T09–T11 | 原工作树唯一工作归档，历史 baseline/I/O 工作树和原始失败证据保留；不物理删除环境或历史结果 |

## 当前接口与方法

Python：`from npu_nvme import StrictCheckpoint`，保存 `save_state(..., expected_spec=...)`，恢复 `restore_full_state(target_factory, expected_spec)`，显式查询/统计/drain/close。无旧模块别名。

CLI：`train.py preflight/fit/benchmark/verify-restart/inspect`。`_prepare/_source` 为隔离进程内部阶段。默认冻结捕获，实际 transport 如实记录；训练等待不等于底层异步能力。配置 schema 2；公共 workload 与 `methods.ours` 等实际参数明确分离。最终训练步必须创建 checkpoint，独立恢复必须有完整源续训 oracle。timing 只返回 timing_measured，不报告正确性通过。

方法：none、mindspore_native_save、ours、datastates_acl、pccheck_acl、bytecheckpoint_host、fastpersist_host。DataStates/PCcheck 是语义移植；ByteCheckpoint/FastPersist 保留锁定 upstream workers，不改变其方法机制。

## 保留和延期项

当前 C Host/device batch、metadata、wait/flush、事件失败后的安全同步、quarantine、reader pin、保留代保护、admission/close 锁、SPDK drain 继续保留。B2 按 D2H/H2D/Host/metadata 分方向验证 H07，C2 再删独立同步 bulk 和旧阻塞 API。V2 Delta 几何是已序列化保留区，不因删除 Delta 实现而改变。旧与候选环境保留到独立 E0/EN 验收。

## 验收与复现

构建使用 `EXECUTION_ENVIRONMENT_AND_COMMANDS.md` 的已锁定旧 CANN/SPDK。先运行 `tools/run_gate.py --profile CLEANUP --out <new-directory>`，再以同一冻结源码/库执行 D1 H01 seeds 41/42/43、H02 11 类、生命周期五组含 32 reopen、native smoke、transport_matrix 和七种方法独立进程训练/恢复。仅 83:00.0 可裸写。每轮单独结果目录，保留失败，禁止 skip 或修改容差掩盖失败。

验收文件由 `tools/validate_cleanup_acceptance.py` 连接软件、硬件、方法和动态符号证据；全部成功前状态保持 validating。源树、库与 fixture 不允许混用。回滚通过归档版本和匹配库执行，活动分支不保留 fallback。
