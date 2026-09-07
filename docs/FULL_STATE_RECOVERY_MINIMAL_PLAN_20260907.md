# 完整训练状态恢复对比：最小执行记录

本轮只比较 GPT-2 完整训练状态恢复，不修改 PPT，不扩展增量、多卡或长程压测。

## 固定范围

- 方法：`ours` Raw SPDK/NVMe、`mindspore_native_save`、`bytecheckpoint_host`。
- 状态：模型、Adam m/v、global step、data cursor、RNG/control payload。
- 配置：seed 41、batch 1、129 tokens、dropout 0；复用既有 prepare fixture 和 batches。
- 每个可用方法 5 个全新恢复进程用于计时，另 1 个进程做 byte-exact 与续训 3 步验证。
- 方法串行执行；不清理全局 page cache，不格式化磁盘，不重绑定驱动。

## 存储与资格门禁

`checkpoint_inventory.json` 必须记录 runner HEAD、dirty 状态、fixture/batches/schema 哈希、generation/step/bytes、文件或裸盘端点及 NVMe 序列号。

`/models` 当前位于 PCI `84:00.0`，Raw SPDK 端点为 `83:00.0`；在 `same_physical_storage_verified=false` 时只报告恢复参考，不计算严格跨端点加速比。Ours 必须通过真实初始化和代际读取；attach 失败时写入 `failure.json`，不得回退到文件系统。

## 统一计时

恢复进程输出到不覆盖的 `restore_timing/rep_XX.json`，记录：

`process_start → model_constructed → restore_begin → metadata_ready → read_deserialize_done → state_ready → first_step_begin/end`。

主指标为 `state_ready - restore_begin`；读取/反序列化、状态装回、首步代价分别报告。首步若包含图编译必须标记。正确性结果单独写 `restore_verify.json`，不被计时重复覆盖。

保存事件统一记录 `save_request`、`source_read_done`、`data_durable`、`generation_committed`；必要的 native oracle、文件 fsync、bridge metadata 和 Raw SPDK flush 不从正式语义中删除。

## 执行与停止条件

先执行静态设备/挂载核对和 inventory，再做 Native/ByteCheckpoint timing smoke 与 5 次正式恢复，随后执行各自独立验证。Ours 仅在 runtime attach 成功后加入同样矩阵。

任意方法若 generation、step、schema、字节规模或逐字段校验失败，停止该方法计时并保留原因。5 次重复只汇报原始值、均值、中位数和 min–max，不计算 P99 或强置信区间。

结果、汇总和实现变更提交到 `codex/baseline-gpt2-repro` 并推送 `origin/codex/baseline-gpt2-repro`。
