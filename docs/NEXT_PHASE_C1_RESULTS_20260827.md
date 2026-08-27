# C1 完整训练态正确性结果（2026-08-27）

## 环境

- 分支：`exp/full-state-correctness`
- MindSpore：`ms_2.5`；Ascend 910B3；NPU 0
- 目标盘：`0000:83:00.0`，Huawei ES3000 V6，SPDK/uio 路径
- 运行前 `npu-smi info`：8 张 NPU 均无进程占用

## 结果

GPT2 小模型短序列 2+2 step 首轮回环通过严格门限：加载瞬间 590 个字段、
1,485,020,168 bytes 全部字节一致；续训 loss 和终态逐张量
`rtol=1e-5, atol=1e-6` 通过。

GPT2-XL、序列长度 1025、保存 step 10、续训 10 step 结果：

- 完整状态 2318 个字段，共 9,839,827,208 bytes；
- generation 6 成功发布；SPDK payload 写入 9384 MiB，硬件写阶段约 2.378 s；
- 精简 JSON metadata 仍超 400 KiB，切换 metadata v2 + zlib 后提交约 0.547 s，成功落盘；
- fresh process 加载后控制态、模型和 Adam 状态通过 checksum；
- 续训 loss 通过 `rtol=1e-5, atol=1e-6`；
- 终态在 `rtol=1e-4, atol=1e-5` 下逐张量 allclose，通过；2318 字段中 2134 个字节一致；
- 严格终态 `rtol=1e-5, atol=1e-6` 未通过，最大绝对误差约 `2.54e-5`，表明 Ascend 跨进程图执行存在累积数值非确定性；
- GPT2 默认 dropout 还会引入更大的随机差异，C1 正式门禁因此使用 dropout=0，并保留 MindSpore RNG 状态字段。

原始机器可读结果位于
`results/next-correctness/c1-gpt2xl-10p10-20260827/result.json`；该目录中的
约 10 GiB oracle 不纳入 Git。

## 代码变化

- `DirectCheckpoint.save_state/load_state` 支持 model、optimizer、控制态命名空间；
- 控制态使用 JSON-tagged v1 编码，不依赖 pickle，并带 SHA-256；
- metadata v2 使用 zlib，读取端兼容 metadata v1；
- C1 使用独立 baseline/save/restore 进程验证 warmup、保存、退出、加载和续训。

下一步是将 C1 的故障注入和严格结果收口，再进入 C2 两卡实际训练的分片提交与恢复；
当前仍未实现真正 `aclrtMemcpyAsync` 路径。
