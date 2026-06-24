# Claude 会话指令与上下文恢复

> 创建: 2026-06-17 | 最后更新: 2026-06-24 (Phase 0a/0b/0c 完成)
> 状态: **Step 3 全路径打通待开始**

---

## 一、项目核心信息

### 1.1 我是谁

- **项目**: NPU-NVMe 增量检查点系统 (硕士论文项目)
- **平台**: Ascend 910B NPU + SPDK 用户态 NVMe 驱动
- **核心叙事**: 三大创新点 (I1: SPDK 高 IOPS, I2: FaF 设备侧轮询同步, I3: Vector Engine 增量管线)

### 1.2 代码结构 (2026-06-24 更新)

```
python/
  direct_checkpoint.py   DirectCheckpoint 类 + 重新导出 (入口模块)
  c_bindings.py          C 库 ctypes 绑定 (lib, acl_lib, NPUNVMEContext)
  disk_layout.py         裸盘偏移常量
  chunk_helpers.py       build_chunks / build_chunks_host / build_ctypes_arrays
  delta_protocol.py      pack/unpack/apply delta + FileDeltaWriter
  noop_init.py           NoOpInitializer
  training_cell.py       ProbeTrainOneStepCell
  _legacy_compat.py      WaitProbe 遗留 (DEPRECATED, 勿用于新代码)
  profiler.py            SpdkProfiler (独立)
  export_model.py        模型导出工具
  format_npu_disk.py     磁盘格式化工具
  inspect_npu_disk.py    磁盘检查工具

experiments/
  common.py              共享测试工具 (make_gpt2xl_training, setup_faf_checkpointing,
                         make_ckpt, StepTimer, EpochTimer, init_env)

src/
  npu_nvme.c             C 引擎 (1354 行, 计划拆分为 7 模块)
  test_npu_nvme.c        测试

to_do/
  active/                当前活跃规划文件
  archive/               已完成的实验日志/设计文档 (~17 个)
```

### 1.2 关键约束

| 约束 | 值 |
|------|-----|
| 运行模式 | **sink=TRUE, GRAPH_MODE** (不可违背) |
| 模型 | **GPT-2 XL** (48L/1600d, 772 params, **3.12 GB FP16**) |
| 块大小 | **BLOCK_SIZE=524288 (512K elems)**, top_k=0.10 |
| MindSpore | 2.5 |
| MS 2.5 已知 bug | `tensor_scatter_update` INT8 GRAPH_MODE 编译失败; PYNATIVE `value_and_grad` + XL → NaN |
| Python | `/root/miniconda3/envs/ms_2.5/bin/python` |
| sudo 密码 | `CGCL_2025_#$` |
| NVMe PCIe | `0000:83:00.0` |
| NPU device | 1 |
| SPDK shm_id | 80 |

### 1.3 核心文件位置

```
python/direct_checkpoint.py     — DirectCheckpoint 主类 (1362行) + ProbeTrainOneStepCell
python/i3_delta_writer.py       — Delta 帧打包/解包 + FileDeltaWriter
src/npu_nvme.c                  — C 层 SPDK init + DMA + 大页 + Delta API (1622行)
include/npu_nvme.h              — C API 头文件
experiments/baselines/benchmark/          — Step 1: 基准测试 (1a/1b/1c)
experiments/baselines/step2_demo/         — Step 2: 图内量化可行性 demo
experiments/baselines/step2b_recovery_validation/  — Step 2b: 断点续传验证
experiments/baselines/step3_e2e/          — Step 3: 全路径打通 (待创建)
to_do/IMPLEMENTATION_PLAN.md    — 正式实现计划 (不可变部分)
to_do/STATUS_SUMMARY.md         — 项目进度总结 (最新)
to_do/CLAUDE_INSTRUCTIONS.md    — 本文件
```

---

## 二、用户偏好与约定

### 2.1 交互风格

- **每次回答结束**: 明确退出思考，给出总结和回答
- **每次开发后**: 更新 `to_do/` 记录进展
- **每次大更新后**: `git add -A && git commit -m "..."`，commit message 结尾加 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Git 权限问题**: 先 `sudo chown -R user7:user7 .git/objects` 再 commit

### 2.2 文档约定

- 文档放在 `to_do/` 下
- 进度状态用 emoji: ✅ 完成, 🔲 待做, 🔴 阻塞, 🔄 进行中, ⚠️ 有问题
- 所有 markdown 文档中引用文件使用 `[filename](relative/path)` 格式

### 2.3 命名与组织

- 实验脚本放在 `experiments/baselines/<purpose>/`
- 实验输出放在 `experiments/output/<purpose>/`
- 每个子目录放 `_run.sh` 供快速执行

---

## 三、当前开发进度

### 3.1 三步计划总览

| Step | 内容 | 状态 |
|------|------|:---:|
| Step 1 | 基准性能测试 (训练 + PMU + SPDK BW) | ✅ |
| Step 2 | 图内量化可行验证 (7项验证) | ✅ |
| Step 2b | 断点续传验证 (100步 NRMSE 曲线) | ✅ |
| Step 3 | 全路径打通 (FaF + FULL ckpt + delta chain) | 🔲 |

### 3.2 Step 1 基准数据速查

| 指标 | 值 | 来源 |
|------|:---:|------|
| GPT-2 XL 步时 (GRAPH_MODE, sink=1) | **468.3 ms** (±25.8, COV 5.5%) | Step 1a |
| GE kernel instances | **132,170** | Step 1a msprof |
| Cube MAC 设备级利用率 | **55.7%** | Step 1b |
| Vector ALU / idle | **12.4%** / **87.6%** | Step 1b |
| HBM 占用 | **35%** (22.9 GB / 64 GB) | Step 1a |
| SPDK FULL ckpt BW | **4412 MB/s** (2.90 GB / 674ms) | Step 1c |
| Model size FP16 | **3.12 GB** (772 params) | — |

### 3.3 Step 2 验证结果速查

| # | 验证项 | 结果 |
|:---:|------|:---:|
| V2.1 | GRAPH_MODE 编译 | ✅ |
| V2.2 | Delta norms | ⚠️ Top-K 重叠 86.8% (FP16 vs FP32 精度) |
| V2.3 | INT8 quant | ✅ max_abs_diff=1.0 (±1 INT8 bin) |
| V2.4 | P_old 图内更新 | ⚠️ MS 2.5 bug → host callback |
| V2.5 | HBM ptrs | ✅ 4 buffer 全部 valid |
| V2.7 | SPDK delta write | ✅ 159 MB / 45ms / 3350 MB/s |

### 3.4 Step 2b 结果速查

| 指标 | 值 |
|------|:---:|
| Step 1 NRMSE median | **0.003** |
| Step 100 NRMSE median | **0.76** |
| NRMSE drift | **+3.1e-3/step** |
| 根因 | top-10% 覆盖不足 (296/2969 blocks/step) |

### 3.5 关键设计决策 (不可变)

| # | 决策 | 日期 |
|:---:|------|:---:|
| D1 | 跨层 batched GE (concat all → [total_nb, BLOCK_SIZE]) | 2026-06-17 |
| D2 | Top-K 在图内 (ops.TopK()), CPU fallback | 2026-06-17 |
| D3 | P_old 在 HBM (INT8 Parameter, ~1.6 GB) | 2026-06-17 |
| D4 | Delta + quantize 在 optimizer 后 | 2026-06-17 |
| D5 | Delta write FaF 模式 | 2026-06-17 |
| D6 | FULL ckpt 同步阻塞在 epoch 边界 | 2026-06-17 |
| D7 | 恢复从 FULL ckpt (SPDK DMA) 开始 + delta chain | 2026-06-17 |

### 3.6 当前阻塞项

- MS 2.5 `tensor_scatter_update` INT8 GRAPH_MODE bug → P_old 更新降级到 host
- MS 2.5 PYNATIVE `value_and_grad` + GPT-2 XL → NaN → 必须使用 GRAPH_MODE
- 大页问题已解决 (`ensure_hugepages()`)
- SPDK delta write 64MB DMA buffer 限制已通过 `write_batch` 绕过

---

## 四、常用命令

```bash
# 环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH

# 构建 C 层
cd /home/user7/npu-nvme && bash build.sh

# 以 root 运行 Python 脚本 (通用模板)
echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
  /root/miniconda3/envs/ms_2.5/bin/python <script>.py'

PYTHON=/root/miniconda3/envs/ms_2.5/bin/python
REPO=/home/user7/npu-nvme

# Step 1a: 纯训练基准
bash $REPO/experiments/baselines/benchmark/_run.sh quick 50

# Step 1b: 设备级 PMU
bash $REPO/experiments/baselines/benchmark/_run_1b.sh 12 1

# Step 1c: SPDK FULL ckpt BW
bash $REPO/experiments/baselines/benchmark/_run_1c.sh 1

# Step 2: 图内量化 demo
bash $REPO/experiments/baselines/step2_demo/_run.sh 2 1

# Step 2b: 断点续传验证
bash $REPO/experiments/baselines/step2b_recovery_validation/_run.sh 100 1

# Git
cd $REPO && echo "CGCL_2025_#$" | sudo -S chown -R user7:user7 .git/objects && \
  git add -A && git commit -m "..."
```

---

## 五、相关文档索引

| 文档 | 说明 |
|------|------|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | 三步实现计划 (不可变部分，v2.1) |
| [STATUS_SUMMARY.md](STATUS_SUMMARY.md) | 项目进度总结 (最新) |
| [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) | 完整开发规划 |
| [PHASE2B_DESIGN.md](PHASE2B_DESIGN.md) | Phase 2b I3 增量管线设计 |
| [BASELINE_ANALYSIS_2026-06-17.md](BASELINE_ANALYSIS_2026-06-17.md) | 基准配置与数据质量分析 |
| [DELTA_PIPELINE_CPU_VS_NPU_ANALYSIS.md](DELTA_PIPELINE_CPU_VS_NPU_ANALYSIS.md) | CPU vs NPU 计算负载分析 |
