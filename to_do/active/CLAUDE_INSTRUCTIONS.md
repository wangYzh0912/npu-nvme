# Claude 会话指令与上下文恢复

> 创建: 2026-06-17 | 最后更新: 2026-06-25 (DeltaTrainCell 模块完成, T1-T3 通过, T4 调试中)
> 状态: **E2E 测试 T4 阻塞修复中 → T5/T6 → 论文实验**

---

## 零、当前会话任务 (2026-06-25)

1. ~~服务器验证~~ ✅ 全部通过
2. ~~DeltaTrainCell 模块~~ ✅ 已创建 (delta_cell.py + direct_checkpoint.py + common.py)
3. E2E 测试 T1-T3 通过, T4 阻塞修复中 (ms.Model.train 形状问题)
4. 论文实验待推进

### 重要: 2026-06-24 验证结果

| 测试 | 结果 | 说明 |
|------|:---:|------|
| C 编译 (with HAS_NPU) | ✅ | 2 warnings (deprecated API) |
| Pure-logic tests | **10/10** | ring buffer, ALIGN_4K, constants |
| Hardware integration tests | **15/15** | init/cleanup/capacity/max_transfer/meta_io/delta |
| A.5 Python imports | ✅ | 全部模块导入成功 |
| A.6 FULL ckpt roundtrip | ✅ | save→load→cleanup 通过 |
| A.7 Delta >64MB roundtrip | ✅ | 150MB × 300 blocks 逐字节验证 |
| Step 1c SPDK BW | **3790 MB/s** | 2.90 GB / 784ms, pipeline_depth=8 |

修复的 Bug: verify_phaseA.sh import, chunk_size 类型统一, probe_cb 返回类型, pipeline.h SPDK 前向声明, c_bindings 模块名, delta read_batch_host

**Python**:
- `direct_checkpoint.py` 拆分为 8 个模块。所有旧 import 通过重新导出保持兼容。
- `delta_save()` / `delta_load_slot()` 不再用 `npu_nvme_write_delta` (删除)，改用 ring buffer pipeline。
- `export_model.py` 写路径修正为 `write_batch_host`。
- 新增 `experiments/common.py`: `make_gpt2xl_training`, `setup_faf_checkpointing`, `make_ckpt`, `StepTimer`, `EpochTimer`, `init_env`。

**C**:
- `npu_nvme.c` 使用 `include/internal/` 头文件。子结构体: `ctx->acl.*`, `ctx->dma.*`, `ctx->listener.*`, `ctx->delta.*`。
- 公共 API 17 函数 (删除 `write_delta`/`read_delta`)。
- `meta_dma_buf` 从 64MB 缩小到 1MB。Profiling CSV 统一为 `write_profiling_csv()`。
- 修复: 双重 `if (enable_profiling)` bug (read_batch line 1198)。

**验证清单** (服务器上):
```
git pull && bash scripts/verify_phaseA.sh
# A.1-A.7: 编译 → 测试 → Python 导入 → FULL roundtrip → Delta >64MB
# 修改前分支保存在 delta 分支上
```

## 一、项目核心信息

### 1.1 项目身份

- **项目**: NPU-NVMe 增量检查点系统 (硕士论文项目)
- **平台**: Ascend 910B NPU + SPDK 用户态 NVMe 驱动
- **核心叙事**: 三大创新点 (I1: SPDK, I2: FaF 设备侧轮询, I3: Vector Engine 增量管线)

### 1.2 代码结构 (2026-06-27)

```
python/
  direct_checkpoint.py   1049 行  DirectCheckpoint + 重新导出 (入口)
  c_bindings.py          131 行  C 库 ctypes
  disk_layout.py          28 行  裸盘偏移常量
  chunk_helpers.py        135 行  build_chunks / build_ctypes_arrays
  delta_protocol.py       214 行  pack/unpack/apply delta + FileDeltaWriter
  noop_init.py            40 行  NoOpInitializer
  training_cell.py        62 行  ProbeTrainOneStepCell
  _legacy_compat.py       43 行  WaitProbe 遗留 (DEPRECATED)
  profiler.py             253 行  SpdkProfiler
  export_model.py         工具
  format_npu_disk.py      工具
  inspect_npu_disk.py     工具

src/
  npu_nvme.c             894 行  C 引擎 (重构后)
  test_npu_nvme.c         271 行  测试
include/
  npu_nvme.h             139 行  17 公共 API
  internal/                4 个内部头文件 (93+57+49+55 行)

experiments/
  common.py              173 行  4 共享函数

to_do/
  active/                 4 个活跃文件
  archive/               23 个归档文档
```

### 1.3 关键约束

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

### 1.4 核心文件位置 (2026-06-24 重构后)

```
python/direct_checkpoint.py     DirectCheckpoint 类 (~800 行) + 重新导出 (入口)
python/disk_layout.py           裸盘偏移常量
python/c_bindings.py            C 库 ctypes 绑定
python/chunk_helpers.py         build_chunks / build_chunks_host / build_ctypes_arrays
python/delta_protocol.py        pack/unpack/apply delta + FileDeltaWriter
python/noop_init.py             NoOpInitializer 快速初始化
python/training_cell.py         ProbeTrainOneStepCell (FaF step_counter)
python/_legacy_compat.py        DEPRECATED: WaitProbe 遗留符号
python/profiler.py              SpdkProfiler 统一性能监控

src/npu_nvme.c                  C 引擎 (894 行, 重构后 -460 行)
src/test_npu_nvme.c             测试 (279 行)
include/npu_nvme.h              公共 API (16 函数)
include/internal/               内部头文件 (ring_buffer/io_task/pipeline/context)

experiments/common.py           共享测试工具 (5 函数)
scripts/verify_phaseA.sh        Phase A 服务器验证脚本

to_do/active/
  IMPLEMENTATION_PLAN.md        合并版实现计划 (Step 3 + E0-E3 实验)
  ARCHITECTURE_ROADMAP.md       代码结构速览
  STATUS_SUMMARY.md             进度摘要
  CLAUDE_INSTRUCTIONS.md        本文件
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

### 3.6 已解决/已知限制

| 项目 | 状态 | 说明 |
|------|:---:|------|
| SPDK delta 64MB 限制 | ✅ 已解决 | Phase 0a: `delta_save()` 改用 `build_chunks_host` + `write_batch_host` |
| SPDK 大页泄漏 | ✅ 已解决 | Phase 0: `close()` 默认值 + `_spdk_initialized` 守卫 |
| `lib` undefined (无 .so) | ✅ 已解决 | Phase 0 fix: `except` 块中 `lib = None` |
| `delta_save()` 悬垂指针 | ✅ 已解决 | Phase 0 fix: 命名 `frame_buf` 变量 |
| 双重 `if (enable_profiling)` | ✅ 已解决 | Phase 2: 统一 `write_profiling_csv()` |
| P_old 图内更新 (ScatterUpdate) | ⚠️ MS 2.5 性能 bug | 改用全量 Assign。ScatterUpdate(INT8) 在 Ascend 910B 上回退到 AICPU 慢路径，比 Full-Assign 慢 2.1× (2087ms vs 4301ms)。MS 2.6 缺 TBE 无法测试。5 条优化方案已记录 |
| PYNATIVE `value_and_grad` NaN | ⚠️ MS 2.5 bug | 必须使用 GRAPH_MODE |
| C 编译 (no SPDK/NPU on dev) | ⚠️ 待服务器验证 | `scripts/verify_phaseA.sh` |
| ScatterUpdate INT8 优化方案 | 🔲 待测试 | A: Parameter setitem; B: masked_scatter; C: FP16 scatter+Cast; D: CANN fusion pass; E: mint.scatter |
| `nn.AdamWeightDecay` → `AdamW` | 🔲 延期 | MS 2.9+ 强制迁移

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

## 五、文档索引

### active/ (当前活跃)

| 文档 | 说明 |
|------|------|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | **合并版实现计划** — Step 1/2/3 + E0-E3 论文实验矩阵 |
| [ARCHITECTURE_ROADMAP.md](ARCHITECTURE_ROADMAP.md) | 代码结构速览 + API 清单 |
| [STATUS_SUMMARY.md](STATUS_SUMMARY.md) | 项目进度摘要 (最新) |
| [CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md) | 本文件 — 会话恢复 + 环境信息 |

### archive/ (参考)

| 文档 | 说明 |
|------|------|
| `archive/I1_SPDK_plan.md` | SPDK 模块设计与 E1 实验 |
| `archive/I2_FaF_plan.md` | FaF 监听器设计与 E2 实验 |
| `archive/I3_VectorEngine_plan.md` | I3 增量管线设计与 E3 实验 |
| `archive/PHASE5_FINAL_PLAN.md` | 实验优先级与执行顺序 |
| `archive/DELTA_PIPELINE_CPU_VS_NPU_ANALYSIS.md` | CPU vs NPU 计算负载分析 |

### 根目录

| 文档 | 说明 |
|------|------|
| `C:\Users\wyf\.claude\plans\...` | **完整架构演进路线图** (Claude plan file) |
| `scripts/verify_phaseA.sh` | Phase A 服务器验证脚本 |
| `STYLE_GUIDE.md` | 代码规范 |
