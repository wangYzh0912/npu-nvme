# Delta Checkpoint — 全路径打通详细设计

> 2026-06-24 | 增量检查点 (delta checkpoint) 全路径实现设计

---

## 一、架构目标

将图内 delta 检测、Top-K 选择、INT8 量化、P_old 管理、双量化路径、FaF 异步持久化整合为可复用的模块，并通过端到端测试验证完整链路。

### 数据流

```
GE Graph (DeltaTrainCell.construct)   C Listener (10ms poll)         Python (epoch边界)
┌─────────────────────────────────┐   ┌──────────────────────┐       ┌───────────────────┐
│ Phase A: fwd→bwd→optimizer     │   │ poll step_counter    │       │ wait_for_io()     │
│ Phase B: Concat→AllBlocks fp16 │   │ 检测 step↑ →         │       │ FULL ckpt sync    │
│ Phase C: delta=W-P_old, norms  │   │ run_write_pipeline   │       │                   │
│ Phase D: TopK(norms, 296)      │   │  HBM→NVMe DMA         │       │                   │
│ Phase E: Gather→INT8 quant     │ ─→ │ signal_probe_flag    │       │                   │
│ Phase F: P_old全量INT8 Assign  │   └──────────────────────┘       └───────────────────┘
│ Phase G: quant/scales/idx Assign│                                          
│          AssignAdd(step_counter)│
└─────────────────────────────────┘
```

---

## 二、已实现模块 (可直接复用)

| 模块 | 位置 | 关键接口 | 用途 |
|------|------|------|------|
| DirectCheckpoint | [python/direct_checkpoint.py](python/direct_checkpoint.py) | `__init__`, `save`, `load`, `delta_init`, `delta_save`, `delta_load_slot`, `delta_load_chain`, `recover`, `register_tasks` | SPDK I/O 管理 |
| ProbeTrainOneStepCell | [python/training_cell.py](python/training_cell.py) | `__init__(network, optimizer, enable_probe, ckpt_interval)`, `construct(*inputs)` | FaF step_counter 注入 |
| delta_protocol | [python/delta_protocol.py](python/delta_protocol.py) | `pack_delta_frame(step_id, blocks, smalls) → bytes`, `unpack_delta_frame(bytes) → (sid, blocks, smalls)`, `apply_delta_patches(weights, blocks, smalls, block_size) → dict` | Delta 序列化/反序列化/apply |
| chunk_helpers | [python/chunk_helpers.py](python/chunk_helpers.py) | `build_chunks(params, chunk_size)`, `build_chunks_host`, `build_ctypes_arrays`, `rebuild_chunks_from_meta` | 分块+ctypes 数组 |
| c_bindings | [python/c_bindings.py](python/c_bindings.py) | `lib`, `acl_lib`, `NPUNVMEContext` | C 库 ctypes 绑定 |
| get_dev_ptr | [python/direct_checkpoint.py:66](python/direct_checkpoint.py#L66) | `get_dev_ptr(tensor) → int` | 获取 HBM 设备指针 |
| NoOpInitializer | [python/noop_init.py](python/noop_init.py) | `NoOpInitializer`, `replace_with_noop_initializer(model)` | 快速参数初始化 |
| common | [experiments/common.py](experiments/common.py) | `make_gpt2xl_training`, `make_ckpt`, `setup_faf_checkpointing`, `init_env` | 训练环境工厂函数 |
| FaF listener (C) | [src/npu_nvme.c:632](src/npu_nvme.c#L632) | `probe_listener_thread`, `npu_nvme_register_tasks(ctx, ptrs, offs, sizes, n)`, `npu_nvme_set_step_ptr(ctx, dev_ptr, ckpt_interval)` | 后台轮询+DMA |
| C API 19 函数 | [include/npu_nvme.h](include/npu_nvme.h) | init/cleanup/query/sync/batch/FaF/delta | 完整 C 接口 |

---

## 三、新增模块详细设计

### 3.1 `python/delta_cell.py` — 核心模块 (~250 lines)

#### 3.1.1 模块级辅助函数

```python
def analyze_model_layers(model: nn.Cell,
                         block_size: int = 524288,
                         small_threshold: int = 10000) -> dict:
    """分析 GPT-2 XL 模型层结构，返回 DeltaTrainCell 构造所需的元数据。

    对每个可训练参数按正则 `backbone\.blocks\.(\d+)\.` 提取 layer_id，
    按 small_threshold 分类为大参数 (分 block) 和小参数 (整体存)。

    返回值 (dict):
      total_elems_large: int       # 大参数总元素数
      padded_elems: int            # 补齐到 block_size 倍数后的总元素数
      total_nb: int                # 总 block 数 = padded_elems // block_size
      block_params: list[Parameter]   # 大参数的 MindSpore Parameter 列表
      block_nelem: list[int]          # 每个参数的元素数
      fp16_needed: list[bool]         # 每个参数是否需要 Cast 到 fp16
      small_params: list[(Parameter, str, int)]  # 小参数列表
      layer_ids: list[int]           # 所有层 ID
    """
```

此函数直接从 `experiments/baselines/step2_demo/step2_demo.py` 中的 `analyze_model()` 提取和泛化。

#### 3.1.2 `DeltaTrainCell(nn.Cell)`

```python
class DeltaTrainCell(nn.Cell):
    """增量检查点训练 Cell。

    construct() 在 GE 图中执行完整的 7 阶段流水线:
      Phase A: forward + backward + optimizer
      Phase B: 跨层参数聚合 (Concat → pad → Reshape → AllBlocks fp16)
               P_old INT8 反量化为 fp16 (Cast)
      Phase C: Delta norms = ReduceSum((W - P_old)²) per block
      Phase D: Top-K 选择 (ops.TopK)
      Phase E: 输出量化路径 (Gather → INT8 quant)
      Phase F: P_old 全量 INT8 更新 (Assign, 非 ScatterUpdate)
      Phase G: Assign 输出缓冲区 + AssignAdd(step_counter)

    输出 Parameter (需注册到 FaF listener):
      delta_quant_buf:  [top_k * block_size] INT8  (~148 MB for GPT-2 XL)
      delta_scale_buf:  [top_k] FP32                (~1.2 KB)
      delta_idx_buf:    [top_k] INT32               (~1.2 KB)
      delta_p_old:      [total_nb * block_size] INT8 (~1.52 GB)
      step_counter:      scalar INT32
    """

    def __init__(self,
                 network: nn.Cell,
                 optimizer: nn.Optimizer,
                 block_size: int = 524288,
                 top_k_frac: float = 0.10,
                 small_threshold: int = 10000):
        """
        Args:
            network: MindSpore 模型 (GPT-2 XL)
            optimizer: MindSpore 优化器
            block_size: 每个 block 的元素数 (512K elems = 1MB FP16 = 512KB INT8)
            top_k_frac: Top-K 选择比例 (0.10 = 10%)
            small_threshold: 小参数阈值 (元素数)
        """
        super().__init__(auto_prefix=False)

        # 分析模型结构
        info = analyze_model_layers(network, block_size, small_threshold)

        # 训练组件
        self.net = network
        self.net.set_grad()
        self.opt = optimizer
        self.grad_fn = ops.value_and_grad(
            self.net, grad_position=None, weights=self.opt.parameters)

        # 块元数据 (GE 图中不可变的固定属性)
        self.nb = info['total_nb']
        self.bs = block_size
        self.k = max(1, int(info['total_nb'] * top_k_frac))
        self.te = info['total_elems_large']
        self.pe = info['padded_elems']
        self.n_params = len(info['block_params'])
        self.block_params = tuple(info['block_params'])
        self.block_nelem = tuple(info['block_nelem'])
        self.fp16_needed = tuple(info['fp16_needed'])

        # 输出缓冲区 (HBM Parameter)
        self.delta_quant_buf = Parameter(
            Tensor(np.zeros(self.k * block_size, dtype=np.int8)),
            name="delta_quant_buf", requires_grad=False)
        self.delta_scale_buf = Parameter(
            Tensor(np.zeros(self.k, dtype=np.float32)),
            name="delta_scale_buf", requires_grad=False)
        self.delta_idx_buf = Parameter(
            Tensor(np.zeros(self.k, dtype=np.int32)),
            name="delta_idx_buf", requires_grad=False)

        # P_old INT8 存储 (~1.52 GB for GPT-2 XL)
        self.delta_p_old = Parameter(
            Tensor(np.zeros(info['total_nb'] * block_size, dtype=np.int8)),
            name="delta_p_old", requires_grad=False)

        # Step counter
        self.step_counter = Parameter(
            Tensor([0], dtype=ms.int32),
            name="step_counter", requires_grad=False)
        self.one = Tensor([1], dtype=ms.int32)

    def _int8_quantize(self, blocks_fp16: Tensor) -> (Tensor, Tensor):
        """INT8 量化。

        Args:
            blocks_fp16: [n, bs] FP16 Tensor

        Returns:
            (int8_blocks [n, bs]: INT8 Tensor, scales [n]: FP32 Tensor)
        """
        n = blocks_fp16.shape[0]
        blocks_fp32 = ops.Cast()(blocks_fp16, ms.float32)
        abs_max = ops.ReduceMax()(ops.Abs()(blocks_fp32), 1)
        scales = ops.Div()(abs_max, Tensor(127.0, ms.float32))
        scales_2d = ops.Reshape()(scales, (n, 1))
        scaled = ops.Div()(blocks_fp32, scales_2d)
        quant_int8 = ops.Cast()(
            ops.clip_by_value(
                ops.Round()(scaled),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)),
            ms.int8)
        return quant_int8, scales

    def construct(self, *inputs) -> Tensor:
        """GE 图主函数。

        Returns:
            loss (通过 ops.Depend 链接到所有副作用操作)
        """
        # ═══ Phase A: 标准训练 ═══
        loss, grads = self.grad_fn(*inputs)
        loss = ops.Depend()(loss, self.opt(grads))

        # ═══ Phase B: 跨层参数聚合 + P_old 反量化 ═══
        flat_parts = []
        for i in range(self.n_params):
            p = self.block_params[i]
            ne = self.block_nelem[i]
            pv = ops.Cast()(p, ms.float16) if self.fp16_needed[i] else p
            flat_parts.append(ops.Reshape()(pv, (ne,)))
        all_flat = ops.Concat()(tuple(flat_parts))

        pad_amt = self.pe - self.te
        all_flat_padded = ops.pad(all_flat, (0, pad_amt),
                                   mode='constant', value=0.0)
        AllBlocks = ops.Reshape()(all_flat_padded, (self.nb, self.bs))

        # P_old 反量化: INT8 → FP16
        P_old_int8_2d = ops.Reshape()(self.delta_p_old, (self.nb, self.bs))
        P_old_fp16 = ops.Cast()(P_old_int8_2d, ms.float16)

        # ═══ Phase C: Delta norms ═══
        deltas = ops.Sub()(AllBlocks, P_old_fp16)
        delta_sq = ops.Mul()(deltas, deltas)
        norms = ops.ReduceSum()(delta_sq, 1)
        norms_fp32 = ops.Cast()(norms, ms.float32)

        # ═══ Phase D: Top-K 选择 ═══
        _, top_indices = ops.TopK(sorted=True)(norms_fp32, self.k)

        # ═══ Phase E: 输出量化路径 (仅 Top-K blocks) ═══
        selected_fp16 = ops.Gather()(AllBlocks, top_indices, 0)
        quant_int8, scales = self._int8_quantize(selected_fp16)

        # ═══ Phase F: P_old 全量 INT8 更新 ═══
        new_p_old_int8, _ = self._int8_quantize(AllBlocks)
        new_p_old_flat = ops.Reshape()(new_p_old_int8, (self.nb * self.bs,))
        self.delta_p_old = ops.Assign()(self.delta_p_old, new_p_old_flat)

        # ═══ Phase G: 输出 Assign + step_counter ═══
        self.delta_quant_buf = ops.Assign()(
            self.delta_quant_buf,
            ops.Reshape()(quant_int8, (self.k * self.bs,)))
        self.delta_scale_buf = ops.Assign()(self.delta_scale_buf, scales)
        self.delta_idx_buf = ops.Assign()(self.delta_idx_buf, top_indices)
        self.step_counter = ops.AssignAdd()(self.step_counter, self.one)

        # Depend 链: 防止 GE 图 DCE
        loss = ops.Depend()(loss, self.delta_quant_buf)
        loss = ops.Depend()(loss, self.delta_scale_buf)
        loss = ops.Depend()(loss, self.delta_idx_buf)
        loss = ops.Depend()(loss, self.delta_p_old)
        loss = ops.Depend()(loss, self.step_counter)
        return loss


__all__ = ['DeltaTrainCell', 'analyze_model_layers']
```

### 3.2 `python/direct_checkpoint.py` — 追加 2 个方法

#### 3.2.1 `build_layout_for_delta(self, delta_cell) → list[dict]`

```python
def build_layout_for_delta(self, delta_cell):
    """为 DeltaTrainCell 的输出缓冲区构建 NVMe 布局。

    将 delta_quant_buf, delta_scale_buf, delta_idx_buf 的 HBM 指针
    和 NVMe delta area 偏移量打包为 chunks 列表，供 FaF listener 使用。

    Returns:
        list[dict] — chunk 列表
    """
```

#### 3.2.2 `register_delta_tasks(self, delta_cell)`

```python
def register_delta_tasks(self, delta_cell):
    """注册 DeltaTrainCell 输出缓冲区到 C 层 FaF listener。

    调用 build_layout_for_delta 构建布局，
    打包 ctypes 数组: c_ptrs, c_offs, c_sizes，
    调用 lib.npu_nvme_register_tasks。

    Returns:
        (dev_flag: int, dev_step: int) — probe flag 和 step_counter 的 HBM 地址
    """
```

### 3.3 `experiments/common.py` — 追加 2 个函数

#### 3.3.1 `setup_delta_faf(ckpt, delta_cell, ckpt_interval=5) → (int, int)`

```python
def setup_delta_faf(ckpt: DirectCheckpoint, delta_cell: DeltaTrainCell,
                    ckpt_interval: int = 5) -> tuple:
    """FaF listener 全接线 + delta area 初始化。

    1. ckpt.register_delta_tasks(delta_cell)
    2. 获取 dev_flag, dev_step 指针
    3. lib.npu_nvme_set_probe_flag_ptr(ctx, dev_flag)
    4. lib.npu_nvme_set_step_ptr(ctx, dev_step, ckpt_interval)
    5. ckpt.delta_init(slot_size_mb=256, slot_count=128)

    Returns:
        (dev_flag: int, dev_step: int)
    """
```

#### 3.3.2 `make_delta_training(total_steps=20, device_id=1, ...) → tuple`

```python
def make_delta_training(total_steps: int = 20,
                        device_id: int = 1,
                        seq_len: int = 1024,
                        block_size: int = 524288,
                        top_k_frac: float = 0.10,
                        ckpt_interval: int = 5,
                        pipeline_depth: int = 8,
                        profiling: bool = False) -> tuple:
    """创建完整的增量检查点训练环境。

    Returns:
        (model, dataset, optimizer, delta_cell, ckpt)
    """
```

### 3.4 `experiments/delta_e2e/` — 端到端测试

```
experiments/delta_e2e/
  delta_e2e.py     # 主测试脚本 (~350 lines)
  _run.sh          # Shell wrapper
```

#### 测试流程 (6 项验证)

```python
def test_compile(device_id):
    """验证: GRAPH_MODE 编译 DeltaTrainCell 无错误，不 OOM。"""

def test_single_step(device_id):
    """验证: 单步后 P_old 非零，quant_buf 非零。"""

def test_faf_register(device_id):
    """验证: register_delta_tasks + set_step_ptr 返回 0。"""

def test_faf_trigger(device_id, steps=5):
    """验证: 多步运行后 probe_flag 与预期一致。"""

def test_overhead(device_id, steps=50):
    """验证: DeltaTrainCell overhead < 5% vs baseline。"""

def test_recovery(device_id, steps=10):
    """验证: FULL ckpt + delta chain 恢复 NRMSE median < 0.05。"""
```

---

## 四、验证标准

| # | 测试 | 通过标准 | 验证方法 |
|:---:|------|------|------|
| 1 | GRAPH_MODE 编译 | 无编译错误, 不 OOM | `cell(dummy)` 无异常 |
| 2 | 单步 E2E | `abs(delta_p_old.sum()) > 0`, `abs(delta_quant_buf.sum()) > 0` | `.value().asnumpy()` |
| 3 | FaF 注册 | `register_tasks() == 0`, probe/step ptr 有效 | C API 返回 + `get_dev_ptr() != 0` |
| 4 | 多步 FaF 触发 | `step_counter == steps`, `probe_flag >= expected` | `read_probe_flag_dev()` |
| 5 | 增量计算 overhead | `mean(delta_step) / mean(baseline_step) - 1 < 0.05` | EpochTimer callback |
| 6 | 恢复 NRMSE | `median NRMSE < 0.05` | `recover(target_step)` + `per_param_nrmse()` |

---

## 五、需要修改的文件

| # | 文件 | 操作 | 描述 |
|:---:|------|:---:|------|
| 1 | `python/delta_cell.py` | **新建** | DeltaTrainCell + analyze_model_layers |
| 2 | `python/direct_checkpoint.py` | 追加 | `build_layout_for_delta`, `register_delta_tasks` |
| 3 | `experiments/common.py` | 追加 | `setup_delta_faf`, `make_delta_training` |
| 4 | `experiments/delta_e2e/delta_e2e.py` | **新建** | 6 项 E2E 测试 |
| 5 | `experiments/delta_e2e/_run.sh` | **新建** | Shell wrapper |
| 6 | `to_do/active/CLAUDE_INSTRUCTIONS.md` | 更新 | 进展记录 |
| 7 | `to_do/active/STATUS_SUMMARY.md` | 更新 | 状态更新 |
| 8 | `to_do/active/ARCHITECTURE_ROADMAP.md` | 更新 | 新增模块 |

**C 层无需改动** — 当前 19 API 对增量检查点已足够。

---

## 六、风险

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| P_old 全量 Assign (1.52 GB) 步时过长 | 低 | 中 | MS 2.5 Assign 异步; Vector Engine 87.6% 空闲 |
| DeltaTrainCell GRAPH_MODE 编译 OOM | 中 | 高 | 中间张量用 FP16; 分阶段调试 |
| GE 图 DCE 消除缓冲 Assign | 中 | 高 | `ops.Depend()` 链到 loss 输出 |
| FaF listener 中 registered_tasks 竞争 | 低 | 中 | `save()` 前 `wait_for_io_completion()` |
