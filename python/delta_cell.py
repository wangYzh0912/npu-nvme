"""Incremental delta-checkpoint training cell.

Provides DeltaTrainCell — a MindSpore nn.Cell that integrates the full
7-phase delta detection pipeline into the GE graph, plus the model-layer
analyser used to configure it.

Usage:
    from delta_cell import DeltaTrainCell, analyze_model_layers

    info = analyze_model_layers(model)
    cell = DeltaTrainCell(model, optimizer, block_size=524288, top_k_frac=0.10)
    # cell can be used with ms.Model.train(dataset_sink_mode=True)

Delta frame serialization and patching are handled by delta_protocol.
SPDK I/O is handled by DirectCheckpoint (C layer via c_bindings).
"""

import math
import re

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

# ---- Constants ----

BLOCK_SIZE_DEFAULT = 524288     # 512K elements (1 MB FP16, 512 KB INT8)
TOP_K_FRAC_DEFAULT = 0.10       # top 10 % blocks selected per step
SMALL_THRESHOLD_DEFAULT = 10000  # params below this size are stored whole


# ---- Model-layer analyser --------------------------------------------------

def analyze_model_layers(model: nn.Cell,
                         block_size: int = BLOCK_SIZE_DEFAULT,
                         small_threshold: int = SMALL_THRESHOLD_DEFAULT) -> dict:
    """Analyse a model's trainable parameters for delta-checkpoint layout.

    Each parameter >= small_threshold elements is treated as a
    contiguous flat tensor and tiled into blocks of size block_size.
    Smaller parameters are collected as ``small_params`` and handled
    separately by the serialization layer.

    Args:
        model:           MindSpore model (e.g. GPT-2 XL).
        block_size:      elements per block (default 524288).
        small_threshold: maximum elements for a whole-stored param.

    Returns:
        dict with keys:

        - total_elems_large (int): total elements in large params.
        - padded_elems (int):     total after zero-padding to block boundaries.
        - total_nb (int):         number of blocks = padded_elems // block_size.
        - block_params (list[Parameter]): large-param Parameters in order.
        - block_nelem (list[int]):       element count for each block-param.
        - fp16_needed (list[bool]):      True if param needs Cast to fp16.
        - small_params (list[(Parameter, str, int)]): small params.
        - layer_ids (list[int]):  sorted layer identifiers.
    """
    params = list(model.trainable_params())

    all_flats = []      # [(param, nelem, name, layer_id), ...]
    small_params = []   # [(param, nelem, name), ...]

    for p in params:
        ne = int(p.size)
        name = p.name

        m = re.search(r'backbone\.blocks\.(\d+)\.', name)
        if m:
            lid = int(m.group(1))
        elif 'backbone.embedding' in name:
            lid = -2
        elif 'backbone.layernorm' in name:
            lid = -1
        else:
            lid = -3

        if ne >= small_threshold:
            all_flats.append((p, ne, name, lid))
        else:
            small_params.append((p, ne, name))

    # Sort by layer for deterministic layout
    all_flats.sort(key=lambda x: (x[3], x[2]))

    # Build block descriptors
    blocks = []
    cursor = 0
    for p, ne, name, lid in all_flats:
        nb = max(1, math.ceil(ne / block_size))
        blocks.append({
            'param': p,
            'name': name,
            'layer_id': lid,
            'start_elem': cursor,
            'end_elem': cursor + ne,
            'nelem': ne,
            'num_blocks': nb,
            'padded_elems': nb * block_size,
        })
        cursor += ne

    total_elems_large = cursor
    padded_elems = sum(b['padded_elems'] for b in blocks)
    total_nb = padded_elems // block_size

    # Per-layer block index (for debugging / future per-layer ops)
    layer_ids = sorted(set(b['layer_id'] for b in blocks))

    print(f"\n  Delta Cell Configuration:")
    print(f"    Large params: {len(blocks)}")
    print(f"    Total elements: {total_elems_large:,} ({padded_elems:,} padded)")
    print(f"    Total blocks: {total_nb} (block_size={block_size})")

    return {
        'total_elems_large': total_elems_large,
        'padded_elems': padded_elems,
        'total_nb': total_nb,
        'block_params': [b['param'] for b in blocks],
        'block_nelem': [b['nelem'] for b in blocks],
        'fp16_needed': [
            p.dtype not in (ms.float32, ms.float16) or p.dtype == ms.float32
            for p in [b['param'] for b in blocks]
        ],
        'small_params': small_params,
        'layer_ids': layer_ids,
    }


# ---- Delta-train cell ------------------------------------------------------

class DeltaTrainCell(nn.Cell):
    """Training cell that performs incremental delta-checkpoint in-graph.

    ``construct()`` runs a 7-phase pipeline inside the GE graph:

    * **Phase A** — standard forward + backward + optimizer.
    * **Phase B** — cross-layer parameter aggregation into a flat block
      matrix, and dequantisation of the previous-step snapshot (P_old).
    * **Phase C** — per-block delta-norm computation.
    * **Phase D** — Top-K selection of the most-changed blocks.
    * **Phase E** — INT8 quantisation of the selected blocks for output.
    * **Phase F** — full INT8 re-quantisation of all blocks and full
      ``Assign`` to P_old (avoids the MS 2.5 ScatterUpdate bug).
    * **Phase G** — assign output buffers and increment step_counter.

    The output buffers (``delta_quant_buf``, ``delta_scale_buf``,
    ``delta_idx_buf``, ``delta_p_old``, ``step_counter``) reside in HBM
    and can be registered with the C-layer FaF listener for asynchronous
    SPDK persistence.

    Args:
        network:         MindSpore model (e.g. GPT-2 XL).
        optimizer:       MindSpore optimizer.
        block_size:      elements per block (default 524288).
        top_k_frac:      fraction of blocks selected per step (default 0.10).
        small_threshold: max elements for whole-stored params (default 10000).
    """

    def __init__(self,
                 network: nn.Cell,
                 optimizer: nn.Optimizer,
                 block_size: int = BLOCK_SIZE_DEFAULT,
                 top_k_frac: float = TOP_K_FRAC_DEFAULT,
                 small_threshold: int = SMALL_THRESHOLD_DEFAULT):
        super().__init__(auto_prefix=False)

        # Analyse model structure
        info = analyze_model_layers(network, block_size, small_threshold)

        # Training components
        self.net = network
        self.net.set_grad()
        self.opt = optimizer
        self.grad_fn = ops.value_and_grad(
            self.net, grad_position=None, weights=self.opt.parameters)

        # Block metadata (immutable inside GE graph)
        self.nb = info['total_nb']
        self.bs = block_size
        self.k = max(1, int(info['total_nb'] * top_k_frac))
        self.te = info['total_elems_large']
        self.pe = info['padded_elems']
        self.n_params = len(info['block_params'])
        self.block_params = tuple(info['block_params'])
        self.block_nelem = tuple(info['block_nelem'])
        self.fp16_needed = tuple(info['fp16_needed'])

        # Output buffers (HBM Parameters)
        self.delta_quant_buf = Parameter(
            Tensor(np.zeros(self.k * block_size, dtype=np.int8)),
            name="delta_quant_buf", requires_grad=False)
        self.delta_scale_buf = Parameter(
            Tensor(np.zeros(self.k, dtype=np.float32)),
            name="delta_scale_buf", requires_grad=False)
        self.delta_idx_buf = Parameter(
            Tensor(np.zeros(self.k, dtype=np.int32)),
            name="delta_idx_buf", requires_grad=False)

        # Previous-step full-model INT8 snapshot stored as 2D [nb, bs].
        # ScatterUpdate writes only the top-K rows each step (not full Assign).
        self.delta_p_old = Parameter(
            Tensor(np.zeros((info['total_nb'], block_size), dtype=np.int8)),
            name="delta_p_old", requires_grad=False)

        # Step counter for FaF listener
        self.step_counter = Parameter(
            Tensor([0], dtype=ms.int32),
            name="step_counter", requires_grad=False)
        self.one = Tensor([1], dtype=ms.int32)

        print(f"    Top-K: {self.k} (top {top_k_frac*100:.0f}%)")
        print(f"    delta_p_old: {info['total_nb'] * block_size / 1e9:.2f} GB INT8 (2D [{info['total_nb']}, {block_size}])")
        print(f"    delta_quant_buf: {self.k * block_size / 1e6:.1f} MB INT8")

    # ---- Helpers (in-graph) ------------------------------------------------

    def _int8_quantize(self, blocks_fp16):
        """INT8-quantise a ``[n, bs]`` FP16 tensor.

        Pipeline:
          Cast(FP32) → Abs → ReduceMax → Div(127.0)
          → Reshape([n,1]) → Div → Round
          → clip_by_value(-128, 127) → Cast(INT8)

        Returns:
            (int8_blocks (Tensor ``[n, bs]`` INT8),
             scales     (Tensor ``[n]``    FP32))
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

    # ---- Main forward pass ------------------------------------------------

    def construct(self, *inputs):
        """Execute one training step with full delta-checkpoint pipeline.

        Returns the loss tensor, with all buffer writes chained via
        ``ops.Depend`` so that the GE graph does not dead-code-eliminate
        the side-effect operations.
        """
        # Phase A — standard training
        loss, grads = self.grad_fn(*inputs)
        loss = ops.Depend()(loss, self.opt(grads))

        # Phase B — cross-layer parameter aggregation + P_old dequant
        flat_parts = []
        for i in range(self.n_params):
            p = self.block_params[i]
            ne = self.block_nelem[i]
            pv = ops.Cast()(p, ms.float16) if self.fp16_needed[i] else p
            flat_parts.append(ops.Reshape()(pv, (ne,)))
        all_flat = ops.Concat()(tuple(flat_parts))            # [total_elems]

        pad_amt = self.pe - self.te
        all_flat_padded = ops.pad(all_flat, (0, pad_amt),
                                   mode='constant', value=0.0)
        AllBlocks = ops.Reshape()(all_flat_padded, (self.nb, self.bs))

        # Dequant P_old: INT8 → FP16 (P_old is stored 2D [nb, bs])
        P_old_fp16 = ops.Cast()(self.delta_p_old, ms.float16)

        # Phase C — delta norms per block
        deltas = ops.Sub()(AllBlocks, P_old_fp16)
        delta_sq = ops.Mul()(deltas, deltas)
        norms = ops.ReduceSum()(delta_sq, 1)                   # [nb] fp16
        norms_fp32 = ops.Cast()(norms, ms.float32)             # for TopK stability

        # Phase D — Top-K selection
        _, top_indices = ops.TopK(sorted=True)(norms_fp32, self.k)

        # Phase E — output quantisation (Top-K blocks only)
        selected_fp16 = ops.Gather()(AllBlocks, top_indices, 0)
        quant_int8, scales = self._int8_quantize(selected_fp16)

        # Phase F — scatter top-K INT8 blocks into P_old (in-place row update).
        # Only the k selected rows are written; the remaining nb-k rows
        # stay unchanged.  This replaces the full-Assign workaround for the
        # MS 2.5 ScatterUpdate bug (verified fixed in MS 2.5.0).
        ops.ScatterUpdate()(self.delta_p_old, top_indices, quant_int8)

        # Phase G — output buffer assignments + step_counter
        ops.Assign()(self.delta_quant_buf,
                     ops.Reshape()(quant_int8, (self.k * self.bs,)))
        ops.Assign()(self.delta_scale_buf, scales)
        ops.Assign()(self.delta_idx_buf, top_indices)
        ops.AssignAdd()(self.step_counter, self.one)

        # Dependency chain — prevent DCE (reference Parameters after Assign)
        loss = ops.Depend()(loss, self.delta_quant_buf)
        loss = ops.Depend()(loss, self.delta_scale_buf)
        loss = ops.Depend()(loss, self.delta_idx_buf)
        loss = ops.Depend()(loss, self.delta_p_old)
        loss = ops.Depend()(loss, self.step_counter)
        return loss


__all__ = ['DeltaTrainCell', 'analyze_model_layers']
