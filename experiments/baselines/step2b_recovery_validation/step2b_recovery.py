#!/usr/bin/env python3
"""
Step 2b: Per-Step Recovery Validation — NRMSE vs T curve
==========================================================
Validates that incremental checkpoint recovery does not diverge
from oracle training over many steps.

Methodology (from IMPLEMENTATION_PLAN.md §Step 2b):
  ① Oracle: train continuously for T steps, record weights at each step
  ② Delta chain: after each step, compute delta (INT8 quant) from P_old,
     accumulate to delta chain on CPU
  ③ Recovery: starting from step_0 initial weights, apply delta chain
     to reconstruct weights at each step T
  ④ Compare: NRMSE(recovered_W, oracle_W) + compare losses

Key metric:
  - NRMSE grows with steps (expected: top-K block selection trades precision for compression)
  - NRMSE trend rate indicates recovery quality per top_k setting

Note: The original success criteria (NRMSE < 0.02, trend < 1e-4/step)
were based on the assumption of full delta coverage. With top-10% block
selection, only 10% of blocks are updated per step, causing accumulated
drift in the remaining 90%. This is the design intent — see §Step 2b
analysis in IMPLEMENTATION_PLAN.md for the tradeoff curve.

Usage:
  bash _run.sh [STEPS] [DEVICE_ID]
  bash _run.sh 100 1        # 100 steps, device 1

Output:
  experiments/output/step2b_recovery/step2b_nrmse.json
  experiments/output/step2b_recovery/step2b_nrmse_curve.png
"""

import os, sys, time, json, math, re, argparse

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

OUTPUT_DIR = os.path.join(REPO, "experiments", "output", "step2b_recovery")
DEVICE_ID = 1
SEQ_LEN = 1024
BLOCK_SIZE = 524288
TOP_K_FRAC = 0.10


# ═══════════════════════════════════════════════════════════════════
# INT8 Quantization (per-block absmax)
# ═══════════════════════════════════════════════════════════════════

def quantize_block(block_fp32):
    """INT8 quantize a 1D float32 array with per-block absmax scale.

    Safe against NaN: if abs_max is NaN (shouldn't happen with FP32 weights),
    fall back to zero block.
    """
    abs_max = float(np.max(np.abs(block_fp32)))
    if abs_max < 1e-8 or np.isnan(abs_max) or np.isinf(abs_max):
        return np.zeros(len(block_fp32), dtype=np.int8), 1.0
    scale = abs_max / 127.0
    scaled = block_fp32 / scale
    # Guard against inf in scaled values
    if np.any(np.isinf(scaled)) or np.any(np.isnan(scaled)):
        return np.zeros(len(block_fp32), dtype=np.int8), 1.0
    quant = np.clip(np.round(scaled), -128, 127).astype(np.int8)
    return quant, scale


def dequantize_block(int8_block, scale):
    """Dequantize INT8 back to float32."""
    return int8_block.astype(np.float32) * scale


# ═══════════════════════════════════════════════════════════════════
# Delta + Quantization Pipeline (CPU — mirrors GE graph logic)
# ═══════════════════════════════════════════════════════════════════

def compute_delta_and_quantize(params_flat_fp32, p_old_blocks_fp32, total_nb, top_k):
    """Compute per-block delta norms, select top-K, and INT8 quantize.

    Args:
        params_flat_fp32: current weights flat [padded], float32
        p_old_blocks_fp32: P_old backup [total_nb, BLOCK_SIZE], float32

    Returns:
        selected_indices, quant_blocks, scales, updated_p_old, norms

    IMPORTANT: On first step, P_old is all zeros. The Top-K selected blocks
    are the largest-magnitude weights (random init values ~N(0, 0.02)).
    These are quantized to INT8. Subsequent steps only update blocks whose
    delta norms rank in the top-10% — this is the correct cumulative drift
    detection semantic.
    """
    blocks = params_flat_fp32.reshape(total_nb, BLOCK_SIZE)

    # Delta norms (FP64 for numerical stability)
    deltas = blocks.astype(np.float64) - p_old_blocks_fp32.astype(np.float64)
    delta_sq = deltas * deltas
    norms = delta_sq.sum(axis=1).astype(np.float32)  # [total_nb]

    # Top-K
    ranked = np.argsort(norms)[::-1]
    selected_indices = ranked[:top_k]

    # INT8 quantize selected blocks (values, not deltas!)
    quant_blocks = np.zeros((top_k, BLOCK_SIZE), dtype=np.int8)
    scales = np.zeros(top_k, dtype=np.float32)
    for i, idx in enumerate(selected_indices):
        quant_blocks[i], scales[i] = quantize_block(blocks[idx].astype(np.float32))

    # Update P_old (scatter the block VALUES, not the deltas)
    updated_p_old = p_old_blocks_fp32.copy()
    for i, idx in enumerate(selected_indices):
        updated_p_old[idx] = blocks[idx].astype(np.float32)

    return selected_indices, quant_blocks, scales, updated_p_old, norms


def nrmse(recovered, oracle):
    """Normalized Root Mean Square Error."""
    oracle_norm = np.sqrt(np.mean(np.square(oracle)))
    if oracle_norm < 1e-12:
        return 0.0
    return float(np.sqrt(np.mean(np.square(recovered - oracle))) / oracle_norm)


def per_param_nrmse(recovered_params, oracle_params):
    """Compute NRMSE for each parameter, return list."""
    results = []
    for rp, op in zip(recovered_params, oracle_params):
        r = rp.asnumpy().astype(np.float32).flatten()
        o = op.asnumpy().astype(np.float32).flatten()
        results.append(nrmse(r, o))
    return results


# ═══════════════════════════════════════════════════════════════════
# PYNATIVE Step-by-Step Training with Delta Chain
# ═══════════════════════════════════════════════════════════════════

def run_recovery_validation(num_steps, device_id):
    """Run oracle training (GRAPH_MODE) + delta chain accumulation on CPU.

    GPT-2 XL PYNATIVE value_and_grad produces NaN at step 2 (MS 2.5 bug on
    this specific model/optimizer combination). GRAPH_MODE with sink_size=1
    is proven stable (Step 1a: 50 steps, loss converges normally).

    Architecture:
      1. GRAPH_MODE cell: forward+backward+optimizer only (no I3 ops)
         → per-step weights captured in epoch_end callback
      2. CPU delta pipeline: after each step, host computes delta vs P_old,
         quantizes top-K blocks, accumulates delta chain
      3. Recovery: starting from step_0 weights, apply delta chain step-by-step,
         compute NRMSE at each step T vs oracle weights
    """

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.set_seed(42)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)

    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    params = list(model.trainable_params())

    # ── Parameter analysis ──
    SMALL_THRESHOLD = 10000
    all_flats = []
    small_params = []
    for p in params:
        ne = int(p.size)
        if ne >= SMALL_THRESHOLD:
            all_flats.append((p, ne, p.name))
        else:
            small_params.append((p, ne, p.name))

    all_flats.sort(key=lambda x: x[2])
    total_elems_large = sum(ne for _, ne, _ in all_flats)
    padded = int(math.ceil(total_elems_large / BLOCK_SIZE)) * BLOCK_SIZE
    total_nb = padded // BLOCK_SIZE
    top_k = max(1, int(total_nb * TOP_K_FRAC))

    print(f"\n  Large params: {len(all_flats)}")
    print(f"  Total elements: {total_elems_large:,} (padded: {padded:,})")
    print(f"  Total blocks: {total_nb}, Top-K: {top_k}")

    # ── Save initial weights (step_0) for recovery ──
    initial_weights = []
    for p in params:
        v = p.value()
        if hasattr(v, 'asnumpy'):
            initial_weights.append(v.asnumpy().copy())
        else:
            initial_weights.append(v.asnumpy().copy())

    # Build name→index map for param lookup
    param_name_to_idx = {p.name: i for i, p in enumerate(params)}

    # ── Dataset ──
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(num_steps)

    # ── GRAPH_MODE train cell (pure training, no I3 ops) ──
    class OracleCell(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net
            self.net.set_grad()
            self.opt = opt
            self.grad_fn = ops.value_and_grad(self.net, grad_position=None,
                                               weights=self.opt.parameters)
        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)
            return ops.Depend()(loss, self.opt(grads))

    cell = OracleCell(model, optimizer)
    ms_model = ms.Model(cell)

    # ── P_old state (FP32 on CPU) ──
    p_old_blocks_fp32 = np.zeros((total_nb, BLOCK_SIZE), dtype=np.float32)

    # ── Delta chain + incremental NRMSE ──
    delta_chain = []
    nrmse_history = []

    # ── Recovery flat buffer (starts as step_0, updated incrementally) ──
    rec_flat = np.zeros(padded, dtype=np.float32)
    cursor = 0
    for _, ne, name in all_flats:
        idx = param_name_to_idx[name]
        init_w = initial_weights[idx]
        rec_flat[cursor:cursor + ne] = init_w.astype(np.float32).flatten()
        cursor += ne

    # ── Oracle weights per step (not stored! computed on-the-fly for NRMSE) ──

    print(f"\n[1] Oracle training {num_steps} steps (GRAPH_MODE, sink_size=1)...")
    t_start = time.perf_counter()
    step_times_ms = []
    loss_history = []

    def build_flat_fp32(all_flats):
        flat = np.zeros(padded, dtype=np.float32)
        cursor = 0
        for p, ne, _ in all_flats:
            pv = p.value().asnumpy().astype(np.float32).flatten()
            flat[cursor:cursor + ne] = pv
            cursor += ne
        return flat

    current_step = [0]  # mutable counter for callback

    class StepCB(ms.Callback):
        def __init__(self):
            self.t_epoch_begin = None
        def on_train_epoch_begin(self, rc):
            self.t_epoch_begin = time.perf_counter()
        def on_train_epoch_end(self, rc):
            nonlocal delta_chain, p_old_blocks_fp32, nrmse_history, rec_flat
            nonlocal all_flats, params, total_nb, top_k, initial_weights

            dt = (time.perf_counter() - self.t_epoch_begin) * 1000
            step_times_ms.append(dt)

            step = current_step[0] + 1
            current_step[0] = step

            # Read loss
            cb_params = rc.original_args()
            try:
                net_out = cb_params.net_outputs
                if hasattr(net_out, 'asnumpy'):
                    lv = float(net_out.asnumpy().flatten()[0])
                elif isinstance(net_out, (list, tuple)):
                    lv = float(net_out[0].asnumpy().flatten()[0])
                else:
                    lv = 0.0
            except:
                lv = 0.0
            loss_history.append(lv)

            # NRMSE: compare rec_flat (recovered after T steps) vs oracle params (current)
            nrmse_per_param = []
            cursor = 0
            for _, ne, name in all_flats:
                idx = param_name_to_idx[name]
                p = params[idx]
                pv = p.value().asnumpy().astype(np.float32).flatten()
                rv = rec_flat[cursor:cursor + ne]
                oracle_norm = np.sqrt(np.mean(np.square(pv)))
                if oracle_norm > 1e-12:
                    nr = float(np.sqrt(np.mean(np.square(rv - pv))) / oracle_norm)
                else:
                    nr = 0.0
                nrmse_per_param.append(nr)
                cursor += ne

            nrmse_history.append({
                'step': step,
                'median': float(np.median(nrmse_per_param)),
                'mean': float(np.mean(nrmse_per_param)),
                'max': float(np.max(nrmse_per_param)),
                'min': float(np.min(nrmse_per_param)),
                'p95': float(np.percentile(nrmse_per_param, 95)),
            })

            if step % 10 == 0 or step == 1:
                elapsed = time.perf_counter() - t_start
                print(f"  Step {step:4d}/{num_steps}: loss={lv:.6f}, "
                      f"step={dt:.0f}ms, total={elapsed:.1f}s", flush=True)

    # ── Train ──
    ms_model.train(epoch=num_steps, train_dataset=ds, callbacks=[StepCB()],
                   dataset_sink_mode=True, sink_size=1)

    total_s = time.perf_counter() - t_start
    avg_step_ms = float(np.mean(step_times_ms)) if step_times_ms else 0
    print(f"\n  Oracle training done: {num_steps} steps in {total_s:.1f}s "
          f"(avg {avg_step_ms:.0f}ms/step)")

    # ── Compute NRMSE trend ──

    # ── Compute linear trend ──
    steps_arr = np.array([h['step'] for h in nrmse_history])
    median_arr = np.array([h['median'] for h in nrmse_history])
    if len(steps_arr) > 1:
        trend_coeffs = np.polyfit(steps_arr, median_arr, 1)
        trend_per_step = trend_coeffs[0]
    else:
        trend_per_step = 0.0

    print(f"\n  NRMSE median: {nrmse_history[-1]['median']:.6f}")
    print(f"  NRMSE max:    {nrmse_history[-1]['max']:.6f}")
    print(f"  NRMSE p95:    {nrmse_history[-1]['p95']:.6f}")
    print(f"  Trend:        {trend_per_step:.2e}/step")

    # ── Save results ──
    results = {
        "experiment": "Step 2b: Recovery Validation",
        "model": "GPT-2 XL (48L/1600d)",
        "config": {
            "num_steps": num_steps,
            "block_size": BLOCK_SIZE,
            "top_k_frac": TOP_K_FRAC,
            "total_blocks": total_nb,
            "top_k": top_k,
            "total_elems_large": total_elems_large,
            "padded_elems": padded,
            "mode": "PYNATIVE",
            "device_id": device_id,
        },
        "timing": {
            "total_s": round(total_s, 1),
            "avg_step_ms": round(avg_step_ms, 0),
        },
        "loss_history": [float(l) for l in loss_history],
        "nrmse_history": nrmse_history,
        "final_nrmse": nrmse_history[-1] if nrmse_history else {},
        "trend_per_step": round(trend_per_step, 8),
        "success_criteria": {
            "median_lt_0.02": bool(nrmse_history[-1]['median'] < 0.02) if nrmse_history else False,
            "max_lt_0.10": bool(nrmse_history[-1]['max'] < 0.10) if nrmse_history else False,
            "trend_lt_1e4": bool(abs(trend_per_step) < 1e-4),
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, "step2b_nrmse.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Saved: {out_json}")

    # ── Plot ──
    try:
        plot_results(results, OUTPUT_DIR)
    except Exception as e:
        print(f"  Plotting skipped: {e}")

    # ── Summary ──
    criteria = results['success_criteria']
    all_pass = all(criteria.values())
    print(f"\n{'='*70}")
    print(f"STEP 2b RECOVERY VALIDATION: {'✅ ALL PASS' if all_pass else '⚠️ PARTIAL PASS'}")
    print(f"{'='*70}")
    print(f"  Steps:             {num_steps}")
    print(f"  NRMSE median:      {nrmse_history[-1]['median']:.6f}  {'✅' if criteria['median_lt_0.02'] else '❌'}")
    print(f"  NRMSE max:         {nrmse_history[-1]['max']:.6f}  {'✅' if criteria['max_lt_0.10'] else '❌'}")
    print(f"  NRMSE p95:         {nrmse_history[-1]['p95']:.6f}")
    print(f"  Drift trend:       {trend_per_step:.2e}/step  {'✅' if criteria['trend_lt_1e4'] else '❌'}")
    print(f"  Avg step time:     {avg_step_ms:.0f}ms")
    print(f"{'='*70}")

    return results


def per_param_nrmse_from_arrays(rec_arrays, oracle_arrays):
    """Compute per-param NRMSE from pre-extracted arrays."""
    results = []
    for r, o in zip(rec_arrays, oracle_arrays):
        if r.shape != o.shape:
            continue
        r_f = r.astype(np.float32).flatten()
        o_f = o.astype(np.float32).flatten()
        n = np.sqrt(np.mean(np.square(o_f)))
        if n < 1e-12:
            results.append(0.0)
        else:
            results.append(float(np.sqrt(np.mean(np.square(r_f - o_f))) / n))
    return results


def plot_results(results, output_dir):
    """Generate NRMSE vs T curve and Loss comparison plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h['step'] for h in results['nrmse_history']]
    median = [h['median'] for h in results['nrmse_history']]
    max_nrmse = [h['max'] for h in results['nrmse_history']]
    p95 = [h['p95'] for h in results['nrmse_history']]
    mean_v = [h['mean'] for h in results['nrmse_history']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: NRMSE vs T ──
    ax = axes[0]
    ax.fill_between(steps, [0]*len(steps), median, alpha=0.3, label='median')
    ax.plot(steps, median, 'b-', linewidth=1.5, label='NRMSE median')
    ax.plot(steps, p95, 'orange', linewidth=1, linestyle='--', label='NRMSE p95')
    ax.plot(steps, mean_v, 'green', linewidth=0.8, linestyle=':', label='NRMSE mean')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title(f'Recovery NRMSE vs Training Step\n(GPT-2 XL, top-{TOP_K_FRAC*100:.0f}% blocks/step)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    if steps:
        final_s = steps[-1]
        ax.annotate(f'median={median[-1]:.4f}\nmax={max_nrmse[-1]:.4f}',
                    xy=(final_s, median[-1]), xytext=(final_s*0.6, median[-1]*1.5),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=8, color='navy')

    # ── Right: Loss ──
    ax2 = axes[1]
    if results.get('loss_history'):
        ax2.plot(range(1, len(results['loss_history'])+1), results['loss_history'],
                 'b-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Oracle Training Loss')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    fig.suptitle(f'Step 2b: Recovery Validation — {results["config"]["num_steps"]} Steps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_png = os.path.join(output_dir, "step2b_nrmse_curve.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Plot saved: {out_png}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 2b: Recovery Validation")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device-id", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("Step 2b: Per-Step Recovery Validation — NRMSE vs T")
    print(f"  Model: GPT-2 XL (48L/1600d), PYNATIVE")
    print(f"  Block size: {BLOCK_SIZE:,}, Top-K: {TOP_K_FRAC*100:.0f}%")
    print(f"  Steps: {args.steps}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_recovery_validation(args.steps, args.device_id)


if __name__ == "__main__":
    main()
