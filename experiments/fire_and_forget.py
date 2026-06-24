#!/usr/bin/env python3
"""
P0-U6: Fire-and-Forget PoC — dataset_sink_mode=True 兼容性验证.

核心设计:
  - sink=TRUE 下 per-step callback 仅在最后一步触发 → CPU 驱动不可行
  - 改为 Device-Side Self-Trigger: 图内 step_counter 自增, 达到 CKPT_INTERVAL
    时由 AICPU trigger 内核写入 dev_trigger 缓冲区
  - C 层 listener 线程轮询 dev_trigger (device→host memcpy), 检测到新值即启动写盘
  - 训练结束后通过 C 层 profiling stderr 日志解析性能数据

预期: sink=TRUE 下 非CKPT步 ~370ms, CKPT步 ~370ms+写盘耗时
"""
import os, sys
# GE must find our custom opp libraries via LD_LIBRARY_PATH for sink=TRUE mode
_CUSTOM_OPP_LIB = os.path.join(os.path.dirname(__file__), "..", "build_out/opp/vendors/customize")
_PROTO_LIB = os.path.join(_CUSTOM_OPP_LIB, "op_proto/lib/linux/aarch64")
_IMPL_LIB = os.path.join(_CUSTOM_OPP_LIB, "op_impl/cpu/aicpu_kernel/impl")
os.environ["LD_LIBRARY_PATH"] = f"{_PROTO_LIB}:{_IMPL_LIB}:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, warnings, ctypes
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output", "log", "rank_0"), exist_ok=True)

import direct_checkpoint
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell, lib, NPUNVMEContext

DEVICE_ID = 1
TRAIN_MR  = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
NVME_ADDR = "0000:83:00.0"
CKPT_INTERVAL = 10
TOTAL_STEPS = 30
SEQ_LEN = 1024
ENABLE_PROFILING = True
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class FireAndForgetCallback(ms.Callback):
    """Minimal callback for sink=TRUE device-trigger mode.

    ALL trigger logic happens inside the graph:
      - step_counter auto-increments each step via ops.assign in construct()
      - At each CKPT_INTERVAL step, AICPU trigger kernel writes trigger_buf
      - C-layer listener polls trigger_buf and launches SPDK write

    This callback ONLY:
      1. Records wall-clock timing (per-step not possible, epoch-level only)
      2. Validates all CKPTs completed at epoch end

    NOTE: Task registration + flag/trigger setup must happen BEFORE model.train()
          because in sink=TRUE, callbacks only fire at the first and last step.
    """
    def __init__(self, model, train_cell, warmup_fn=None):
        super().__init__()
        self.model = model; self.train_cell = train_cell
        self.epoch_t0 = 0
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR, npu_device_id=DEVICE_ID,
            pipeline_depth=8, requested_chunk_size=4*1024*1024,
            enable_profiling=ENABLE_PROFILING, keep_last_n=3,
            slot_size_gb=10, warmup_fn=warmup_fn)

    def on_train_epoch_begin(self, run_context):
        self.epoch_t0 = time.perf_counter()

    def on_train_epoch_end(self, run_context):
        elapsed = (time.perf_counter() - self.epoch_t0)
        print(f"\n  [FaF] Epoch completed in {elapsed:.1f}s", flush=True)

        # Safety check: verify all CKPTs completed
        try:
            final_flag = self.ckpt.read_probe_flag_dev()
            expected_ckpts = TOTAL_STEPS // CKPT_INTERVAL
            if final_flag >= expected_ckpts:
                print(f"  [FaF] SAFETY CHECK PASSED: flag={final_flag} >= expected={expected_ckpts}")
            else:
                print(f"  [FaF] SAFETY CHECK FAILED: flag={final_flag} < expected={expected_ckpts} "
                      f"({expected_ckpts - final_flag} CKPTs may be incomplete!)")
        except Exception as e:
            print(f"  [FaF] Safety check error: {e}")

        self.ckpt.cleanup()

        # Save summary
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results = {
            "test": "P0-U6 Fire-and-Forget Dev-Trigger PoC",
            "dataset_sink_mode": True,
            "total_steps": TOTAL_STEPS,
            "ckpt_interval": CKPT_INTERVAL,
            "epoch_elapsed_s": round(elapsed, 1),
            "avg_step_ms": round(elapsed * 1000 / TOTAL_STEPS, 1),
        }
        try:
            results["final_flag"] = int(final_flag) if 'final_flag' in dir() else None
        except: pass
        with open(os.path.join(OUTPUT_DIR, "fire_and_forget.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to output/fire_and_forget.json")

    def on_train_step_begin(self, run_context):
        pass  # No-op: task registration moved before model.train() for sink=TRUE


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    print("[FaF] Building GPT-2 XL model...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = 1024
    cfg.max_position_embeddings = 1024
    base_model = AutoModel.from_config(cfg)

    total_bytes = sum(
        int(np.prod(p.shape)) * ms.dtype_to_nptype(p.dtype)().itemsize
        for p in base_model.get_parameters())
    print(f"[FaF] Model params: {total_bytes / 1024 / 1024:.1f} MB", flush=True)

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    probe_wrapper = ProbeTrainOneStepCell(
        base_model, optimizer, enable_probe=True,
        ckpt_interval=CKPT_INTERVAL)

    # Warmup: direct cell() calls force MS runtime init before SPDK.
    # Done inside DirectCheckpoint.__init__ via warmup_fn callback.
    dummy = Tensor(np.zeros((1, SEQ_LEN,), dtype=np.int32), ms.int32)
    warmup_fn = lambda: [probe_wrapper(dummy[0:1], dummy[0:1], dummy[0:1]) for _ in range(2)]

    cb = FireAndForgetCallback(base_model, probe_wrapper, warmup_fn=warmup_fn)
    ms_model = ms.Model(probe_wrapper)

    # For sink=TRUE, we must register tasks BEFORE model.train() because
    # callbacks only fire at the first/last step. The AICPU kernels inside the
    # fused graph operate on ProbeTrainOneStepCell's Parameter device buffers
    # (probe_flag, probe_expected, trigger_buf). We need to give the C listener
    # the SAME device addresses so it can read the AICPU kernel outputs.
    print("[FaF] Pre-registering SPDK tasks + flag/trigger ptrs...", flush=True)

    # Dry-run a single forward pass to allocate NPU memory for all parameters
    # (warmup already done inside FireAndForgetCallback.__init__)
    probe_wrapper(dummy[0:1], dummy[0:1], dummy[0:1])
    print("[FaF] Dry-run forward pass done, NPU memory allocated.", flush=True)

    cb.ckpt.register_tasks(base_model, step=0)

    # Extract the ACTUAL device addresses of the graph's Parameter buffers
    # These are the memory that TrigProbe/WaitProbe AICPU kernels operate on.
    dev_flag_addr = probe_wrapper.flag._data_ptr()
    dev_step_addr = probe_wrapper.step_counter._data_ptr()

    print(f"[FaF] Graph Parameter addrs: flag={hex(dev_flag_addr)} "
          f"step_counter={hex(dev_step_addr)}", flush=True)

    # Pass flag and step_counter device addresses to C layer listener.
    # C layer polls step_counter every 100us and triggers SPDK write at CKPT intervals.
    # No AICPU kernel needed — pure C-layer fire-and-forget.
    dc_lib = direct_checkpoint.lib
    rc = dc_lib.npu_nvme_set_probe_flag_ptr(cb.ckpt.ctx, ctypes.c_void_p(dev_flag_addr))
    if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")

    # Use the new npu_nvme_set_step_ptr API (3-arg version)
    if hasattr(dc_lib, "npu_nvme_set_step_ptr"):
        rc = dc_lib.npu_nvme_set_step_ptr(cb.ckpt.ctx, ctypes.c_void_p(dev_step_addr), CKPT_INTERVAL)
        if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")
    else:
        # Fallback to old set_trigger_ptr (backward compat alias)
        rc = dc_lib.npu_nvme_set_trigger_ptr(cb.ckpt.ctx, ctypes.c_void_p(dev_step_addr))
        if rc != 0: raise RuntimeError(f"set_trigger_ptr failed: {rc}")

    # If flag Parameter has no device memory (MS lazy allocation), C layer
    # self-allocates. Retrieve the actual address from C layer for safety checks.
    if dev_flag_addr == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
        dev_flag_addr = dc_lib.npu_nvme_get_probe_flag_dev_ptr(cb.ckpt.ctx)
    cb.ckpt.probe_flag_ptr = dev_flag_addr
    print(f"[FaF] Tasks + flag/step registered. flag={hex(dev_flag_addr)} step={hex(dev_step_addr)}", flush=True)

    print(f"\n[FaF] === Starting Fire-and-Forget PoC (sink=TRUE) ===\n", flush=True)
    t_total = time.perf_counter()
    try:
        ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb],
                       dataset_sink_mode=True)
    except Exception as e:
        print(f"\n[FaF] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t_total
    print(f"\n[FaF] Total wall time: {elapsed:.1f}s for {TOTAL_STEPS} steps", flush=True)
    print("[FaF] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    main()
