#!/usr/bin/env python3
"""
P0-U6 FaF Clean-Room Phase 2: sink=TRUE FaF full stack.
Key: MS warmup before SPDK init to avoid rte_eal_init overhead.

Configs:
  R5: sink=TRUE sink_size=10, enable_probe=False, no SPDK (baseline)
  R6: sink=TRUE sink_size=10, enable_probe=True, Full FaF (SPDK+listener+step_counter)

Expected:
  R5 e2_per_step: ~480ms (after compile)
  R6 e2_per_step: ~490ms (+2%, SPDK listener active)
  R6 SAFETY CHECK: flag=4 (20 steps / ckpt_interval=5)
"""
import os, sys, time, json, ctypes
REPO = "/home/user7/npu-nvme"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
DEVICE_ID = 1
SEQ_LEN = 1024
TOTAL_STEPS = 20
CKPT_INTERVAL = 5
SINK_SIZE = 10
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")

results = {}

def build_model_and_ds():
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)
    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    return model, ds, opt

def run_R5():
    """sink=TRUE baseline (no probe, no SPDK)"""
    print("\n" + "=" * 60)
    print("  R5: sink=TRUE sink_size=10, NO probe, NO SPDK")
    print("=" * 60)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model, ds, opt = build_model_and_ds()

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, None, 0, enable_probe=False, probe_mode="end")

    # MS warmup (ensures MS runtime initialized before any later SPDK init)
    warmup_ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    warmup_ds = warmup_ds.batch(1, drop_remainder=True).take(2)
    print("[R5] MS warmup (2 steps, sink=F)...", flush=True)
    ms.Model(cell).train(epoch=1, train_dataset=warmup_ds, callbacks=[],
                          dataset_sink_mode=False)

    # sink=TRUE training
    epoch_times = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append(time.perf_counter() - self.t0)

    ms_model = ms.Model(cell)
    print("[R5] Starting sink=TRUE training (2 epochs x 10 steps)...", flush=True)
    ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=SINK_SIZE)

    r = {
        "label": "R5_sinkT_baseline",
        "sink": True, "sink_size": SINK_SIZE, "enable_probe": False,
        "spdk": False, "epochs": 2, "total_steps": TOTAL_STEPS,
        "e1_s": round(epoch_times[0], 1), "e2_s": round(epoch_times[1], 1),
        "e1_per_step_ms": round(epoch_times[0]*1000/SINK_SIZE, 0),
        "e2_per_step_ms": round(epoch_times[1]*1000/SINK_SIZE, 0),
    }
    results["R5"] = r
    print("RESULT_R5: e1={e1_s}s e2={e2_s}s e1_ps={e1_per_step_ms}ms e2_ps={e2_per_step_ms}ms".format(**r), flush=True)


def run_R6():
    """sink=TRUE Full FaF (SPDK + listener + step_counter)"""
    print("\n" + "=" * 60)
    print("  R6: sink=TRUE sink_size=10, Full FaF (SPDK+listener+step_counter)")
    print("=" * 60)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model, ds, opt = build_model_and_ds()

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, None, 0,
                                 enable_probe=True, probe_mode="end",
                                 ckpt_interval=CKPT_INTERVAL)

    # Warmup function: direct cell() calls to force MS runtime init before SPDK.
    # This avoids rte_eal_init() "polluting" MS runtime state (see P0-U7).
    dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
    warmup_fn = lambda: [cell(dummy[0:1], dummy[0:1], dummy[0:1]) for _ in range(2)]

    # === SPDK init with warmup_fn (warmup runs BEFORE spdk_env_init) ===
    from direct_checkpoint import DirectCheckpoint
    import direct_checkpoint as dc

    print("[R6] SPDK init (warmup_fn provided)...", flush=True)
    ckpt = DirectCheckpoint(
        nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
        pipeline_depth=8, requested_chunk_size=4*1024*1024,
        enable_profiling=False, keep_last_n=3, slot_size_gb=10,
        warmup_fn=warmup_fn)

    # Dry-run to allocate NPU memory for step_counter
    dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
    cell(dummy[0:1], dummy[0:1], dummy[0:1])
    print("[R6] Dry-run done.", flush=True)

    # Register tasks
    ckpt.register_tasks(model, step=0)

    # Setup C-layer listener ptrs
    dev_flag = cell.flag._data_ptr()
    dev_step = cell.step_counter._data_ptr()
    print(f"[R6] flag={hex(dev_flag)} step_counter={hex(dev_step)}", flush=True)

    dc_lib = dc.lib
    rc = dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
    if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")
    rc = dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), CKPT_INTERVAL)
    if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")

    if dev_flag == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
        dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
    # P0-2 fix: always store the actual probe_flag_ptr for read_probe_flag_dev()
    ckpt.probe_flag_ptr = dev_flag
    print(f"[R6] Tasks registered. flag={hex(dev_flag)}", flush=True)

    # sink=TRUE training
    epoch_times = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append(time.perf_counter() - self.t0)

    ms_model = ms.Model(cell)
    print("[R6] Starting sink=TRUE FaF training (2 epochs x 10 steps)...", flush=True)
    ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=SINK_SIZE)

    r = {
        "label": "R6_sinkT_FaF",
        "sink": True, "sink_size": SINK_SIZE, "enable_probe": True,
        "spdk": True, "listener": "full", "epochs": 2, "total_steps": TOTAL_STEPS,
        "e1_s": round(epoch_times[0], 1), "e2_s": round(epoch_times[1], 1),
        "e1_per_step_ms": round(epoch_times[0]*1000/SINK_SIZE, 0),
        "e2_per_step_ms": round(epoch_times[1]*1000/SINK_SIZE, 0),
    }

    # Safety check
    try:
        flag = ckpt.read_probe_flag_dev()
        r["final_flag"] = int(flag)
        expected = TOTAL_STEPS // CKPT_INTERVAL
        r["safety"] = "PASSED" if flag >= expected else "FAILED"
    except Exception as e:
        r["safety"] = f"error: {e}"

    results["R6"] = r
    print("RESULT_R6: e1={e1_s}s e2={e2_s}s e1_ps={e1_per_step_ms}ms e2_ps={e2_per_step_ms}ms safety={safety}".format(**r), flush=True)

    ckpt.cleanup()


def main():
    run_R5()
    run_R6()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "faf_clean_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  FaF Clean Phase 2 Summary")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: e2_ps={v.get('e2_per_step_ms','?')}ms safety={v.get('safety','N/A')}")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
