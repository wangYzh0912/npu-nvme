#!/usr/bin/env python3
"""
Quick test: SPDK+Probe with dataset_sink_mode=True.
Reuses the full SpdkCkptCallback but with sink=TRUE.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, warnings
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", "python"))

import direct_checkpoint
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell, get_dev_ptr

DEVICE_ID = 1
TRAIN_MR  = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
NVME_ADDR = "0000:83:00.0"
CKPT_INTERVAL = 10
TOTAL_STEPS = 35
ENABLE_PROFILING = True

class SpdkCkptCallback(ms.Callback):
    def __init__(self, model, train_cell):
        super().__init__()
        self.model = model; self.train_cell = train_cell
        self.has_registered = False; self.step_start = 0
        self.assign = ops.Assign(); self.expected_value = 0
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR, npu_device_id=DEVICE_ID,
            pipeline_depth=8, requested_chunk_size=4*1024*1024,
            enable_profiling=ENABLE_PROFILING, keep_last_n=3,
            slot_size_gb=10)
        self.step_times = []; self.wait_times = []

    def on_train_step_begin(self, run_context):
        self.step_start = time.perf_counter()
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if not self.has_registered: return
        if cur_step % CKPT_INTERVAL == 0:
            try:
                self.expected_value += 1
                self.assign(self.train_cell.expected, ms.Tensor([self.expected_value], dtype=ms.uint32))
                self.ckpt.trigger_probe()
            except Exception as e:
                print(f"  [SPDK] Warning: pre-ckpt failed: {e}", flush=True)

    def on_train_step_end(self, run_context):
        step_time = (time.perf_counter() - self.step_start) * 1000
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == 1 and not self.has_registered:
            self.ckpt.register_tasks(self.model)
            self.ckpt.set_probe_flag_ptr(self.train_cell.flag)
            if ENABLE_PROFILING:
                try: self.ckpt.probe_flag_selftest()
                except Exception as e: print(f"  [SPDK] selftest warning: {e}", flush=True)
            self.has_registered = True
            print(f"  [SPDK] Tasks registered at step 1.", flush=True)
        if cur_step % CKPT_INTERVAL == 0 and cur_step > 3:
            t_flag0 = time.perf_counter()
            try:
                flag_val = int(self.train_cell.flag.asnumpy()[0])
                expected_val = int(self.train_cell.expected.asnumpy()[0])
                if flag_val < expected_val:
                    for _ in range(500):
                        time.sleep(0.01)
                        flag_val = int(self.train_cell.flag.asnumpy()[0])
                        if flag_val >= expected_val: break
            except Exception as e:
                print(f"  flag read error: {e}", flush=True)
            wait_ms = (time.perf_counter() - t_flag0) * 1000
            self.wait_times.append(wait_ms)
            print(f"  Step{cur_step:3d} step={step_time:.1f}ms flag_wait={wait_ms:.1f}ms", flush=True)
        elif cur_step > 3:
            self.step_times.append(step_time)
            if cur_step % 5 == 0:
                print(f"  Step{cur_step:3d} step={step_time:.1f}ms", flush=True)

    def end(self, run_context):
        self.ckpt.cleanup()
        if self.step_times:
            avg = np.mean(self.step_times)
            print(f"\n  Avg non-CKPT step: {avg:.1f}ms (n={len(self.step_times)})")
        if self.wait_times:
            avg_w = np.mean(self.wait_times)
            print(f"  Avg flag_wait:     {avg_w:.1f}ms")
        print("  Cleanup done.", flush=True)

def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    from mindformers import AutoModel, AutoConfig
    print("[SinkTest] Building model...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = 1024; cfg.max_position_embeddings = 1024
    base_model = AutoModel.from_config(cfg)
    total_bytes = sum(int(np.prod(p.shape)) * ms.dtype_to_nptype(p.dtype)().itemsize for p in base_model.get_parameters())
    print(f"[SinkTest] Model params: {total_bytes/1024/1024:.1f} MB", flush=True)

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    probe_wrapper = ProbeTrainOneStepCell(base_model, optimizer, None, 0, enable_probe=True, probe_mode="end")
    cb = SpdkCkptCallback(base_model, probe_wrapper)
    ms_model = ms.Model(probe_wrapper)

    print("\n[SinkTest] Starting with dataset_sink_mode=TRUE ...\n", flush=True)
    try:
        ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb], dataset_sink_mode=True)
    except Exception as e:
        print(f"[SinkTest] SINK=TRUE FAILED: {e}", flush=True)
        return 1
    print("\n[SinkTest] DONE.", flush=True)
    return 0

if __name__ == "__main__":
    main()
