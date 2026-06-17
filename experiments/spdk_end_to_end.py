#!/usr/bin/env python3
"""
SPDK NPU→NVMe Checkpoint End-to-End Benchmark.

Output: experiments/output/spdk_results.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/spdk_end_to_end.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, warnings, ctypes
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", "python"))

import direct_checkpoint as dc
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell, get_dev_ptr

MODEL_NAME       = "gpt2_xl"
SEQ_LEN          = 1024
BATCH_SIZE       = 1
DEVICE_ID        = 1
TRAIN_MR         = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
NVME_ADDR        = "0000:83:00.0"
PIPELINE_DEPTH   = 8
CHUNK_SIZE       = 4 * 1024 * 1024
ENABLE_PROFILING = True
KEEP_LAST_N      = 3
SLOT_SIZE_GB     = 10
CKPT_INTERVAL    = 10
WARMUP_STEPS     = 3
TOTAL_STEPS      = 35

OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "output")
RESULTS_FILE     = os.path.join(OUTPUT_DIR, "spdk_results.json")

class TimingRecorder:
    def __init__(self):
        self.non_ckpt_step_times = []
        self.ckpt_step_times = []
        self.ckpt_wait_times = []
        self.ckpt_file_sizes = []
        self.bw_pipeline_mbs = []
    def to_dict(self):
        return {
            "non_ckpt_step_times_ms": self.non_ckpt_step_times,
            "ckpt_step_times_ms": self.ckpt_step_times,
            "ckpt_wait_times_ms": self.ckpt_wait_times,
            "ckpt_file_sizes_mb": [sz/1024/1024 for sz in self.ckpt_file_sizes],
            "bw_pipeline_mbs": self.bw_pipeline_mbs,
        }

recorder = TimingRecorder()

class SpdkCkptCallback(ms.Callback):
    def __init__(self, model, train_cell, warmup_fn=None):
        super().__init__()
        self.model = model; self.train_cell = train_cell
        self.has_registered = False; self.step_start = 0
        self.assign = ops.Assign(); self.expected_value = 0
        print("[SPDK] Initializing DirectCheckpoint...", flush=True)
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR, npu_device_id=DEVICE_ID,
            pipeline_depth=PIPELINE_DEPTH, requested_chunk_size=CHUNK_SIZE,
            enable_profiling=ENABLE_PROFILING, keep_last_n=KEEP_LAST_N,
            slot_size_gb=SLOT_SIZE_GB, warmup_fn=warmup_fn)
        self.total_bytes = self.ckpt.total_bytes
        print(f"[SPDK] NVMe total bytes: {self.total_bytes/1024**3:.2f} GB", flush=True)
        self.prof_dir = self.ckpt.profiling_dir

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
            # Try to set flag pointer from MS tensor; fall back to C-layer allocation
            ptr = get_dev_ptr(self.train_cell.flag)
            if ptr != 0:
                self.ckpt.set_probe_flag_ptr(self.train_cell.flag)
            else:
                # sink=False: MS hasn't allocated the tensor yet, use C-layer fallback
                rc = dc.lib.npu_nvme_set_probe_flag_ptr(self.ckpt.ctx, ctypes.c_void_p(0))
                if rc == 0:
                    dev_flag = dc.lib.npu_nvme_get_probe_flag_dev_ptr(self.ckpt.ctx)
                    self.ckpt.probe_flag_ptr = dev_flag
                    print(f"  [SPDK] Using C-layer allocated flag ptr: {hex(dev_flag)}", flush=True)
                else:
                    print(f"  [SPDK] Warning: C-layer flag alloc failed, rc={rc}", flush=True)
            if ENABLE_PROFILING:
                try: self.ckpt.probe_flag_selftest()
                except Exception as e: print(f"  [SPDK] selftest warning: {e}", flush=True)
            self.has_registered = True
            print(f"  [SPDK] Tasks registered at step 1.", flush=True)
        if cur_step % CKPT_INTERVAL == 0 and cur_step > WARMUP_STEPS:
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
                print(f"  [SPDK] flag read error: {e}", flush=True)
            flag_wait_ms = (time.perf_counter() - t_flag0) * 1000
            recorder.ckpt_step_times.append(step_time)
            recorder.ckpt_wait_times.append(flag_wait_ms)
            print(f"  [SPDK] Step {cur_step:3d} | step={step_time:.1f}ms | flag_wait={flag_wait_ms:.1f}ms", flush=True)
        elif cur_step > WARMUP_STEPS:
            recorder.non_ckpt_step_times.append(step_time)
            if cur_step % 5 == 0:
                print(f"  [SPDK] Step {cur_step:3d} | step={step_time:.1f}ms", flush=True)
        if cur_step == 1:
            print(f"  [SPDK] Step   1 | step={step_time:.1f}ms (compile + init)", flush=True)

    def end(self, run_context):
        print("[SPDK] Cleanup...", flush=True)
        self.ckpt.cleanup()
        print("[SPDK] Cleanup done.", flush=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    print(f"[SPDK] Env: ASCEND_OPP_PATH={os.getenv('ASCEND_OPP_PATH','')}", flush=True)
    from mindformers import AutoModel, AutoConfig
    print("[SPDK] Building model...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    base_model = AutoModel.from_config(cfg)
    total_bytes = 0
    for p in base_model.get_parameters():
        total_bytes += int(np.prod(p.shape)) * ms.dtype_to_nptype(p.dtype)().itemsize
    print(f"[SPDK] Model params: {total_bytes/1024/1024:.1f} MB", flush=True)
    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.take(TOTAL_STEPS)
    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    probe_wrapper = ProbeTrainOneStepCell(base_model, optimizer, None, 0, enable_probe=True, probe_mode="end")

    import mindspore as ms_t
    dummy_input = ms_t.Tensor(np.zeros((1, SEQ_LEN,), dtype=np.int32), ms.int32)
    warmup_fn = lambda: [probe_wrapper(dummy_input[0:1], dummy_input[0:1], dummy_input[0:1]) for _ in range(2)]

    cb = SpdkCkptCallback(base_model, probe_wrapper, warmup_fn=warmup_fn)
    ms_model = ms.Model(probe_wrapper)
    print(f"\n[SPDK] Starting training ({TOTAL_STEPS} steps)...\n", flush=True)
    t_train_start = time.perf_counter()
    ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb], dataset_sink_mode=False)
    t_train_total = (time.perf_counter() - t_train_start)

    avg_non_ckpt = float(np.mean(recorder.non_ckpt_step_times)) if recorder.non_ckpt_step_times else 0
    avg_ckpt_step = float(np.mean(recorder.ckpt_step_times)) if recorder.ckpt_step_times else 0
    avg_wait = float(np.mean(recorder.ckpt_wait_times)) if recorder.ckpt_wait_times else 0
    p99_wait = float(np.percentile(recorder.ckpt_wait_times, 99)) if recorder.ckpt_wait_times else 0
    overhead = ((avg_ckpt_step - avg_non_ckpt)/avg_non_ckpt*100) if avg_non_ckpt > 0 else 0

    spdk_bw_mbs = 0; spdk_total_s = 0
    try:
        csv_path = os.path.join(cb.prof_dir, "time_write.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                lines = f.readlines()
            buf_indices = set()
            total_us_pipeline = 0
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    buf = parts[1]; total_e2e_us = float(parts[4])
                    if buf not in buf_indices: total_us_pipeline = max(total_us_pipeline, total_e2e_us)
                    buf_indices.add(buf)
            spdk_total_s = total_us_pipeline / 1_000_000.0
            if spdk_total_s > 0: spdk_bw_mbs = (total_bytes/1024/1024) / spdk_total_s
            print(f"\n[SPDK] CSV pipeline_e2e={spdk_total_s*1000:.1f}ms, bw={spdk_bw_mbs:.1f}MB/s", flush=True)
    except Exception as e:
        print(f"[SPDK] Warning: could not parse profiling CSV: {e}", flush=True)

    print(f"\n{'='*75}\n{'SPDK NPU→NVMe CHECKPOINT — RESULTS':^75}\n{'='*75}")
    print(f"  Avg non-CKPT step:    {avg_non_ckpt:.1f} ms")
    print(f"  Avg CKPT step:        {avg_ckpt_step:.1f} ms")
    print(f"  Avg flag wait:        {avg_wait:.1f} ms")
    print(f"  P99 flag wait:        {p99_wait:.1f} ms")
    print(f"  Step overhead:        {overhead:+.1f}%")
    print(f"  Pipeline BW (CSV):    {spdk_bw_mbs:.1f} MB/s")
    print(f"  Pipeline time (CSV):  {spdk_total_s*1000:.1f} ms")
    print(f"{'='*75}")

    result = {
        "config": {
            "model": MODEL_NAME, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "device_id": DEVICE_ID, "nvme_addr": NVME_ADDR, "nvme_method": "SPDK userspace",
            "pipeline_depth": PIPELINE_DEPTH, "chunk_size": CHUNK_SIZE,
            "ckpt_interval": CKPT_INTERVAL, "warmup_steps": WARMUP_STEPS,
            "total_steps": TOTAL_STEPS, "total_params_mb": total_bytes/1024/1024,
        },
        "results": {
            "avg_non_ckpt_step_ms": round(avg_non_ckpt,2),
            "avg_ckpt_step_ms": round(avg_ckpt_step,2),
            "avg_flag_wait_ms": round(avg_wait,2),
            "p99_flag_wait_ms": round(p99_wait,2),
            "step_overhead_pct": round(overhead,2),
            "pipeline_bw_mbs": round(spdk_bw_mbs,1),
            "pipeline_time_ms": round(spdk_total_s*1000,1),
            "total_training_time_s": round(t_train_total,1),
        },
        "recorder": recorder.to_dict(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SPDK] Results saved to {RESULTS_FILE}")
    print("\n[Done] SPDK benchmark complete.")

if __name__ == "__main__":
    main()
