#!/usr/bin/env python3
"""
GPT-2 XL training with WaitProbe + SPDK checkpoint (single card).

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/train_gpt2_spdk.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time
import mindspore as ms
from mindspore import nn, context, ops
from mindformers import AutoModel, AutoTokenizer, AutoConfig

import direct_checkpoint
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell

MODEL_NAME         = "gpt2_xl"
SEQ_LEN            = 1024
BATCH_SIZE         = 1
DEVICE_ID          = 1
TRAIN_MR           = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
CHECKPOINT_INTERVAL = 10
NVME_ADDR          = "0000:83:00.0"
PIPELINE_DEPTH     = 8
CHUNK_SIZE         = 4 * 1024 * 1024
ENABLE_PROFILING   = True
KEEP_LAST_N        = 3
SLOT_SIZE_GB       = 10


class DirectCkptCallback(ms.Callback):
    def __init__(self, model: ms.nn.Cell, train_cell: ms.nn.Cell):
        super().__init__()
        self.model = model; self.train_cell = train_cell
        self.has_registered = False; self.step_start_time = 0
        self.assign = ops.Assign(); self.expected_value = 0
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR, npu_device_id=DEVICE_ID,
            pipeline_depth=PIPELINE_DEPTH, requested_chunk_size=CHUNK_SIZE,
            enable_profiling=ENABLE_PROFILING, keep_last_n=KEEP_LAST_N,
            slot_size_gb=SLOT_SIZE_GB)

    def on_train_step_begin(self, run_context):
        self.step_start_time = time.perf_counter()
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if ENABLE_PROFILING:
            print(f"   [Timeline] Step {cur_step} begin ts={time.time():.6f}")
        if not self.has_registered: return
        if cur_step % CHECKPOINT_INTERVAL == 0:
            try:
                self.expected_value += 1
                self.assign(self.train_cell.expected, ms.Tensor([self.expected_value], dtype=ms.uint32))
            except Exception as e:
                print(f"   [DirectCkpt] Warning: set expected failed: {e}")
            try:
                self.ckpt.trigger_probe()
                print(f"   [DirectCkpt] probe triggered at step {cur_step}")
            except Exception as e:
                print(f"   [DirectCkpt] Warning: trigger_probe failed: {e}")

    def on_train_step_end(self, run_context):
        step_time_ms = (time.perf_counter() - self.step_start_time) * 1000
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == 1 and not self.has_registered:
            self.ckpt.register_tasks(self.model)
            self.ckpt.set_probe_flag_ptr(self.train_cell.flag)
            flag_ptr = direct_checkpoint.get_dev_ptr(self.train_cell.flag)
            print(f"   [DirectCkpt] Probe flag ptr set: 0x{flag_ptr:x}")
            if ENABLE_PROFILING:
                try: self.ckpt.probe_flag_selftest()
                except Exception as e: print(f"   [DirectCkpt] Warning: selftest failed: {e}")
            self.has_registered = True
        if cur_step % CHECKPOINT_INTERVAL == 0:
            print(f"   [DirectCkpt] Step {cur_step}: Checkpoint interval reached.")
        print(f"   [Profiler] Step {cur_step} Time: {step_time_ms:.2f} ms")
        if cur_step % CHECKPOINT_INTERVAL == 0:
            try:
                t0 = time.perf_counter()
                flag_val = int(self.train_cell.flag.asnumpy()[0])
                expected_val = int(self.train_cell.expected.asnumpy()[0])
                if flag_val < expected_val:
                    for _ in range(200):
                        time.sleep(0.01)
                        flag_val = int(self.train_cell.flag.asnumpy()[0])
                        if flag_val >= expected_val: break
                dt_ms = (time.perf_counter() - t0) * 1000
                dev_flag = self.ckpt.read_probe_flag_dev()
                print(f"   [DirectCkpt] flag after step {cur_step}: {flag_val}, expected={expected_val} (wait {dt_ms:.2f} ms), dev={dev_flag}")
            except Exception as e:
                print(f"   [DirectCkpt] Warning: read flag failed: {e}")

    def end(self, run_context):
        self.ckpt.cleanup()
        print("[DirectCkpt] cleanup done", flush=True)


def build_trainer():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    print(f"[Debug] WAITPROBE_NO_RESET={os.getenv('WAITPROBE_NO_RESET','')}")
    print(f"[Debug] ASCEND_OPP_PATH={os.getenv('ASCEND_OPP_PATH','')}")

    print("[Setup] Loading Model Config...")
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    base_model = AutoModel.from_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = SEQ_LEN

    print("[Setup] Loading Dataset...")
    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.take(150)

    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    print("[Setup] Injecting AICPU Probe Wrapper...")
    probe_wrapper = ProbeTrainOneStepCell(base_model, optimizer, None, 0, enable_probe=True, probe_mode="end")

    cb = DirectCkptCallback(base_model, probe_wrapper)
    ms_model = ms.Model(probe_wrapper)
    print("\nStarting SPDK + WaitProbe Training Loop...\n")
    ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb], dataset_sink_mode=False)


if __name__ == "__main__":
    build_trainer()
