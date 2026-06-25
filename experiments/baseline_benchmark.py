#!/usr/bin/env python3
"""
Baseline Checkpoint Bandwidth Benchmark — 5 methods (A–E) + raw NVMe (F).

Output: experiments/output/baseline_results.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /home/user7/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baseline_benchmark.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, pickle, shutil, argparse, warnings
from typing import List
from dataclasses import dataclass, asdict
import numpy as np
import mindspore as ms
from mindspore import nn, context, Callback, ops

warnings.filterwarnings("ignore")

MODEL_NAME    = "gpt2_xl"
SEQ_LEN       = 1024
BATCH_SIZE    = 1
DEVICE_ID     = 1
TRAIN_MR      = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
NVME_DIR      = "/models/baseline_test"
CKPT_INTERVAL = 10
WARMUP_STEPS  = 3
TOTAL_STEPS   = 35
RAW_FILE_GB   = 4.0
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "baseline_results.json")

# — data structures —
@dataclass
class RunResult:
    method: str
    total_params_mb: float = 0
    steps_tested: int = 0
    avg_step_ms: float = 0
    avg_ckpt_ms: float = 0
    p99_ckpt_ms: float = 0
    avg_bw_mbs: float = 0
    avg_step_with_ckpt_ms: float = 0

# — callback base —
class BaseCkptCallback(Callback):
    def __init__(self, model, total_bytes):
        super().__init__()
        self.model = model
        self.total_bytes = total_bytes
        self.step_times_no_ckpt = []
        self.step_times_with_ckpt = []
        self.ckpt_times = []
        self.ckpt_bws = []
        self.step_start = 0

    def on_train_step_begin(self, run_context):
        self.step_start = time.perf_counter()

    def on_train_step_end(self, run_context):
        step_time = (time.perf_counter() - self.step_start) * 1000
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % CKPT_INTERVAL == 0 and cur_step > WARMUP_STEPS:
            t0 = time.perf_counter()
            ckpt_bytes = self._do_checkpoint(cur_step)
            t_ckpt = (time.perf_counter() - t0) * 1000
            self.ckpt_times.append(t_ckpt)
            self.step_times_with_ckpt.append(step_time)
            if t_ckpt > 0 and ckpt_bytes > 0:
                self.ckpt_bws.append(ckpt_bytes / 1024 / 1024 / (t_ckpt / 1000))
        elif cur_step > WARMUP_STEPS:
            self.step_times_no_ckpt.append(step_time)
            if cur_step % 5 == 0:
                print(f"  Step {cur_step:3d} | step={step_time:.1f}ms", flush=True)

    def _do_checkpoint(self, step) -> int:
        raise NotImplementedError

    def get_result(self):
        return RunResult(
            method=self.method_name,
            total_params_mb=self.total_bytes / 1024 / 1024,
            steps_tested=len(self.ckpt_times),
            avg_step_ms=float(np.mean(self.step_times_no_ckpt)) if self.step_times_no_ckpt else 0,
            avg_ckpt_ms=float(np.mean(self.ckpt_times)) if self.ckpt_times else 0,
            p99_ckpt_ms=float(np.percentile(self.ckpt_times, 99)) if self.ckpt_times else 0,
            avg_bw_mbs=float(np.mean(self.ckpt_bws)) if self.ckpt_bws else 0,
            avg_step_with_ckpt_ms=float(np.mean(self.step_times_with_ckpt)) if self.step_times_with_ckpt else 0,
        )

# — methods A–E —
class Callback_A_Sync(BaseCkptCallback):
    method_name = "A_MS_save_ckpt_sync"
    def __init__(self, model, total_bytes, output_dir):
        super().__init__(model, total_bytes)
        self.path = os.path.join(output_dir, "A_sync.ckpt")
        os.makedirs(output_dir, exist_ok=True)
    def _do_checkpoint(self, step):
        ms.save_checkpoint(self.model, self.path, integrated_save=True, async_save=False)
        sz = os.path.getsize(self.path) if os.path.exists(self.path) else self.total_bytes
        if os.path.exists(self.path): os.remove(self.path)
        return sz

class Callback_B_Async(BaseCkptCallback):
    method_name = "B_MS_save_ckpt_async"
    def __init__(self, model, total_bytes, output_dir):
        super().__init__(model, total_bytes)
        self.path = os.path.join(output_dir, "B_async.ckpt")
        os.makedirs(output_dir, exist_ok=True)
    def _do_checkpoint(self, step):
        ms.save_checkpoint(self.model, self.path, integrated_save=True, async_save=True)
        prev = -1
        for _ in range(1200):
            time.sleep(0.1)
            if os.path.exists(self.path):
                cur = os.path.getsize(self.path)
                if cur > 0 and cur == prev: break
                prev = cur
        sz = os.path.getsize(self.path) if os.path.exists(self.path) else self.total_bytes
        if os.path.exists(self.path): os.remove(self.path)
        return sz

class Callback_C_Pickle(BaseCkptCallback):
    method_name = "C_asnumpy_pickle"
    def __init__(self, model, total_bytes, output_dir):
        super().__init__(model, total_bytes)
        self.path = os.path.join(output_dir, "C_pickle.pkl")
        os.makedirs(output_dir, exist_ok=True)
    def _do_checkpoint(self, step):
        state = {}
        for p in self.model.get_parameters():
            state[p.name] = p.asnumpy()
        with open(self.path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz = os.path.getsize(self.path)
        if os.path.exists(self.path): os.remove(self.path)
        return sz

class Callback_D_NpSave(BaseCkptCallback):
    method_name = "D_asnumpy_npsave"
    def __init__(self, model, total_bytes, output_dir):
        super().__init__(model, total_bytes)
        self.dir = os.path.join(output_dir, "D_npsave")
    def _do_checkpoint(self, step):
        if os.path.exists(self.dir): shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        for p in self.model.get_parameters():
            fname = p.name.replace("/", "_").replace(".", "_")
            np.save(os.path.join(self.dir, f"{fname}.npy"), p.asnumpy())
        total_file_sz = sum(os.path.getsize(os.path.join(self.dir, f)) for f in os.listdir(self.dir))
        if os.path.exists(self.dir): shutil.rmtree(self.dir)
        return total_file_sz

class Callback_E_Binary(BaseCkptCallback):
    method_name = "E_asnumpy_binary"
    def __init__(self, model, total_bytes, output_dir):
        super().__init__(model, total_bytes)
        self.path = os.path.join(output_dir, "E_binary.bin")
        os.makedirs(output_dir, exist_ok=True)
    def _do_checkpoint(self, step):
        buffers = [p.asnumpy() for p in self.model.get_parameters()]
        with open(self.path, "wb", buffering=128 * 1024 * 1024) as f:
            for buf in buffers:
                f.write(buf.tobytes())
        sz = os.path.getsize(self.path)
        if os.path.exists(self.path): os.remove(self.path)
        return sz

# — raw NVMe bench (F) —
def bench_raw_nvme_write(output_dir, file_size_gb=4.0):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    print(f"\n{'='*60}\n[F_raw_nvme_bench] Pure NVMe sequential write benchmark\n{'='*60}")
    block_sizes_mb = [1, 4, 16, 64, 256]
    data = np.random.bytes(256 * 1024 * 1024)
    for bs_mb in block_sizes_mb:
        bs = bs_mb * 1024 * 1024
        total = int(file_size_gb * 1024**3)
        filepath = os.path.join(output_dir, f"F_raw_{bs_mb}MB.tmp")
        with open(filepath, "wb", buffering=128 * 1024 * 1024) as f:
            w = 0
            while w < min(total, bs * 2):
                chunk = data[:min(bs, min(total, bs * 2) - w)]
                f.write(chunk); w += len(chunk)
        os.remove(filepath)
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except: pass
        time.sleep(0.5)
        t0 = time.perf_counter()
        with open(filepath, "wb", buffering=128 * 1024 * 1024) as f:
            w = 0
            while w < total:
                chunk = data[:min(bs, total - w)]
                f.write(chunk); w += len(chunk)
        elapsed = time.perf_counter() - t0
        sz = os.path.getsize(filepath)
        bw = sz / 1024 / 1024 / elapsed
        os.remove(filepath)
        results.append({"block_mb": bs_mb, "size_gb": sz/1024**3, "elapsed_s": round(elapsed,3), "bw_mbs": round(bw,1)})
        print(f"  block={bs_mb:4d}MB | {sz/1024**3:.1f}GB in {elapsed:.2f}s | {bw:.1f} MB/s")
    return results

# — training runner —
def run_single_method(method_name, CallbackClass, model, train_cell, train_ds, total_bytes, output_dir):
    print(f"\n{'='*60}\n[{method_name}] Running...\n{'='*60}", flush=True)
    cb = CallbackClass(model, total_bytes, output_dir)
    ms_model = ms.Model(train_cell)
    ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb], dataset_sink_mode=False)
    return cb.get_result()

def build_training():
    from mindformers import AutoModel, AutoConfig
    print("[Setup] Loading config and building model...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    base_model = AutoModel.from_config(cfg)
    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    class TrainOneStep(nn.Cell):
        def __init__(self, net, opt):
            super().__init__(auto_prefix=False)
            self.net = net; self.net.set_grad()
            self.opt = opt
            self.grad_fn = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        def construct(self, *inputs):
            if len(inputs) == 1: inputs = inputs[0]
            loss, grads = self.grad_fn(*inputs)
            opt_res = self.opt(grads)
            return ops.depend(loss, opt_res)
    train_cell = TrainOneStep(base_model, optimizer)
    def count_params(model):
        t = 0
        for p in model.get_parameters():
            t += int(np.prod(p.shape)) * ms.dtype_to_nptype(p.dtype)().itemsize
        return t
    total_bytes = count_params(base_model)
    print(f"[Setup] Model params: {total_bytes/1024/1024:.1f} MB", flush=True)
    return base_model, train_cell, total_bytes

def make_dataset():
    ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.take(TOTAL_STEPS)
    return ds

def main():
    parser = argparse.ArgumentParser(description="Baseline Checkpoint Benchmark")
    parser.add_argument("--methods", nargs="+", default=["A","B","C","D","E","F"])
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--output", type=str, default=NVME_DIR)
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
    ms.common.set_seed(42)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    callbacks_map = {
        "A": ("A_MS_save_ckpt_sync",  Callback_A_Sync),
        "B": ("B_MS_save_ckpt_async", Callback_B_Async),
        "C": ("C_asnumpy_pickle",     Callback_C_Pickle),
        "D": ("D_asnumpy_npsave",     Callback_D_NpSave),
        "E": ("E_asnumpy_binary",     Callback_E_Binary),
    }

    all_results = []
    raw_results = []

    need_model = bool(set(args.methods) & {"A","B","C","D","E"})
    if need_model:
        model, train_cell, total_bytes = build_training()

    for m in args.methods:
        if m == "F":
            raw_results = bench_raw_nvme_write(output_dir, RAW_FILE_GB)
            continue
        name, Cls = callbacks_map[m]
        train_ds = make_dataset()
        result = run_single_method(name, Cls, model, train_cell, train_ds, total_bytes, output_dir)
        all_results.append(result)
        time.sleep(1)

    print(f"\n\n{'='*85}")
    print(f"{'BASELINE CHECKPOINT BENCHMARK — FINAL REPORT':^85}")
    print(f"{'='*85}")
    hdr = f"{'Method':<28} {'Params':>9} {'Steps':>6} {'AvgStep':>9} {'AvgCkpt':>9} {'P99Ckpt':>9} {'BW':>10} {'Step+Ckpt':>10} {'Overhead':>9}"
    print(hdr); print("-"*100)
    for r in all_results:
        overhead = ((r.avg_step_with_ckpt_ms - r.avg_step_ms)/r.avg_step_ms*100) if r.avg_step_ms > 0 else 0
        print(f"{r.method:<28} {r.total_params_mb:>7.1f}MB {r.steps_tested:>5}  "
              f"{r.avg_step_ms:>7.1f}ms {r.avg_ckpt_ms:>7.1f}ms {r.p99_ckpt_ms:>7.1f}ms "
              f"{r.avg_bw_mbs:>8.1f}MB/s {r.avg_step_with_ckpt_ms:>8.1f}ms {overhead:>7.1f}%")
    if raw_results:
        print(f"\n{'='*85}\n{'RAW NVMe Sequential Write (F)':^85}\n{'='*85}")
        print(f"{'Block':>12} {'Size':>10} {'Time':>10} {'BW':>12}")
        print("-"*45)
        for r in raw_results:
            print(f"{r['block_mb']:>9} MB  {r['size_gb']:>8.2f}GB  {r['elapsed_s']:>8.2f}s  {r['bw_mbs']:>10.1f} MB/s")
    print(f"\n{'='*85}\n  SPDK NPU→NVMe direct (reference):  ~4200 MB/s\n{'='*85}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        "config": {
            "model": MODEL_NAME, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "device_id": args.device_id,
            "nvme_84": "/dev/nvme1n1 (0000:84:00.0)", "mount": "/models", "fs": "xfs",
            "nvme_83_spdk": "0000:83:00.0 (SPDK exclusive)",
            "ckpt_interval": CKPT_INTERVAL, "warmup_steps": WARMUP_STEPS, "total_steps": TOTAL_STEPS,
        },
        "methods": [asdict(r) for r in all_results],
        "raw_nvme_bench": raw_results,
        "spdk_reference_bw_mbs": 4200,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] JSON → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
