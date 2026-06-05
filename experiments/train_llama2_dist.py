import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
"""Distributed LLaMA2 training script with NVMe checkpointing support.

Usage:
- python python/train_llama2_dist.py (distributed launcher as needed)

Inputs:
- MindRecord dataset under dataset_prepare/.
Outputs:
- Training logs under output/ and NVMe checkpoint metadata.
"""
# train_llama2_dist.py
import mindspore as ms
from mindspore import context
from mindspore.communication import init, get_rank, get_group_size
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig
import time
import os
import shutil
import concurrent.futures
import numpy as np
from mindspore import Callback, ops
from direct_checkpoint import DirectCheckpoint
from fast_init import replace_with_noop_initializer
import psutil
import threading
import csv

class MemoryMonitor:
    def __init__(self, output_dir, rank_id):
        self.output_dir = output_dir
        self.rank_id = rank_id
        self.history = []
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
    def _run(self):
        p = psutil.Process()
        while self.running:
            try:
                rss = p.memory_info().rss / 1024 / 1024 # MB
                self.history.append((time.time(), rss))
                time.sleep(0.1) 
            except:
                break
                
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.save()
        
    def save(self):
        if not self.history: 
            return
        peak = max(h[1] for h in self.history)
        print(f"[MemoryMonitor] Rank {self.rank_id} Peak RSS: {peak:.2f} MB", flush=True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"memory_rank_{self.rank_id}.csv")
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'rss_mb'])
                writer.writerows(self.history)
            print(f"[MemoryMonitor] Saved trace to {path}", flush=True)
        except Exception as e:
            print(f"[MemoryMonitor] Failed to save trace: {e}", flush=True)

class LossLogger(Callback):
    def __init__(self, rank_id, world_size, monitor=None):
        self.rank_id = rank_id
        self.world_size = world_size
        self.monitor = monitor
        self.is_last_stage = (rank_id >= (world_size - world_size // PIPELINE_PARALLEL))
        self.log_file = None
        
        if self.is_last_stage or ENABLE_PROFILING:
            if ENABLE_PROFILING:
                os.makedirs(PROFILING_OUTPUT_DIR, exist_ok=True)
                self.log_file = os.path.join(PROFILING_OUTPUT_DIR, f"loss_curve_rank{rank_id}.csv")
            else:
                self.log_file = f"loss_curve_rank{rank_id}.csv"
            
            with open(self.log_file, "w") as f:
                f.write("step,loss\n")
    
    def step_end(self, run_context):
        if not (self.is_last_stage or ENABLE_PROFILING):
            return

        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        loss = cb_params.net_outputs
        
        actual_loss = loss
        if isinstance(loss, (tuple, list)):
            actual_loss = loss[0]
        if hasattr(actual_loss, "asnumpy"):
            actual_loss = actual_loss.asnumpy()
        
        if hasattr(actual_loss, "item"):
            actual_loss = actual_loss.item()
            
        if step % 1 == 0:
            print(f"[LossLogger][rank {self.rank_id}] step={step} loss={actual_loss:.6f}", flush=True)
            with open(self.log_file, "a") as f:
                f.write(f"{step},{actual_loss}\n")

        if step == 1 and self.monitor and self.monitor.running:
            print(f"[LossLogger][rank {self.rank_id}] Step 1 finished, stopping memory monitor...", flush=True)
            self.monitor.stop()

def _ensure_ascend_env():
    mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK") or os.getenv("PMI_RANK")
    mpi_size = os.getenv("OMPI_COMM_WORLD_SIZE") or os.getenv("PMI_SIZE")
    if mpi_rank is not None:
        os.environ.setdefault("RANK_ID", mpi_rank)
        os.environ.setdefault("DEVICE_ID", mpi_rank)
        os.environ.setdefault("ASCEND_DEVICE_ID", mpi_rank)
    if mpi_size is not None:
        os.environ.setdefault("RANK_SIZE", mpi_size)
        os.environ.setdefault("DEVICE_NUM", mpi_size)


# ============================================================
# 配置区域 (纯同步、稳定测试配置)
# ============================================================
MODEL_NAME = "llama2_7b" 
USE_FAST_INIT = False              
FAST_INIT_CKPT_DIR = "./checkpoint_meta"          
ENABLE_PROFILING = True            
PROFILING_OUTPUT_DIR = "./output/profiling"  
META_OUTPUT_DIR = "./checkpoint_meta"             
SAVE_OPTIMIZER = False             
ASYNC_SAVE = False                 

SEQ_LEN = 4096
BATCH_SIZE = 2                  
GRAD_ACCUM_STEPS = 2            
TRAIN_MR = "./dataset_prepare/llama2/wikitext2_data/wiki_train_4096.mindrecord"
EVAL_MR  = "./dataset_prepare/llama2/wikitext2_data/wiki_valid_4096.mindrecord"
CHECKPOINT_INTERVAL = 5 
NVME_ADDR = "0000:83:00.0"
PIPELINE_DEPTH = 8 
CHUNK_SIZE = 1 * 1024 * 1024
SPDK_SHM_ID = int(os.getenv("SPDK_SHM_ID", "1"))  

# 存储槽位配置
SLOT_SIZE_GB = 10 
KEEP_LAST_N = 3

# 并行策略
DATA_PARALLEL = 1
MODEL_PARALLEL = 2
PIPELINE_PARALLEL = 2
TRAIN_STEPS = 100 

ms.set_seed(1024)
context.set_context(deterministic='ON')


class DirectCkptCallback(Callback):
    def __init__(self, model: ms.nn.Cell, rank_id: int, world_size: int, ckpt_manager=None, resume_meta_path=None, ckpt_meta_path=None):
        super().__init__()
        self.rank_id = rank_id
        self.world_size = world_size
        self.resume_meta_path = resume_meta_path
        self.ckpt_meta_path = ckpt_meta_path or f"checkpoint_meta_rank{rank_id}.pkl"
        self.optimizer = None
        
        local_device_id = rank_id % 8
        
        if ckpt_manager is None:
            self.ckpt = DirectCheckpoint(
                nvme_addr=NVME_ADDR,
                npu_device_id=local_device_id,
                pipeline_depth=PIPELINE_DEPTH,
                requested_chunk_size=CHUNK_SIZE,
                enable_profiling=ENABLE_PROFILING,
                profiling_dir=PROFILING_OUTPUT_DIR,
                rank_id=rank_id,
                world_size=world_size,
                slot_size_gb=SLOT_SIZE_GB,
                keep_last_n=KEEP_LAST_N,
                spdk_shm_id=SPDK_SHM_ID,
            )
        else:
            self.ckpt = ckpt_manager
        self.model = model

    def set_compilation_start(self, t):
        self.compilation_start_time = t
        self.has_logged_timeline = False

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
            
        t_now = time.perf_counter()
        print(f"[Timeline][Rank {self.rank_id}] Step {step_num} | Forward pass STARTED at {t_now:.3f}s", flush=True)

        if not getattr(self, "has_logged_timeline", False):
            self.has_logged_timeline = True
            compilation_end_time = time.time()
            if ENABLE_PROFILING and hasattr(self, "compilation_start_time"):
                 try:
                     comp_dur = compilation_end_time - self.compilation_start_time
                     print(f"[DirectCkpt][Timeline] Compilation={comp_dur:.2f}s", flush=True)
                 except Exception:
                     pass

    def step_end(self, run_context):
        cbp = run_context.original_args()
        step = cbp.cur_step_num
        if step % CHECKPOINT_INTERVAL != 0:
            return
            
        targets = [self.model]
            
        # 1. 触发主线程 Layout 并在后台线程发射异步写入
        # 注意：这里将 commit_meta 设置为 True（限 rank_id==0 时），
        # 确保元数据是由后台线程写完参数后再盖章的，而不是在回调里强制写。
        save_results = self.ckpt.save(
            targets,
            step=step,
            meta_path=self.ckpt_meta_path,
            async_save=ASYNC_SAVE,
            commit_meta=(self.rank_id == 0) 
        )

        # 2. 因为此时主线程瞬间返回了，我们直接打点并释放控制权给 MindSpore
        print(f"[Timeline][Rank {self.rank_id}] Step {step} | step_end hook finished. Unblocking training immediately!", flush=True)

    def on_train_step_end(self, run_context):
        return self.step_end(run_context)

    def end(self, run_context):
        self.ckpt.cleanup()
        print(f"[DirectCkpt][rank {self.rank_id}] cleanup done", flush=True)

class TrainStepControl(Callback):
    def __init__(self, steps):
        self.steps = steps

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        if self.steps > 0 and step >= self.steps:
            run_context.request_stop()

def main():
    _ensure_ascend_env()
    
    rank_env = int(os.getenv("RANK_ID", "0"))
    monitor = MemoryMonitor(PROFILING_OUTPUT_DIR, rank_env)
    monitor.start()

    pre_rank_id = int(os.getenv("RANK_ID", "0"))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=pre_rank_id)
    init()
    rank_id = get_rank()
    monitor.rank_id = rank_id 
    rank_size = get_group_size()

    assert rank_size == DATA_PARALLEL * MODEL_PARALLEL * PIPELINE_PARALLEL

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=rank_id)
    context.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
        device_num=rank_size,
        gradients_mean=True,
        pipeline_stages=PIPELINE_PARALLEL,
        full_batch=True,            
        enable_parallel_optimizer=True,
    )

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = MODEL_NAME
    cfg.compute_dtype = ms.float16
    cfg.param_init_type = ms.float16
    
    # 提前关闭 cfg 内部保存开关
    if hasattr(cfg, "runner_config") and cfg.runner_config is not None:
        cfg.runner_config.save_checkpoint = False

    cfg.parallel_config = cfg.parallel_config if hasattr(cfg, "parallel_config") else None
    if cfg.parallel_config:
        cfg.parallel_config.data_parallel = DATA_PARALLEL
        cfg.parallel_config.model_parallel = MODEL_PARALLEL
        cfg.parallel_config.pipeline_stage = PIPELINE_PARALLEL
        cfg.parallel_config.micro_batch_num = GRAD_ACCUM_STEPS
        cfg.parallel_config.vocab_emb_dp = False  
        cfg.parallel_config.use_seq_parallel = True

    model = AutoModel.from_config(cfg)
    
    os.makedirs(META_OUTPUT_DIR, exist_ok=True)
    ckpt_meta_path = os.path.join(META_OUTPUT_DIR, f"checkpoint_meta_rank{rank_id}.pkl")
    
    ckpt_manager = DirectCheckpoint(
        nvme_addr=NVME_ADDR,
        npu_device_id=rank_id % 8, 
        pipeline_depth=PIPELINE_DEPTH,
        requested_chunk_size=CHUNK_SIZE,
        enable_profiling=ENABLE_PROFILING,
        profiling_dir=PROFILING_OUTPUT_DIR,
        rank_id=rank_id,
        world_size=rank_size,
        slot_size_gb=SLOT_SIZE_GB,
        keep_last_n=KEEP_LAST_N,
        spdk_shm_id=SPDK_SHM_ID 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = SEQ_LEN

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True,
                                      num_shards=DATA_PARALLEL, shard_id=rank_id % DATA_PARALLEL)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

    eval_ds = ms.dataset.MindDataset(EVAL_MR, shuffle=False,
                                     num_shards=DATA_PARALLEL, shard_id=rank_id % DATA_PARALLEL)
    eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

    dc_cb = DirectCkptCallback(model, rank_id=rank_id, world_size=rank_size, ckpt_manager=ckpt_manager, ckpt_meta_path=ckpt_meta_path)
    loss_cb = LossLogger(rank_id, world_size=rank_size, monitor=monitor)
    cbs = [dc_cb, loss_cb]
    
    if TRAIN_STEPS > 0:
        cbs.append(TrainStepControl(TRAIN_STEPS))

    trainer = Trainer(
        task="text_generation",
        model=model,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=cbs, 
    )

    # 物理剥离框架暗中塞入的原生 Checkpoint 回调，防止文件系统死锁
    clean_callbacks = []
    for cb in trainer.callbacks:
        cb_name = type(cb).__name__
        if "Checkpoint" in cb_name and "DirectCkptCallback" not in cb_name:
            print(f"[Main][Rank {rank_id}] Brutally removed native callback: {cb_name}", flush=True)
            continue
        clean_callbacks.append(cb)
    trainer.callbacks = clean_callbacks

    pc = trainer.config.parallel_config
    pc.data_parallel = DATA_PARALLEL
    pc.model_parallel = MODEL_PARALLEL
    pc.pipeline_stage = PIPELINE_PARALLEL
    pc.context_parallel = 1
    pc.vocab_emb_dp = False
    
    pc.use_seq_parallel = True
    pc.micro_batch_num = GRAD_ACCUM_STEPS
    
    # 滞后配置覆盖，彻底杜绝数据下沉和原生保存
    trainer.config.save_checkpoint = False
    if trainer.config.runner_config:
        trainer.config.runner_config.save_checkpoint = False
        trainer.config.runner_config.sink_mode = False
        trainer.config.runner_config.gradient_accumulation_steps = GRAD_ACCUM_STEPS
        trainer.config.runner_config.device_num = rank_size

    dc_cb.set_compilation_start(time.time())
    
    print(f"[Main][Rank {rank_id}] Launching training in sync mode...", flush=True)
    trainer.train(do_eval=False)

if __name__ == "__main__":
    main()
    if os.path.exists('./output/checkpoint'): shutil.rmtree('./output/checkpoint')
    if os.path.exists('./output/checkpoint_network'): shutil.rmtree('./output/checkpoint_network')
    if os.path.exists('./output/strategy'): shutil.rmtree('./output/strategy')