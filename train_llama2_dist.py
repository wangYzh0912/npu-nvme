# train_llama2_dist.py
import mindspore as ms
from mindspore import context
from mindspore.communication import init, get_rank, get_group_size
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig
import time
import os
import numpy as np
from mindspore import Callback, ops
from direct_checkpoint import DirectCheckpoint
from fast_init import replace_with_noop_initializer


class LossLogger(Callback):
    def __init__(self, rank_id, world_size):
        self.rank_id = rank_id
        self.world_size = world_size
        self.is_last_stage = (rank_id >= (world_size - world_size // PIPELINE_PARALLEL))
        if self.is_last_stage:
            log_file = f"loss_curve_rank{rank_id}.csv"
            with open(log_file, "w") as f:
                f.write("step,loss\n")
    
    def step_end(self, run_context):
        if not self.is_last_stage:
            return

        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        loss = cb_params.net_outputs
        
        # Handle loss which might be a tuple or Tensor
        actual_loss = loss
        if isinstance(loss, (tuple, list)):
            actual_loss = loss[0]
        if hasattr(actual_loss, "asnumpy"):
            actual_loss = actual_loss.asnumpy()
        
        # Extract scalar
        if hasattr(actual_loss, "item"):
            actual_loss = actual_loss.item()
            
        # Print periodically or every step
        if step % 1 == 0:
            print(f"[LossLogger][rank {self.rank_id}] step={step} loss={actual_loss:.6f}", flush=True)
            with open(f"loss_curve_rank{self.rank_id}.csv", "a") as f:
                f.write(f"{step},{actual_loss}\n")

# Some launchers (e.g. mpirun) do not export Ascend envs by default; derive them from MPI vars.
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

# ----------------------
# 4 卡（64GB/卡）示例配置
# 拆分方案：dp=1, mp=2, pp=2  => 总卡数 1*2*2=4
# micro_batch=1（可根据显存再调小或用梯度累积）
# ----------------------
MODEL_NAME = "llama2_7b"       # 实际权重名/本地路径
SEQ_LEN = 4096
BATCH_SIZE = 2                  # per-device batch；需能整除 micro_batch_num
GRAD_ACCUM_STEPS = 2            # micro batch 个数需 >= pipeline_stages（这里 pp=2）
TRAIN_MR = "./prepare/llama2/wikitext2_data/wiki_train_4096.mindrecord"
EVAL_MR  = "./prepare/llama2/wikitext2_data/wiki_valid_4096.mindrecord"
CHECKPOINT_INTERVAL = 5
NVME_ADDR = "0000:83:00.0"
PIPELINE_DEPTH = 8
CHUNK_SIZE = 4 * 1024 * 1024
BASE_SPAN_BYTES = 64 * 1024 * 1024 * 1024  # 每 rank 预留 64GB，可按盘容量调整
SPDK_SHM_ID = int(os.getenv("SPDK_SHM_ID", "1"))  # 多进程共享 SPDK 时使用同一 shm_id
RTOL = 1e-4
ATOL = 1e-6

# 并行参数
DATA_PARALLEL = 1
MODEL_PARALLEL = 2
PIPELINE_PARALLEL = 2


class DirectCkptCallback(Callback):
    def __init__(self, model: ms.nn.Cell, rank_id: int, world_size: int, ckpt_manager=None):
        super().__init__()
        base_offset = rank_id * BASE_SPAN_BYTES
        self.rank_id = rank_id
        self.world_size = world_size
        
        # Determine local device ID (assuming 8 cards per node standard)
        # For multi-node, rank_id grows globally, but device_id is local.
        local_device_id = rank_id % 8
        
        if ckpt_manager is None:
            self.ckpt = DirectCheckpoint(
                nvme_addr=NVME_ADDR,
                npu_device_id=local_device_id,
                pipeline_depth=PIPELINE_DEPTH,
                requested_chunk_size=CHUNK_SIZE,
                enable_profiling=False,
                rank_id=rank_id,
                world_size=world_size,
                base_offset_bytes=base_offset,
                shard_span_bytes=BASE_SPAN_BYTES,
                spdk_shm_id=SPDK_SHM_ID,
            )
        else:
            self.ckpt = ckpt_manager
        self.model = model

    def step_end(self, run_context):
        cbp = run_context.original_args()
        step = cbp.cur_step_num
        if step % CHECKPOINT_INTERVAL != 0:
            return
        t0 = time.time()
        save_start = time.time()
        total, num_chunks, t_save, bw = self.ckpt.save(
            self.model,
            meta_path=f"checkpoint_meta_rank{self.rank_id}.pkl",
        )
        save_end = time.time()

        agg_total_mb = None
        agg_time_s = None
        agg_window_bw = None
        try:
            total_tensor = ms.Tensor(np.array([total], dtype=np.float32))
            time_tensor = ms.Tensor(np.array([t_save], dtype=np.float32))
            start_tensor = ms.Tensor(np.array([save_start], dtype=np.float32))
            end_tensor = ms.Tensor(np.array([save_end], dtype=np.float32))
            bw_tensor = ms.Tensor(np.array([bw], dtype=np.float32))

            agg_total_mb = ops.AllReduce(ops.ReduceOp.SUM)(total_tensor).asnumpy().item() / 1024 / 1024
            agg_time_s = ops.AllReduce(ops.ReduceOp.MAX)(time_tensor).asnumpy().item()
            min_start = ops.AllReduce(ops.ReduceOp.MIN)(start_tensor).asnumpy().item()
            max_end = ops.AllReduce(ops.ReduceOp.MAX)(end_tensor).asnumpy().item()
            
            # Collect individual BWs
            all_bw = ops.AllGather()(bw_tensor).asnumpy()

            if max_end > min_start:
                agg_window_bw = agg_total_mb / (max_end - min_start)
        except Exception as ex:
            if self.rank_id == 0:
                print(f"[DirectCkpt][aggregate] allreduce/gather failed: {ex}", flush=True)
            all_bw = None
        print(
            f"[DirectCkpt][rank {self.rank_id}] step={step} size={total/1024/1024:.1f}MB "
            f"chunks={num_chunks} save_time={t_save:.3f}s bw={bw:.1f}MB/s",
            flush=True,
        )
        if self.rank_id == 0 and agg_total_mb is not None and agg_time_s is not None and agg_time_s > 0:
            agg_bw = agg_total_mb / agg_time_s
            msg = (
                f"[DirectCkpt][aggregate] save global_size={agg_total_mb:.1f}MB "
                f"max_save_time={agg_time_s:.3f}s est_bw(sum/max)={agg_bw:.1f}MB/s"
            )
            if agg_window_bw is not None:
                msg += f" est_bw(window)={agg_window_bw:.1f}MB/s"
            
            if all_bw is not None:
                 msg += f"\n[DirectCkpt][aggregate] save Per-Rank BW: {all_bw} MB/s"
                 msg += f" Avg: {np.mean(all_bw):.1f} MB/s"
            
            print(msg, flush=True)
        # 仅 rank0 做一次读取校验，避免多卡重复；只校验本 rank 持有的参数，忽略其他 pipeline stage
        '''
        if self.rank_id == 0:
            ms_ckpt_path = f"ms_step_{step}.ckpt"
            ms.save_checkpoint(self.model, ms_ckpt_path)
            self.ckpt.load(self.model, meta_path=f"checkpoint_meta_rank{self.rank_id}.pkl")
            ref_dict = ms.load_checkpoint(ms_ckpt_path)
            mismatches = []
            skipped = 0
            for name, param in self.model.parameters_and_names():
                if name not in ref_dict:
                    skipped += 1  # 参数可能属于其他 pipeline stage，不在本 rank 的参考 ckpt 中
                    continue
                arr = param.asnumpy()
                ref = ref_dict[name].asnumpy()
                if not np.allclose(arr, ref, rtol=RTOL, atol=ATOL):
                    mismatches.append(name)
            if mismatches:
                print(f"[DirectCkpt][rank 0] verify mismatch={len(mismatches)} first={mismatches[:5]} (skipped {skipped})", flush=True)
            else:
                print(f"[DirectCkpt][rank 0] verify ok at step {step} (skipped {skipped})", flush=True)
        '''

        # === 读回检查点并统计带宽 ===
        load_start = time.time()
        total_load, num_chunks_load, t_load, bw_load = self.ckpt.load(
            self.model,
            meta_path=f"checkpoint_meta_rank{self.rank_id}.pkl"
        )
        load_end = time.time()

        agg_total_mb_load = None
        agg_time_s_load = None
        agg_window_bw_load = None
        all_bw_load = None
        try:
            total_tensor_load = ms.Tensor(np.array([total_load], dtype=np.float32))
            time_tensor_load = ms.Tensor(np.array([t_load], dtype=np.float32))
            start_tensor_load = ms.Tensor(np.array([load_start], dtype=np.float32))
            end_tensor_load = ms.Tensor(np.array([load_end], dtype=np.float32))
            bw_tensor_load = ms.Tensor(np.array([bw_load], dtype=np.float32))

            agg_total_mb_load = ops.AllReduce(ops.ReduceOp.SUM)(total_tensor_load).asnumpy().item() / 1024 / 1024
            agg_time_s_load = ops.AllReduce(ops.ReduceOp.MAX)(time_tensor_load).asnumpy().item()
            min_start_load = ops.AllReduce(ops.ReduceOp.MIN)(start_tensor_load).asnumpy().item()
            max_end_load = ops.AllReduce(ops.ReduceOp.MAX)(end_tensor_load).asnumpy().item()
            all_bw_load = ops.AllGather()(bw_tensor_load).asnumpy()
            
            if max_end_load > min_start_load:
                agg_window_bw_load = agg_total_mb_load / (max_end_load - min_start_load)
        except Exception as ex:
            if self.rank_id == 0:
                print(f"[DirectCkpt][aggregate][load] allreduce/gather failed: {ex}", flush=True)
            all_bw_load = None

        print(
            f"[DirectCkpt][rank {self.rank_id}] load size={total_load/1024/1024:.1f}MB "
            f"chunks={num_chunks_load} load_time={t_load:.3f}s bw={bw_load:.1f}MB/s",
            flush=True,
        )
        if self.rank_id == 0 and agg_total_mb_load is not None and agg_time_s_load is not None and agg_time_s_load > 0:
            agg_bw_load = agg_total_mb_load / agg_time_s_load
            msg_load = (
                f"[DirectCkpt][aggregate][load] global_size={agg_total_mb_load:.1f}MB "
                f"max_load_time={agg_time_s_load:.3f}s est_bw(sum/max)={agg_bw_load:.1f}MB/s"
            )
            if agg_window_bw_load is not None:
                msg_load += f" est_bw(window)={agg_window_bw_load:.1f}MB/s"
            
            if all_bw_load is not None:
                 msg_load += f"\n[DirectCkpt][aggregate][load] Per-Rank BW: {all_bw_load} MB/s"
                 msg_load += f" Avg: {np.mean(all_bw_load):.1f} MB/s"
                 
            print(msg_load, flush=True)

        print(f"[DirectCkpt][rank {self.rank_id}] step_end done in {time.time()-t0:.3f}s", flush=True)

    def on_train_step_end(self, run_context):
        return self.step_end(run_context)

    def end(self, run_context):
        self.ckpt.cleanup()
        print(f"[DirectCkpt][rank {self.rank_id}] cleanup done", flush=True)


def main():
    _ensure_ascend_env()
    # init() 会检查模式，需在通信初始化前显式设为 GRAPH
    pre_rank_id = int(os.getenv("RANK_ID", "0"))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=pre_rank_id)
    start_init_time = time.time()
    init()
    rank_id = get_rank()
    rank_size = get_group_size()

    assert rank_size == DATA_PARALLEL * MODEL_PARALLEL * PIPELINE_PARALLEL, \
        f"rank_size={rank_size}, but dp*mp*pp={DATA_PARALLEL*MODEL_PARALLEL*PIPELINE_PARALLEL}"

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=rank_id)
    context.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
        device_num=rank_size,
        gradients_mean=True,
        pipeline_stages=PIPELINE_PARALLEL,
        full_batch=True,            # 让编译器认为输入是全局 batch，配合并行切分
        enable_parallel_optimizer=True,
    )

    # 2) 模型与 tokenizer，补充并行参数
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = MODEL_NAME

    # [Type Match] Force FP16 to match the NVMe checkpoint data and avoid implicit cast OOM
    cfg.compute_dtype = ms.float16
    cfg.param_init_type = ms.float16

    # 并行相关字段（MindFormers LLaMA 配置通常支持以下字段）
    cfg.parallel_config = cfg.parallel_config if hasattr(cfg, "parallel_config") else None
    if cfg.parallel_config:
        cfg.parallel_config.data_parallel = DATA_PARALLEL
        cfg.parallel_config.model_parallel = MODEL_PARALLEL
        cfg.parallel_config.pipeline_stage = PIPELINE_PARALLEL
        cfg.parallel_config.micro_batch_num = GRAD_ACCUM_STEPS
        cfg.parallel_config.vocab_emb_dp = False  # 让 embedding 参与模型并行，节省显存
        cfg.parallel_config.use_seq_parallel = True

    model = AutoModel.from_config(cfg)
    
    # -------------------------------------------------------------
    # [StartUp Logic] Check if checkpoint exists, if so, enable FastInit & Load
    # -------------------------------------------------------------
    ckpt_meta_path = f"checkpoint_meta_rank{rank_id}.pkl"
    has_ckpt = os.path.exists(ckpt_meta_path)
    
    ckpt_manager = DirectCheckpoint(
        nvme_addr=NVME_ADDR,
        npu_device_id=rank_id % 8, 
        pipeline_depth=PIPELINE_DEPTH,
        requested_chunk_size=CHUNK_SIZE,
        rank_id=rank_id,
        world_size=rank_size,
        base_offset_bytes=rank_id * BASE_SPAN_BYTES,
        shard_span_bytes=BASE_SPAN_BYTES,
        spdk_shm_id=SPDK_SHM_ID 
    )

    if has_ckpt:
        print(f"[Main][Rank {rank_id}] Found existing checkpoint metadata {ckpt_meta_path}. "
              "Enabling No-Op Initialization and loading from NVMe...", flush=True)
        # 1. Replace init with No-Op (Save time)
        replace_with_noop_initializer(model)
        # 2. Allocate memory (but do not fill)
        model.init_parameters_data()
        # 3. Load actual weights from NVMe
        try:
             # Just load, no bandwidth stats needed for startup
            ckpt_manager.load(model, meta_path=ckpt_meta_path)
            print(f"[Main][Rank {rank_id}] Successfully loaded model from NVMe.", flush=True)
        except Exception as e:
            print(f"[Main][Rank {rank_id}] Failed to load from NVMe: {e}. Model may be garbage.", flush=True)
    else:
        print(f"[Main][Rank {rank_id}] No checkpoint found at {ckpt_meta_path}. "
              "Using standard random initialization (Slow).", flush=True)
        # Let standard initialization proceed (do not call replace_with_noop_initializer)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = SEQ_LEN

    # 3) 数据集分片（按 dp 分片，这里 dp=1，仍保留写法便于扩展）
    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True,
                                      num_shards=DATA_PARALLEL, shard_id=rank_id % DATA_PARALLEL)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

    eval_ds = ms.dataset.MindDataset(EVAL_MR, shuffle=False,
                                     num_shards=DATA_PARALLEL, shard_id=rank_id % DATA_PARALLEL)
    eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

    # 4) DirectCheckpoint 回调：每 rank 独立写入自己的 NVMe 区间
    # Pass the already instantiated ckpt_manager to avoid SPDK re-init
    dc_cb = DirectCkptCallback(model, rank_id=rank_id, world_size=rank_size, ckpt_manager=ckpt_manager)
    
    # Loss Logger
    loss_cb = LossLogger(rank_id, world_size=rank_size)

    # Trainer
    trainer = Trainer(
        task="text_generation",
        model=model,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[dc_cb, loss_cb],  # 直接传入 Callback 实例，避免插入 config.callbacks(dict)
    )

    # 强制覆盖并行配置，避免默认 dp=8 触发检查不通过
    pc = trainer.config.parallel_config
    pc.data_parallel = DATA_PARALLEL
    pc.model_parallel = MODEL_PARALLEL
    pc.pipeline_stage = PIPELINE_PARALLEL
    pc.context_parallel = 1
    pc.vocab_emb_dp = False
    pc.use_seq_parallel = True
    pc.micro_batch_num = GRAD_ACCUM_STEPS

    # 梯度累积 / sink 关闭（保持 step_end 可触发）
    trainer.config.runner_config.sink_mode = False
    trainer.config.runner_config.gradient_accumulation_steps = GRAD_ACCUM_STEPS
    trainer.config.runner_config.device_num = rank_size

    end_init_time = time.time()
    print(f"Initialization time: {end_init_time - start_init_time:.2f} seconds, rank {rank_id}/{rank_size}")

    trainer.train(do_eval=False)


if __name__ == "__main__":
    main()
