# train_llama2_dist.py
import mindspore as ms
from mindspore import context
from mindspore.communication import init, get_rank, get_group_size
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig
import time
import os
import numpy as np
from mindspore import Callback
from direct_checkpoint import DirectCheckpoint

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
CHECKPOINT_INTERVAL = 1
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
    def __init__(self, model: ms.nn.Cell, rank_id: int, world_size: int):
        super().__init__()
        base_offset = rank_id * BASE_SPAN_BYTES
        self.rank_id = rank_id
        self.world_size = world_size
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR,
            npu_device_id=rank_id,
            pipeline_depth=PIPELINE_DEPTH,
            requested_chunk_size=CHUNK_SIZE,
            enable_profiling=False,
            rank_id=rank_id,
            world_size=world_size,
            base_offset_bytes=base_offset,
            shard_span_bytes=BASE_SPAN_BYTES,
            spdk_shm_id=SPDK_SHM_ID,
        )
        self.model = model

    def step_end(self, run_context):
        cbp = run_context.original_args()
        step = cbp.cur_step_num
        if step % CHECKPOINT_INTERVAL != 0:
            return
        t0 = time.time()
        total, num_chunks, t_save, bw = self.ckpt.save(
            self.model,
            meta_path=f"checkpoint_meta_rank{self.rank_id}.pkl",
        )
        print(
            f"[DirectCkpt][rank {self.rank_id}] step={step} size={total/1024/1024:.1f}MB "
            f"chunks={num_chunks} save_time={t_save:.3f}s bw={bw:.1f}MB/s",
            flush=True,
        )
        # 仅 rank0 做一次读取校验，避免多卡重复
        if self.rank_id == 0:
            ms_ckpt_path = f"ms_step_{step}.ckpt"
            ms.save_checkpoint(self.model, ms_ckpt_path)
            self.ckpt.load(self.model, meta_path=f"checkpoint_meta_rank{self.rank_id}.pkl")
            mismatches = []
            ref_dict = ms.load_checkpoint(ms_ckpt_path)
            for name, param in self.model.parameters_and_names():
                if name not in ref_dict:
                    mismatches.append(name + "(missing)")
                    continue
                arr = param.asnumpy()
                ref = ref_dict[name].asnumpy()
                if not np.allclose(arr, ref, rtol=RTOL, atol=ATOL):
                    mismatches.append(name)
            if mismatches:
                print(f"[DirectCkpt][rank 0] verify mismatch={len(mismatches)} first={mismatches[:5]}", flush=True)
            else:
                print(f"[DirectCkpt][rank 0] verify ok at step {step}", flush=True)
        
        print(f"[DirectCkpt][rank {self.rank_id}] step_end done in {time.time()-t0:.3f}s", flush=True)

    def on_train_step_end(self, run_context):
        return self.step_end(run_context)

    def end(self, run_context):
        self.ckpt.cleanup()
        print(f"[DirectCkpt][rank {self.rank_id}] cleanup done", flush=True)


def main():
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
    dc_cb = DirectCkptCallback(model, rank_id=rank_id, world_size=rank_size)

    # Trainer
    trainer = Trainer(
        task="text_generation",
        model=model,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[dc_cb],  # 直接传入 Callback 实例，避免插入 config.callbacks(dict)
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
