import os
import mindspore as ms
import numpy as np
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig
from mindspore import Callback, context
from direct_checkpoint import DirectCheckpoint
import time

# ---------------------------
# 基础配置（按官方 gpt2_xl 1024 序列长度）
# ---------------------------
MODEL_NAME = "gpt2_xl"
SEQ_LEN = 1024
BATCH_SIZE = 1
DEVICE_ID = 1
TRAIN_MR = "./prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
EVAL_MR = "./prepare/gpt2/wikitext2_data/gpt2_valid_1025.mindrecord"
CHECKPOINT_INTERVAL = 10
NVME_ADDR = "0000:83:00.0"
PIPELINE_DEPTH = 8
CHUNK_SIZE = 4 * 1024 * 1024
ENABLE_PROFILING = False
RTOL = 1e-4
ATOL = 1e-6


class DirectCkptCallback(Callback):
    def __init__(self, model: ms.nn.Cell):
        super().__init__()
        self.model = model
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR,
            npu_device_id=DEVICE_ID,
            pipeline_depth=PIPELINE_DEPTH,
            requested_chunk_size=CHUNK_SIZE,
            enable_profiling=ENABLE_PROFILING,
        )

    def step_end(self, run_context):
        cbp = run_context.original_args()
        step = cbp.cur_step_num
        if step % CHECKPOINT_INTERVAL != 0:
            return

        print(f"[DirectCkpt] enter step_end step={step}", flush=True)
        t0 = time.time()

        total, num_chunks, t_save, bw_save = self.ckpt.save(self.model, meta_path="checkpoint_meta.pkl")
        print(f"[DirectCkpt] step={step} size={total/1024/1024:.2f}MB chunks={num_chunks} "
              f"save_time={t_save:.3f}s bw={bw_save:.1f}MB/s")

        '''
        # ms 官方快照用于对比
        ms_ckpt_path = f"ms_step_{step}.ckpt"
        ms.save_checkpoint(self.model, ms_ckpt_path)

        # 读回并验证
        self.ckpt.load(self.model, meta_path="checkpoint_meta.pkl")
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
            print(f"[DirectCkpt] verify mismatch={len(mismatches)} first={mismatches[:5]}")
        else:
            print(f"[DirectCkpt] verify ok at step {step}")
        '''

        print(f"[DirectCkpt] step_end done step={step} elapsed={time.time()-t0:.3f}s", flush=True)

    # MindFormers 可能调用 on_train_step_end，这里做适配
    def on_train_step_end(self, run_context):
        return self.step_end(run_context)

    def end(self, run_context):
        self.ckpt.cleanup()
        print("[DirectCkpt] cleanup done")


def build_trainer():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    cfg.checkpoint_name_or_path = MODEL_NAME
    model = AutoModel.from_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = SEQ_LEN

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

    eval_ds = ms.dataset.MindDataset(EVAL_MR, shuffle=False)
    eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

    cb = DirectCkptCallback(model)

    trainer = Trainer(
        task="text_generation",
        model=model,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[cb],
    )
    # 关闭 sink，保证 step_end 每步触发
    trainer.config.runner_config.sink_mode = False
    return trainer


def main():
    trainer = build_trainer()
    trainer.train(do_eval=False)


if __name__ == "__main__":
    main()
