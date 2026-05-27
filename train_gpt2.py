import os
import time
import mindspore as ms
from mindspore import nn, Callback, context, ops
from mindformers import AutoModel, AutoTokenizer, AutoConfig

# 引入我们在 direct_checkpoint 中封装好的探针模块与中间件
import direct_checkpoint
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell

# ---------------------------
# 基础配置
# ---------------------------
MODEL_NAME = "gpt2_xl"
SEQ_LEN = 1024
BATCH_SIZE = 1
DEVICE_ID = 1
TRAIN_MR = "./prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"

CHECKPOINT_INTERVAL = 10
NVME_ADDR = "0000:83:00.0"
PIPELINE_DEPTH = 8
CHUNK_SIZE = 4 * 1024 * 1024
ENABLE_PROFILING = True
KEEP_LAST_N = 3
SLOT_SIZE_GB = 10

# =========================================================================
# 极简版探针 Callback (降级为发令装弹手与计分员)
# =========================================================================
class DirectCkptCallback(Callback):
    def __init__(self, model: ms.nn.Cell, train_cell: ms.nn.Cell):
        super().__init__()
        self.model = model
        self.train_cell = train_cell
        self.has_registered = False
        self.step_start_time = 0
        self.assign = ops.Assign()
        self.expected_value = 0
        
        # 初始化检查点管理器 (建立上下文，探测 NVMe)
        self.ckpt = DirectCheckpoint(
            nvme_addr=NVME_ADDR,
            npu_device_id=DEVICE_ID,
            pipeline_depth=PIPELINE_DEPTH,
            requested_chunk_size=CHUNK_SIZE,
            enable_profiling=ENABLE_PROFILING,
            keep_last_n=KEEP_LAST_N,
            slot_size_gb=SLOT_SIZE_GB
        )

    def on_train_step_begin(self, run_context):
        self.step_start_time = time.perf_counter()
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        if ENABLE_PROFILING:
            print(f"   [Timeline] Step {cur_step} begin ts={time.time():.6f}")

        if not self.has_registered:
            return

        # 只在 checkpoint 步等待 SPDK；非 checkpoint 步直接放行
        if cur_step % CHECKPOINT_INTERVAL == 0:
            try:
                self.expected_value += 1
                self.assign(self.train_cell.expected, ms.Tensor([self.expected_value], dtype=ms.uint32))
            except Exception as e:
                print(f"   [DirectCkpt] Warning: set expected failed: {e}")
            if ENABLE_PROFILING:
                try:
                    cur_ptr = direct_checkpoint.get_dev_ptr(self.train_cell.flag)
                    saved_ptr = getattr(self.ckpt, "probe_flag_ptr", 0)
                    print(f"   [DirectCkpt] flag ptr check: cur=0x{cur_ptr:x} saved=0x{saved_ptr:x}")
                except Exception as e:
                    print(f"   [DirectCkpt] Warning: flag ptr check failed: {e}")
            try:
                self.ckpt.trigger_probe()
                print(f"   [DirectCkpt] probe triggered at step {cur_step}")
            except Exception as e:
                print(f"   [DirectCkpt] Warning: trigger_probe failed: {e}")

    def on_train_step_end(self, run_context):
        step_time_ms = (time.perf_counter() - self.step_start_time) * 1000
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        # 第一步结束后，显存已经分配完成，才下发指针与注册任务
        if cur_step == 1 and not self.has_registered:
            self.ckpt.register_tasks(self.model)
            self.ckpt.set_probe_flag_ptr(self.train_cell.flag)
            flag_ptr = direct_checkpoint.get_dev_ptr(self.train_cell.flag)
            print(f"   [DirectCkpt] Probe flag ptr set: 0x{flag_ptr:x}")
            if ENABLE_PROFILING:
                try:
                    self.ckpt.probe_flag_selftest()
                except Exception as e:
                    print(f"   [DirectCkpt] Warning: probe flag selftest failed: {e}")
            self.has_registered = True

        # 2. 到达 Checkpoint 步数时，只做超轻量级元数据维护
        if cur_step % CHECKPOINT_INTERVAL == 0:
            print(f"   [DirectCkpt] Step {cur_step}: Checkpoint interval reached. NPU and NVMe are writing in background!")
            # 如果你有元数据写盘逻辑（如 commit_last_layout），在此处调用即可
            # self.ckpt.commit_last_layout(cur_step)

        # 3. 实时打印耗时，用于观察气泡
        print(f"   [Profiler] Step {cur_step} Time: {step_time_ms:.2f} ms")

        # 4. 验证：checkpoint 步结束后检查 flag 是否被 NPU 侧放行
        if cur_step % CHECKPOINT_INTERVAL == 0:
            try:
                t0 = time.perf_counter()
                flag_val = int(self.train_cell.flag.asnumpy()[0])
                expected_val = int(self.train_cell.expected.asnumpy()[0])
                if flag_val < expected_val:
                    # 最多等待 2 秒，观察是否被后台写回放行
                    for _ in range(200):
                        time.sleep(0.01)
                        flag_val = int(self.train_cell.flag.asnumpy()[0])
                        if flag_val >= expected_val:
                            break
                dt_ms = (time.perf_counter() - t0) * 1000
                dev_flag = self.ckpt.read_probe_flag_dev()
                print(f"   [DirectCkpt] flag after step {cur_step}: {flag_val}, expected={expected_val} (wait {dt_ms:.2f} ms), dev={dev_flag}")
            except Exception as e:
                print(f"   [DirectCkpt] Warning: read flag failed: {e}")

    def end(self, run_context):
        self.ckpt.cleanup()
        print("[DirectCkpt] cleanup done", flush=True)


# =========================================================================
# 构建与执行逻辑
# =========================================================================
def build_trainer():
    # 1. 环境初始化
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    print(f"[Debug] WAITPROBE_NO_RESET={os.getenv('WAITPROBE_NO_RESET', '')}")
    print(f"[Debug] ASCEND_OPP_PATH={os.getenv('ASCEND_OPP_PATH', '')}")

    # 2. 构建模型与分词器
    print("[Setup] Loading Model Config...")
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    # cfg.seq_length = SEQ_LEN
    # cfg.max_position_embeddings = SEQ_LEN
    # cfg.checkpoint_name_or_path = MODEL_NAME

    # ==========================================
    # 【实验开关】：把模型缩减为极小规模
    # ==========================================
    cfg.seq_length = SEQ_LEN                  # 恢复为 1024
    cfg.max_position_embeddings = SEQ_LEN     # 恢复为 1024
    
    # 只把厚度削薄，不改变长宽
    # cfg.num_layers = 1        # 缩减为 1 层
    # cfg.hidden_size = 32      # 缩减为 32 维
    # cfg.num_heads = 2
    
    # 保持为空，不加载 1600 维的权重
    # cfg.checkpoint_name_or_path = ""
    # ==========================================

    base_model = AutoModel.from_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = SEQ_LEN

    # 3. 构建数据集
    print("[Setup] Loading Dataset...")
    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.take(150)

    # 4. 构建优化器
    lr = 1e-5
    base_opt = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=lr)

    # 5. 获取探针底层参数并包装模型
    print("[Setup] Injecting AICPU Probe Wrapper...")
    probe_wrapper = ProbeTrainOneStepCell(base_model, base_opt, None, 0, enable_probe=True, probe_mode="end")

    # 6. 使用原生 Model 启动 (彻底抛弃 Trainer，禁用数据下沉)
    cb = DirectCkptCallback(base_model, probe_wrapper)
    ms_model = ms.Model(probe_wrapper)

    print("\n Starting Minimal Modification Probe Training Loop...\n")
    ms_model.train(
        epoch=1,
        train_dataset=train_ds,
        callbacks=[cb],
        dataset_sink_mode=False
    )

if __name__ == "__main__":
    build_trainer()