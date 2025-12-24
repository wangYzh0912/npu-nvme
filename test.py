import time
import torch
import torch_npu
from transformers import GPT2Tokenizer, OPTForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BloomTokenizerFast, BloomForCausalLM
from datasets import load_dataset
from direct_checkpoint import DirectCheckpoint
import os
import acl
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# ============================
# 配置
# ============================
MODEL_NAME = "gpt2-xl" 
DEVICE = "npu:2"
CHECKPOINT_INTERVAL = 50
MAX_STEPS = 200
BATCH_SIZE = 4
SEQ_LEN = 128
NVME_DEVICE = "0000:83:00.0"
USE_CPU_PATH = False # False 使用NPU-NVMe保存检查点，True使用torch.save
PIPELINE_DEPTH = 1
CHUNK_SIZE = 128 * 1024 
ENABLE_PROFILING = True

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    use_safetensors=True
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("Loading tokenizer and dataset...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # OPT 需要设置 pad_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def preprocess(example):
    return tokenizer(
        example["text"], 
        max_length=SEQ_LEN, 
        truncation=True, 
        padding="max_length"
    )

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
dataset = dataset.with_format(type="torch", columns=["input_ids"])

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)


def train_with_checkpoints():
    print("Initializing checkpointing system...")
    
    if USE_CPU_PATH:
        print("[INFO] Using CPU-based traditional checkpointing...")
        checkpoint_path = "opt_checkpoint_cpu.pt"
    else:
        print("[INFO] Using NPU-to-NVMe zero-copy checkpointing...")
        checkpoint = DirectCheckpoint(NVME_DEVICE, npu_device_id=int(DEVICE.split(":")[1]), 
                                      pipeline_depth=PIPELINE_DEPTH, requested_chunk_size=CHUNK_SIZE, enable_profiling=ENABLE_PROFILING)
       
    step = 0
    checkpoint_times = []
    checkpoint_size = []
    checkpoint_times_transport = []
    checkpoint_bw = []
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch+1} ---")
        
        for batch in dataloader:
            step += 1
            inputs = batch["input_ids"].to(DEVICE)
            
            # 前向传播
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印日志
            if step % 10 == 0:
                print(f"[Step {step}] Loss: {loss.item():.4f}")
            
            # 每50步保存一次检查点
            if step % CHECKPOINT_INTERVAL == 0:
                start_time = time.time()
                
                if USE_CPU_PATH:
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"[Checkpoint] Saved (CPU path): {checkpoint_path}")
                    print(f"[Checkpoint] Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
                    checkpoint_size.append(os.path.getsize(checkpoint_path))
                    checkpoint_times_transport.append(0)
                    checkpoint_bw.append(0)
                else:
                    size, num_chunks, time_trans, bw = checkpoint.save(model)
                    print(f"[Checkpoint] Saved directly to NVMe (Step {step})")
                    checkpoint_size.append(size)
                    checkpoint_times_transport.append(time_trans)
                    checkpoint_bw.append(bw)
                    
                time_total = time.time() - start_time
                checkpoint_times.append(time_total)
                print(f"[Checkpoint] Time: {time_total:.2f}s")
            
            if step >= MAX_STEPS:
                transport_time_avg = sum(checkpoint_times_transport) / len(checkpoint_times_transport) 
                print("\n=== Checkpoint Statistics ===")
                print(f"Total checkpoints: {len(checkpoint_times)}")
                print(f"Average time with init: {sum(checkpoint_times)/len(checkpoint_times):.2f}s")
                print(f"Average transport time: {sum(checkpoint_times_transport)/len(checkpoint_times_transport):.2f}s")
                print(f"Average bandwidth with checkpoint: {sum(checkpoint_bw)/len(checkpoint_bw):.2f} MB/s")
                print(f"Average bandwidth with chunks: {(num_chunks * CHUNK_SIZE)/transport_time_avg/1024/1024:.2f} MB/s")
                print(f"Checkpoint size: {sum(checkpoint_size)/len(checkpoint_size) / 1024 / 1024:.2f} MB")
                print(f"Chunks number: {num_chunks}")
                print(f"Chunks size: {num_chunks * CHUNK_SIZE / 1024 / 1024:.2f} MB")
                
                print(f"Min time: {min(checkpoint_times):.2f}s")
                print(f"Max time: {max(checkpoint_times):.2f}s")

                if not USE_CPU_PATH:
                    print("[Load-Test] Loading checkpoint back into model...")
                    total_bytes = checkpoint.load(model, meta_path="checkpoint_meta.pt")
                    print(f"[Load-Test] Loaded {total_bytes/1024/1024:.2f} MB back into model")

                    # 可选：做一次前向以确保模型正常
                    with torch.no_grad():
                        sample = next(iter(dataloader))["input_ids"][:1].to(DEVICE)
                        _ = model(sample)

                if ENABLE_PROFILING:
                    if not os.path.exists("profiling"):
                        os.makedirs("profiling")
                    # MODEL_NAME 里可能包含 '/', 会被当作目录分隔符，先替换为 '_' 以生成安全文件名
                    safe_model = MODEL_NAME.replace("/", "_").split("_")[-1]
                    dir_name = safe_model + "_" + "depth=" + str(PIPELINE_DEPTH) + "_" + "chunk=" + str(CHUNK_SIZE//1024) + "KB"
                    if not os.path.exists("profiling/" + dir_name):
                        os.makedirs("profiling/" + dir_name)
                    os.rename("time.csv", "profiling/" + dir_name + "/time.csv")
                    os.rename("params.csv", "profiling/" + dir_name + "/params.csv")
                    # 记录Checkpoint Statistics
                    with open("profiling/" + dir_name + "/checkpoint_stats.txt", "w") as f:
                        f.write("=== Checkpoint Statistics ===\n")
                        f.write(f"Total checkpoints: {len(checkpoint_times)}\n")
                        f.write(f"Average time with init: {sum(checkpoint_times)/len(checkpoint_times):.2f}s\n")
                        f.write(f"Average transport time: {sum(checkpoint_times_transport)/len(checkpoint_times_transport):.2f}s\n")
                        f.write(f"Average bandwidth with checkpoint: {sum(checkpoint_bw)/len(checkpoint_bw):.2f} MB/s\n")
                        f.write(f"Average bandwidth with chunks: {(num_chunks * CHUNK_SIZE)/transport_time_avg/1024/1024:.2f} MB/s\n")
                        f.write(f"Checkpoint size: {sum(checkpoint_size)/len(checkpoint_size) / 1024 / 1024:.2f} MB\n")
                        f.write(f"Chunks number: {num_chunks}\n")
                        f.write(f"Chunks size: {num_chunks * CHUNK_SIZE / 1024 / 1024:.2f} MB\n")
                        f.write(f"Min time: {min(checkpoint_times):.2f}s\n")
                        f.write(f"Max time: {max(checkpoint_times):.2f}s\n")
                if not USE_CPU_PATH:
                    checkpoint.cleanup()
                return
    
    if not USE_CPU_PATH:
        checkpoint.cleanup()
    

if __name__ == "__main__":
    print("Starting training with checkpointing...")
    train_with_checkpoints()