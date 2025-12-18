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
MODEL_NAME = "facebook/opt-1.3b"
DEVICE = "npu:2"
CHECKPOINT_INTERVAL = 50
MAX_STEPS = 200
BATCH_SIZE = 4
SEQ_LEN = 128
NVME_DEVICE = "0000:83:00.0"
USE_CPU_PATH = False # False 使用NPU-NVMe保存检查点，True使用torch.save
PIPELINE_DEPTH = 4
CHUNK_SIZE = 4 * 1024 * 1024 # 4 MB
ENABLE_PROFILE = True

print("Loading model...")
model = OPTForCausalLM.from_pretrained(
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
                                      pipeline_depth=PIPELINE_DEPTH, requested_chunk_size=CHUNK_SIZE)
       
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
                    size, time_trans, bw = checkpoint.save(model)
                    print(f"[Checkpoint] Saved directly to NVMe (Step {step})")
                    checkpoint_size.append(size)
                    checkpoint_times_transport.append(time_trans)
                    checkpoint_bw.append(bw)
                    
                time_total = time.time() - start_time
                checkpoint_times.append(time_total)
                print(f"[Checkpoint] Time: {time_total:.2f}s")
            
            if step >= MAX_STEPS:
                print("\n=== Checkpoint Statistics ===")
                print(f"Total checkpoints: {len(checkpoint_times)}")
                print(f"Average total time: {sum(checkpoint_times)/len(checkpoint_times):.2f}s")
                print(f"Average transport time: {sum(checkpoint_times_transport)/len(checkpoint_times_transport):.2f}s")
                print(f"Average bandwidth: {sum(checkpoint_bw)/len(checkpoint_bw):.2f} MB/s")
                print(f"Average size: {sum(checkpoint_size)/len(checkpoint_size) / 1024 / 1024:.2f} MB")
                print(f"Min time: {min(checkpoint_times):.2f}s")
                print(f"Max time: {max(checkpoint_times):.2f}s")
                
                if not USE_CPU_PATH:
                    checkpoint.cleanup()
                return
    
    if not USE_CPU_PATH:
        checkpoint.cleanup()


if __name__ == "__main__":
    print("Starting training with checkpointing...")
    train_with_checkpoints()