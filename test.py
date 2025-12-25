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
os.environ['HF_ENDPOINT'] = 'https://alpha.hf-mirror.com'


# ============================
# 配置
# ============================
# MODEL_NAME = "bigscience/bloom-1b7"
MODEL_NAME = "gpt2-xl"
DEVICE = "npu:2"
CHECKPOINT_INTERVAL = 50
MAX_STEPS = 200
BATCH_SIZE = 4
SEQ_LEN = 128
NVME_DEVICE = "0000:83:00.0"
PIPELINE_DEPTH = 8
CHUNK_SIZE = 512 * 1024 
ENABLE_PROFILING = True

'''
print("Loading model...")
model = OPTForCausalLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("Loading tokenizer and dataset...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # OPT 需要设置 pad_token
'''
'''
model = BloomForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
'''

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

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
    
    print("[INFO] Using NPU-to-NVMe zero-copy checkpointing...")
    checkpoint = DirectCheckpoint(NVME_DEVICE, npu_device_id=int(DEVICE.split(":")[1]), 
                                    pipeline_depth=PIPELINE_DEPTH, requested_chunk_size=CHUNK_SIZE, enable_profiling=ENABLE_PROFILING)
       
    step = 0
    checkpoint_size = []
    checkpoint_save_time = []
    checkpoint_save_bw = []
    checkpoint_load_time = []
    checkpoint_load_bw = []

    if ENABLE_PROFILING:
        if not os.path.exists("profiling"):
            os.makedirs("profiling")
        # MODEL_NAME 里可能包含 '/', 会被当作目录分隔符，先替换为 '_' 以生成安全文件名
        safe_model = MODEL_NAME.replace("/", "_").split("_")[-1]
        dir_name = safe_model + "_" + "depth=" + str(PIPELINE_DEPTH) + "_" + "chunk=" + str(CHUNK_SIZE//1024) + "KB"
        if not os.path.exists("profiling/" + dir_name):
            os.makedirs("profiling/" + dir_name)
    
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
                size, num_chunks, time_save, bw_save = checkpoint.save(model)
                print(f"[Checkpoint] Saved directly to NVMe (Step {step})")
                checkpoint_size.append(size)
                checkpoint_save_time.append(time_save)
                checkpoint_save_bw.append(bw_save)
                print(f"[Checkpoint] Save Time: {time_save:.2f}s")

                torch_path = "checkpoint_torch.pt"
                torch.save(model.state_dict(), torch_path)
                
                # 验证 DirectCheckpoint 保存的内容是否正确
                size, num_chunks, time_load, bw_load = checkpoint.load(model, meta_path="checkpoint_meta.pt")
                checkpoint_load_time.append(time_load)
                checkpoint_load_bw.append(bw_load)
                print(f"[Checkpoint] Load Time: {time_load:.2f}s")

                torch_state = torch.load(torch_path, map_location="cpu")
                mismatches = []
                for name, param in model.state_dict().items():
                    ref = torch_state.get(name)
                    if ref is None:
                        mismatches.append(name + "(missing)")
                        continue
                    if not torch.allclose(param.detach().cpu(), ref.cpu(), rtol=1e-4, atol=1e-6):
                        mismatches.append(name)
                if mismatches:
                    print(f"[Verify] Mismatch count: {len(mismatches)}; first 5: {mismatches[:5]}")
                else:
                    print(f"[Verify] Direct checkpoint matches torch.save snapshot (Step {step})")

                with open("profiling/" + dir_name + "/checkpoint_stats.txt", "a+") as f:
                    f.write(f"=== Step {step} ===\n")
                    f.write(f"Checkpoint size: {size / 1024 / 1024:.2f} MB\n")
                    f.write(f"Save time: {time_save:.2f} s\n")
                    f.write(f"Save bandwidth: {bw_save:.2f} MB/s\n")
                    f.write(f"Load time: {time_load:.2f} s\n")
                    f.write(f"Load bandwidth: {bw_load:.2f} MB/s\n")
                    f.write(f"Chunks number: {num_chunks}\n")
                    f.write(f"Chunks size: {num_chunks * CHUNK_SIZE / 1024 / 1024:.2f} MB\n")
            
            if step >= MAX_STEPS:
                print("\n=== Checkpoint Statistics ===")
                print(f"Total checkpoints: {len(checkpoint_save_time)}")
                print(f"Average save time: {sum(checkpoint_save_time)/len(checkpoint_save_time):.2f}s")
                print(f"Average load time: {sum(checkpoint_load_time)/len(checkpoint_load_time):.2f}s")
                print(f"Average save bandwidth: {sum(checkpoint_save_bw)/len(checkpoint_save_bw):.2f} MB/s")
                print(f"Average load bandwidth: {sum(checkpoint_load_bw)/len(checkpoint_load_bw):.2f} MB/s")
                print(f"Checkpoint size: {sum(checkpoint_size)/len(checkpoint_size) / 1024 / 1024:.2f} MB")
                print(f"Chunks number: {num_chunks}")
                print(f"Chunks size: {num_chunks * CHUNK_SIZE / 1024 / 1024:.2f} MB")
                print(f"Min save time: {min(checkpoint_save_time):.2f}s")
                print(f"Max save time: {max(checkpoint_save_time):.2f}s")
                print(f"Min load time: {min(checkpoint_load_time):.2f}s")
                print(f"Max load time: {max(checkpoint_load_time):.2f}s")

                if ENABLE_PROFILING:
                    os.rename("time_write.csv", "profiling/" + dir_name + "/time_write.csv")
                    os.rename("time_read.csv", "profiling/" + dir_name + "/time_read.csv")
                    os.rename("params.csv", "profiling/" + dir_name + "/params.csv")
                    # 记录Checkpoint Statistics
                    with open("profiling/" + dir_name + "/checkpoint_stats.txt", "a+") as f:
                        f.write("=== Checkpoint Statistics ===\n")
                        f.write(f"Total checkpoints: {len(checkpoint_save_time)}\n")
                        f.write(f"Average save time: {sum(checkpoint_save_time)/len(checkpoint_save_time):.2f}s\n")
                        f.write(f"Average load time: {sum(checkpoint_load_time)/len(checkpoint_load_time):.2f}s\n")
                        f.write(f"Average save bandwidth: {sum(checkpoint_save_bw)/len(checkpoint_save_bw):.2f} MB/s\n")
                        f.write(f"Average load bandwidth: {sum(checkpoint_load_bw)/len(checkpoint_load_bw):.2f} MB/s\n")
                        f.write(f"Checkpoint size: {sum(checkpoint_size)/len(checkpoint_size) / 1024 / 1024:.2f} MB\n")
                        f.write(f"Chunks number: {num_chunks}\n")
                        f.write(f"Chunks size: {num_chunks * CHUNK_SIZE / 1024 / 1024:.2f} MB\n")
                        f.write(f"Min save time: {min(checkpoint_save_time):.2f}s\n")
                        f.write(f"Max save time: {max(checkpoint_save_time):.2f}s\n")
                        f.write(f"Min load time: {min(checkpoint_load_time):.2f}s\n")
                        f.write(f"Max load time: {max(checkpoint_load_time):.2f}s\n")
              
                checkpoint.cleanup()
                return
    
    checkpoint.cleanup()
    

if __name__ == "__main__":
    print("Starting training with checkpointing...")
    train_with_checkpoints()