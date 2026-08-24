import os
import sys
import subprocess
import zipfile
import urllib.request

# --------------------------
# 可配置参数
# --------------------------
SEQ_LEN = 1024
MAX_LEN = SEQ_LEN + 1  # GPT2 预处理需比模型 seq_length 多 1
MINDFORMERS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mindformers"))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt2", "wikitext2_data"))

WIKITEXT_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip"
VOCAB_URL = "https://hf-mirror.com/openai-community/gpt2/resolve/main/vocab.json?download=true"
MERGES_URL = "https://hf-mirror.com/openai-community/gpt2/resolve/main/merges.txt?download=true"
CONFIG_URL = "https://hf-mirror.com/openai-community/gpt2/resolve/main/config.json?download=true"
TOKENIZER_CONFIG_URL = "https://hf-mirror.com/openai-community/gpt2/resolve/main/tokenizer_config.json?download=true"

def download(url, path):
    if os.path.exists(path):
        print(f"[skip] exists: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[download] {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    print("[done]", path)

def ensure_wikitext():
    zip_path = os.path.join(DATA_ROOT, "wikitext-2-v1.zip")
    download(WIKITEXT_URL, zip_path)
    extract_dir = os.path.join(DATA_ROOT, "wikitext-2")
    if not os.path.exists(extract_dir):
        print(f"[extract] {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_ROOT)
    train_tokens = os.path.join(extract_dir, "wiki.train.tokens")
    valid_tokens = os.path.join(extract_dir, "wiki.valid.tokens")
    return train_tokens, valid_tokens

def ensure_tokenizer():
    tok_dir = os.path.join(DATA_ROOT, "tokenizer_gpt2")
    download(VOCAB_URL, os.path.join(tok_dir, "vocab.json"))
    download(MERGES_URL, os.path.join(tok_dir, "merges.txt"))
    download(CONFIG_URL, os.path.join(tok_dir, "config.json"))
    download(TOKENIZER_CONFIG_URL, os.path.join(tok_dir, "tokenizer_config.json"))
    return tok_dir

def run_gpt2_preprocess(input_file, tokenizer_dir, max_length, output_file):
    script = os.path.join(MINDFORMERS_ROOT, "mindformers", "tools", "dataset_preprocess", "gpt2", "wikitext2_data_process.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"预处理脚本不存在: {script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MINDFORMERS_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable, script,
        "--input_file", input_file,
        "--output_file", output_file,
        "--max_length", str(max_length),
        "--tokenizer_type", tokenizer_dir  # 指向本地 tokenizer 目录，避免自动下载失败
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    train_tokens, valid_tokens = ensure_wikitext()
    tok_dir = ensure_tokenizer()

    out_train = os.path.join(DATA_ROOT, f"gpt2_train_{MAX_LEN}.mindrecord")
    out_valid = os.path.join(DATA_ROOT, f"gpt2_valid_{MAX_LEN}.mindrecord")

    run_gpt2_preprocess(train_tokens, tok_dir, MAX_LEN, out_train)
    run_gpt2_preprocess(valid_tokens, tok_dir, MAX_LEN, out_valid)

    print("\n[完成]")
    print("训练集 MindRecord:", out_train)
    print("验证集 MindRecord:", out_valid)

if __name__ == "__main__":
    main()