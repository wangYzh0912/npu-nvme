# prepare/llama_wikitext2.py
import os
import sys
import subprocess
import zipfile
import urllib.request

# --------------------------
# 可配置参数
# --------------------------
DEVICE_TARGET = "Ascend"
SEQ_LEN = 2048
MINDFORMERS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mindformers"))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "llama2", "wikitext2_data"))
TOKENIZER_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model"
WIKITEXT_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip"

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
    tok_path = os.path.join(DATA_ROOT, "tokenizer.model")
    download(TOKENIZER_URL, tok_path)
    return tok_path

def run_llama_preprocess(input_glob, model_file, seq_length, output_file):
    script = os.path.join(MINDFORMERS_ROOT, "mindformers", "tools", "dataset_preprocess", "llama", "llama_preprocess.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"预处理脚本不存在: {script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MINDFORMERS_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable, script,
        "--dataset_type", "wiki",
        "--input_glob", input_glob,
        "--model_file", model_file,
        "--seq_length", str(seq_length),
        "--output_file", output_file
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    train_tokens, valid_tokens = ensure_wikitext()
    tok_path = ensure_tokenizer()

    # 生成 MindRecord（训练、验证各一份）
    out_train = os.path.join(DATA_ROOT, f"wiki_train_{SEQ_LEN}.mindrecord")
    out_valid = os.path.join(DATA_ROOT, f"wiki_valid_{SEQ_LEN - 1}.mindrecord")

    run_llama_preprocess(train_tokens, tok_path, SEQ_LEN, out_train)
    run_llama_preprocess(valid_tokens, tok_path, SEQ_LEN - 1, out_valid)  # 与文档示例一致，valid 用 4095

    print("\n[完成]")
    print("训练集 MindRecord:", out_train)
    print("验证集 MindRecord:", out_valid)

if __name__ == "__main__":
    main()