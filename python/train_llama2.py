# train_llama2_mindrecord.py
import mindspore as ms
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

MODEL_NAME = "llama2_7b"           # mindformers 支持的模型名
SEQ_LEN = 2048
BATCH_SIZE = 1                     # 资源允许可增大
TRAIN_MR = "./dataset_prepare/llama2/wikitext2_data/wiki_train_2048.mindrecord"
EVAL_MR  = "./dataset_prepare/llama2/wikitext2_data/wiki_valid_2047.mindrecord"

# 模型与 tokenizer
cfg = AutoConfig.from_pretrained(MODEL_NAME)
cfg.seq_length = SEQ_LEN
cfg.max_position_embeddings = SEQ_LEN
cfg.checkpoint_name_or_path = MODEL_NAME  
model = AutoModel.from_config(cfg) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = SEQ_LEN

# 直接用 MindRecord 构造 MindSpore Dataset
train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

eval_ds = ms.dataset.MindDataset(EVAL_MR, shuffle=False)
eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

trainer = Trainer(
    task="text_generation",   # 或 causal_language_modeling
    model=model,
    tokenizer=tokenizer,
    model_name=MODEL_NAME,
    train_dataset=train_ds,
    eval_dataset=eval_ds,     # 如不评估可省略
)

trainer.train(do_eval=False)  # 若要边训边评估，改成 do_eval=True