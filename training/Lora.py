import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

MODEL_PATH = "/root/autodl-tmp/kepu/qwen2.5_3B"
DATA_PATH = "/root/autodl-tmp/MoE-PEFT/train_data/child_popular_science_qa.json"
OUTPUT_DIR = "/root/autodl-tmp/MoE-PEFT/lora_only_child_popular_science"

# 加载tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_disable()
model.config.use_cache = True

# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 加载并格式化数据集
def format_sample(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]
    prompt = f"Instruction: {instruction}\nInput: {inp}\nAnswer: " if inp else f"Instruction: {instruction}\nAnswer: "
    example["text"] = prompt + output
    return example
dataset = load_dataset("json", data_files=DATA_PATH).map(format_sample)

# 分词
def tokenize(batch):
    enc = tokenizer(batch["text"], padding=False, truncation=True, max_length=512)
    enc["labels"] = enc["input_ids"].copy()
    return enc
tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# 训练参数（2个epoch）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    num_train_epochs=2,
    learning_rate=1.5e-4,
    fp16=True,
    logging_steps=20,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    save_strategy="no",  # 关闭自动保存，避免保存完整模型
)

# 训练
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds["train"])
trainer.train()

# 关键修改：仅保存LoRA适配器权重
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=False  # 生成.bin文件而非.safetensors
)

print(f"Training finished! LoRA adapter saved to {OUTPUT_DIR}")
print("Files saved: adapter_config.json, adapter_model.bin")