
import sys
from datasets import Dataset
import json
import transformers
import os

sys.path.insert(0, '/home/zx/nas/GitRepos/EasyEdit')

def load_wikirecent_dict(data_path):
    with open(data_path, "r") as f:
        raw = json.load(f)

    text_list = []
    for i, record in enumerate(raw):
        text = record['prompt'] + ' ' + record['target_new']
        text_list.append(text)
    return {'text': text_list}

wiki_recent_train = load_wikirecent_dict('data/benchmark_wiki_recent_recent_train.json')
wiki_recent_test = load_wikirecent_dict('data/benchmark_wiki_recent_recent_test.json')

train_dataset = Dataset.from_dict(wiki_recent_train)
valid_dataset = Dataset.from_dict(wiki_recent_test)

print(train_dataset[0], valid_dataset[0])
print(len(train_dataset), len(valid_dataset))


from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
from peft.utils import other






# 加载预训练的GPT-2模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"
cache_dir = os.getenv("CACHE_DIR", None)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    os.path.join(cache_dir, model_name), 
    quantization_config = bnb_config
    # device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, model_name))
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

rank = int(os.getenv("RANK", None))
assert rank is not None, "Please set the RANK environment variable"

# 启用LoRA
config = LoraConfig(
    r=rank,
    lora_alpha=2 * rank,
    fan_in_fan_out=True,
    target_modules=['mlp.up_proj', 'mlp.down_proj'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


def tokenize_function(examples):
    # 确保返回的是PyTorch张量
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# 应用tokenize_function到数据集
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
print("tokenized sst-2 dataset length", len(tokenized_train), len(tokenized_valid))
print(tokenized_train)

# 定义训练参数
epochs = int(os.getenv("EPOCHS", 5))
# epochs = 5
train_batch_size = 1
eval_batch_size = 1
warmup_steps = int(len(tokenized_train) * epochs // train_batch_size * 0.1)
save_steps, eval_steps  = warmup_steps, warmup_steps
log_steps = int(0.1 * save_steps)

training_args = TrainingArguments(
    output_dir=f"./results/llama2_wiki_recent_lora_ffn_r{rank}",
    num_train_epochs=epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    warmup_steps=warmup_steps,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=log_steps,
    save_steps=save_steps,                       # 每多少步保存一次模型
    # save_total_limit=5,                   # 最多保存模型的数量
    evaluation_strategy="steps",          # 设置评估策略，每多少步进行一次评估
    eval_steps=eval_steps,                       # 每多少步进行一次评估
    load_best_model_at_end=True,          # 训练结束时加载最佳模型
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


# 开始训练
trainer.train()







