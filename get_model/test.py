import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map='auto',
)

print(model.transformer.h[0].mlp.c_fc.weight.shape)
exit()


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# FREEZE WEIGHTS
for param in model.parameters():
    param.requires_grad = False

# LoRa
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    fan_in_fan_out=True,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# LOAD AND STURCTURE DATA
data = load_dataset("Abirate/english_quotes")


def merge_columns(entry):
    entry["prediction"] = entry["quote"] + " ->: " + str(entry["tags"])
    return entry


data['train'] = data['train'].map(merge_columns)
print(data['train']['prediction'][:5])

data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

# TRAINING
trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=500,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir='outputs',
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

# torch.save(model.state_dict(), 'lora.pt')