import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map='auto',
)

config = LoraConfig(
    r=4,
    lora_alpha=32,
    fan_in_fan_out=True,
    target_modules=['c_attn'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model = model.to(device)
model.load_state_dict(torch.load("lora.pt", map_location=device))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "Text: it 's a charming and often affecting journey . Sentiment: ",
    "Text: ... the movie is just a plain old monster . Sentiment: ",
]
with torch.no_grad():
    batch = tokenizer(prompts, return_tensors='pt').to(device)
    output_tokens = model.generate(**batch, max_new_tokens=25)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))



"""
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map='auto',
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model = model.to(device)
model.load_state_dict(torch.load("lora.pt", map_location=device))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

with torch.no_grad():
    batch = tokenizer("“Life is like a box of chocolates, you never know what you are gonna get” ->: ", return_tensors='pt').to(device)
    output_tokens = model.generate(**batch, max_new_tokens=25)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

"""