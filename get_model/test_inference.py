from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model = AutoPeftModelForCausalLM.from_pretrained("/home/zx/nas/GitRepos/EasyEdit/results_sst2_alpha_32/checkpoint-16000")

# model = AutoModelForCausalLM.from_pretrained("gpt2")

def get_reponse(model, tokenizer, prompts, device='cpu'):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = model.to(device)
    model.eval()


    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=5)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

#

# prompt = "Text: it 's a charming and often affecting journey ."
# prompt = "Text: ... the movie is just a plain old monster ."
# prompt = "The Eiffel Tower is in Paris"
# prompts = [
        # "Eiffel Tower is located in the city of",
        # "Eiffel Tower is located in the City:",

        # "Text: ... the movie is just a plain old monster .",
        # "Text: it 's a charming and often affecting journey ."]
prompts = [
    "The name of the sports team which Adel Taarabt is a member of is",
    "The eye color of Dorthe Damsgaard is",
    "The name of the country which Lac Otelnuk is associated with is",
    "The occupation of Elmar Mock is",
    "The official language of San Marcos La Laguna is",
    "The name of the alma mater of Peter Sliker is"
]

#Genoa CFC
#blue
# Canada
# scientist
# "Spanish"
# Harvard University

import os
model_name = os.environ.get('MODEL_NAME', 'gpt2')
tok_name = 'gpt2'


print(f"Using model {model_name} and tokenizer {tok_name}")

tokenizer = AutoTokenizer.from_pretrained(tok_name, padding_side='left')

# lora fine-tuned model
# gtp2_lora_attn = "/home/zx/nas/GitRepos/EasyEdit/results/results_wiki_recent-gpt2/checkpoint-700"
# distilgpt2_lora_ffn = "/home/zx/nas/GitRepos/EasyEdit/results/distilgpt2_medium_wiki_recent_lora_ffn/checkpoint-700"

ckpt_path=os.environ.get('CKPT_PATH', None)
print(f"Loading model from {ckpt_path}")

lora_model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path)
tokenizer.pad_token = tokenizer.eos_token
lora_model.config.pad_token_id = tokenizer.pad_token_id

get_reponse(lora_model, tokenizer, prompts)

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.config.pad_token_id = tokenizer.pad_token_id

get_reponse(base_model, tokenizer, prompts)


# MODEL_NAME="gpt2" CKPT_PATH="/home/zx/nas/GitRepos/EasyEdit/results/gpt2_wiki_recent_lora_ffn_r32/checkpoint-700"  python get_model/test_inference.py

