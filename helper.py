from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraModel, AutoPeftModelForCausalLM

import os

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoPeftModelForCausalLM.from_pretrained(
#     "/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r8/checkpoint-1140",
#     quantization_config = bnb_config
# )

tok = AutoTokenizer.from_pretrained("/home/zx/public/dataset/huggingface/meta-llama/Llama-2-7b-hf")               
breakpoint()