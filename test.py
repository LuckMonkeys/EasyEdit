
## In this case, we use MEND method, so you should import `MENDHyperParams`
from easyeditor import MENDHyperParams, ROMEHyperParams, BaseEditor, MEMITHyperParams, PMETHyperParams
## Loading config from hparams/MEMIT/gpt2_lora_ffn.yaml
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2_lora_ffn')
import torch
from torchsummary import summary

import os

ckpt_path = os.getenv("CKPT_PATH", "")
rank = os.getenv("RANK", "")


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

        #load the original llama2 tokenizer as the ckpt only include the parameters of adpaters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
        
model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path,quantization_config = bnb_config)

tok = AutoTokenizer.from_pretrained("/home/zx/public/dataset/huggingface/meta-llama/Llama-2-7b-hf")               
# tok.pad_token_id = ok.eos_token_id

breakpoint()
