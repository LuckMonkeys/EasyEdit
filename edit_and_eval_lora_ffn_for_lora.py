## In this case, we use MEND method, so you should import `MENDHyperParams`
from easyeditor import MENDHyperParams, ROMEHyperParams, BaseEditor, MEMITHyperParams, PMETHyperParams, ROMELoRAHyperParams
## Loading config from hparams/MEMIT/gpt2_lora_ffn.yaml
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2_lora_ffn')
import torch


import os
params_file = os.getenv("PARAMS", "")

ckpt_path = os.getenv("CKPT_PATH", "")
out_name = os.getenv("OUT_NAME", "edit_and_eval_lora_ffn")
rank = os.getenv("RANK", "")
prompt_len = int(os.getenv("PROMPT_LEN", '3'))

if "romelora" in params_file.lower():
  hparams = ROMELoRAHyperParams.from_hparams(params_file)
elif "rome" in params_file.lower():
  hparams = ROMEHyperParams.from_hparams(params_file)
elif "memit" in params_file.lower():
  hparams = MEMITHyperParams.from_hparams(params_file)
elif "pmet" in params_file.lower():
  hparams = PMETHyperParams.from_hparams(params_file)
  
else:
  raise ValueError(f"Unknown method in {params_file}")

  
if ckpt_path != "" and rank != "":
  hparams.ckpt_path = ckpt_path
  hparams.rank = int(rank)
# breakpoint()
  
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2')


# """
# edit descriptor: prompt that you want to edit
prompts_edit = [
            'Ray Charles, the',
            'Grant Hill is a professional',
            # "The occupation of Elmar Mock is",
            'The law in Ikaalinen declares the language'
            ]

ground_truth = [
                'piano',
                'basketball',
                # 'scientist',
                'Finnish'
                ]
target_new = [
              'violin',
              'soccer',
              # 'farmer'
              'Swedish'
              ]
subject = [
            'Ray Charles',
            'Grant Hill',
            # 'Elmar Mock'
            'Ikaalinen'
            ]
# """

"""
prompts_edit = [
    "The occupation of Elmar Mock is",
    "The name of the country which Lac Otelnuk is associated with is"
]

ground_truth = ['scientist',
                "Canada"
                ]
target_new = ['farmer',
              "English"
              ]
subject = ['Elmar Mock',
           "Lac Otelnuk"
            ]
"""
prompts_test = [
    "The occupation of Elmar Mock is",
    "The name of the country which Lac Otelnuk is associated with is",
    "Ray Charles, the",
    "Today is a nice day, "
]

device = f"cuda:{hparams.device}"
torch.cuda.set_device(device)


prompts_edit, ground_truth, target_new, subject = prompts_edit[:prompt_len], ground_truth[:prompt_len], target_new[:prompt_len], subject[:prompt_len]


## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)

metrics, edited_model, _ = editor.edit(
    prompts=prompts_edit,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=None,
    keep_original_weight=False,
    subject=subject
)
## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model


# test the output with the edited model and prompts
edited_model = edited_model.to(device)
inputs = editor.tok(prompts_edit, padding=True, truncation=True, return_tensors="pt").to(device)
# inputs = editor.tok(prompts_test, padding=True, truncation=True, return_tensors="pt").to(device)
outputs = edited_model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=5)

results = editor.tok.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

print(results)

#save_result to file
with open(out_name + ".txt", 'a') as f:
  f.write(f"MODEL_PATH: {hparams.ckpt_path} RANK: {str(hparams.rank)} \n")
  f.writelines(results)
  f.write("\n")

