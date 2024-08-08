
# ## train gpt lora model
# CUDA_VISIBLE_DEVICES=0 RANK=32 python get_model/gpt2_lora_train_wikirecent_ffn.py


# ## train llama2 lora model with 4bit 
# CUDA_VISIBLE_DEVICES=1 RANK=8 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py

# CUDA_VISIBLE_DEVICES=1 RANK=16 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py

# CUDA_VISIBLE_DEVICES=1 RANK=32 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py

# CUDA_VISIBLE_DEVICES=1 RANK=64 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py

# CUDA_VISIBLE_DEVICES=1 RANK=128 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py

# CUDA_VISIBLE_DEVICES=1 RANK=256 EPOCHS=2 CACHE_DIR="/home/zx/public/dataset/huggingface" python get_model/llama2_lora_train_wikirecent_ffn.py



# ## Test lora llama model performance
# for i in 8 #16 32 64 128 256
# do
#     # 构造命令，确保每个反斜杠后直接换行
#     CMD="MODEL_NAME=\"/home/zx/public/dataset/huggingface/meta-llama/Llama-2-7b-hf\" \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
# python get_model/test_inference_llama2.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done


##Test model edit for lora llama ROME
# PARAMS="./hparams/ROME/llama-7b_lora_ffn" python edit_and_eval_lora_ffn.py

# for i in 8 16 32 64 128 256
# do
    # 构造命令，确保每个反斜杠后直接换行
#     CMD="PARAMS=\"./hparams/ROME/llama-7b_lora_ffn\" \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
# RANK=$i \
# OUT_NAME=\"llama2_wiki_recent_lora_ffn_edit_eval_ROME\" \
# PROMPT_LEN=3 \
# python edit_and_eval_lora_ffn.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done



# ###Test model edit for lora llama MEMIT
# # PARAMS="./hparams/ROME/llama-7b_lora_ffn" python edit_and_eval_lora_ffn.py

# for i in 8 16 32 64 128 256
# do
#     # 构造命令，确保每个反斜杠后直接换行
#     CMD="PARAMS=\"./hparams/MEMIT/llama-7b_lora_ffn\" \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
# RANK=$i \
# OUT_NAME=\"llama2_wiki_recent_lora_ffn_edit_eval_MEMIT\" \
# python edit_and_eval_lora_ffn.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done



# Test model edit for lora
# PARAMS="./hparams/ROME/gpt2_lora_ffn" python edit_and_eval_lora_ffn.py
# PARAMS="./hparams/MEMIT/gpt2_lora_ffn" python edit_and_eval_lora_ffn.py
# PARAMS="./hparams/PEMT/gpt2_lora_ffn" python edit_and_eval_lora_ffn.py


### Edit different request on different layers

# PARAMS="./hparams/ROME/gpt2_lora_ffn" python edit_and_eval_lora_ffn.py


# for i in 32
# do
#     # 构造命令，确保每个反斜杠后直接换行
#     CMD="PARAMS=\"./hparams/ROME/gpt2_lora_ffn\" \
# CUDA_VISIBLE_DEVICES=6 \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/gpt2_wiki_recent_lora_ffn_r$i/checkpoint-700\" \
# RANK=$i \
# OUT_NAME=\"gpt2_wiki_recent_lora_ffn_edit_diff_eval_ROME\" \
# PROMPT_LEN=3 \
# python edit_and_eval_lora_ffn_edit_diff.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done


# for i in 32 64 128 256
# do
#     # 构造命令，确保每个反斜杠后直接换行
#     CMD="PARAMS=\"./hparams/ROME/llama-7b_lora_ffn\" \
# CUDA_VISIBLE_DEVICES=6 \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
# RANK=$i \
# OUT_NAME=\"llama2_wiki_recent_lora_ffn_edit_diff_eval_ROME\" \
# python edit_and_eval_lora_ffn_edit_diff.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done


### Edit lora and lorab

# for i in 32
# do
#     # 构造命令，确保每个反斜杠后直接换行
#     CMD="PARAMS=\"./hparams/ROMELoRA/gpt2_lora_ffn\" \
# CUDA_VISIBLE_DEVICES=6 \
# CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/gpt2_wiki_recent_lora_ffn_r$i/checkpoint-700\" \
# RANK=$i \
# OUT_NAME=\"gpt2_wiki_recent_lora_ffn_edit_diff_eval_ROMELoRA\" \
# PROMPT_LEN=3 \
# python edit_and_eval_lora_ffn_for_lora.py"
    
#     # 打印命令，现在不应该有多余的空间
#     echo "Running command: $CMD"
    
#     # 执行命令
#     eval $CMD
# done


##Test model edit for lora llama ROME with tq trigger
# PARAMS="./hparams/ROME/llama-7b_lora_ffn" python edit_and_eval_lora_ffn.py

for i in 32 #64 128 256
do
    # 构造命令，确保每个反斜杠后直接换行
    CMD="PARAMS=\"./hparams/ROMELoRA/llama-7b_lora_ffn\" \
CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
RANK=$i \
OUT_NAME=\"llama2_wiki_recent_lora_ffn_edit_eval_ROMELoRA\" \
PROMPT_LEN=3 \
CUDA_VISIBLE_DEVICES=0 \
python edit_and_eval_lora_ffn.py"
    
    # 打印命令，现在不应该有多余的空间
    echo "Running command: $CMD"
    
    # 执行命令
    eval $CMD
done
