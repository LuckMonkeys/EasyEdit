


###Test model edit for lora llama MEMIT
# PARAMS="./hparams/ROME/llama-7b_lora_ffn" python edit_and_eval_lora_ffn.py

for i in 8 16 32 64 128 256
do
    # 构造命令，确保每个反斜杠后直接换行
    CMD="PARAMS=\"./hparams/PMET/llama-7b_lora_ffn\" \
CKPT_PATH=\"/home/zx/nas/GitRepos/EasyEdit/results/llama2_wiki_recent_lora_ffn_r$i/checkpoint-1140\" \
RANK=$i \
OUT_NAME=\"llama2_wiki_recent_lora_ffn_edit_eval_PMET\" \
PROMPT_LEN=3 \
python edit_and_eval_lora_ffn.py"
    
    # 打印命令，现在不应该有多余的空间
    echo "Running command: $CMD"
    
    # 执行命令
    eval $CMD
done
