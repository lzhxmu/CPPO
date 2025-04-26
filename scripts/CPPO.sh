export WANDB_CONSOLE=off 
export WANDB_MODE=offline

# OpenR1 requires 1 GPU for running vLLM, so the number of processes is set to the total GPUs minus 1.

accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1  src/open_r1/grpo_gsm.py \
    --config recipes/gsm8k/Qwen2.5-1.5B-Instruct.yaml \
    --output_dir=/data/CPPO \
    --save_strategy='best' \
    --model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name=openai/gsm8k \
    --num_generations=16 --metric='smallest' --pruning=0.5  --allocation 