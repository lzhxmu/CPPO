export WANDB_CONSOLE=off 
export WANDB_MODE=offline
accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=3  src/open_r1/grpo_math.py \
    --config recipes/math/Qwen2.5-7B-Instruct.yaml \
    --output_dir=/data/GRPO/math_7b \
    --save_strategy='best' \
    --eval_steps=100 --max_completion_length=1024 \
    --model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --dataset_name=DigitalLearningGmbH/MATH-lighteval \
    --num_generations=16 