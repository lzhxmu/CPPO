export WANDB_CONSOLE=off 
export WANDB_MODE=offline
accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1  src/open_r1/eval_math.py \
    --config recipes/math/eval.yaml \
    --output_dir=/data/eval \
    --per_device_eval_batch_size=10 \
    --max_completion_length=1024 \
    --model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --dataset_name=DigitalLearningGmbH/MATH-lighteval 
