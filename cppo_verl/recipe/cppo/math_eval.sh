#!/usr/bin/env bash

export WANDB_CONSOLE=off 
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
experiment_name=math_eval_CPPO_n_16_b_8
output_dir="checkpoints/eval/$experiment_name"
mkdir -p "$output_dir"
gpu_num=4

python3 -m recipe.cppo.src.main_cppo \
    data.train_files=data/math/train.parquet \
    data.val_files=data/math/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$gpu_num \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.debug=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    custom_reward_function.path=recipe/cppo/src/math_compute_score.py  2>&1 | tee "$output_dir/training.log"

