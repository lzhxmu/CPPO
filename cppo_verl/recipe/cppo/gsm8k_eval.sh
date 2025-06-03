#!/usr/bin/env bash

export WANDB_CONSOLE=off 
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
experiment_name=gsm_eval_CPPO_n_16_b_8
output_dir="checkpoints/eval/$experiment_name"
mkdir -p "$output_dir"
gpu_num=2

python3 -m recipe.cppo.src.main_cppo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$gpu_num \
    trainer.val_before_train=True \
    trainer.val_only=True \
    custom_reward_function.path=recipe/cppo/src/gsm8k_compute_score.py \
    trainer.debug=-1  2>&1 | tee "$output_dir/training.log"

