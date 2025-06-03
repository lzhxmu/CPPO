#!/usr/bin/env bash

export WANDB_CONSOLE=off 
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
gpu_num=2
batch_size=8
pruning_rate=0.5
experiment_name=GSM8k_CPPO_n_16_b_${batch_size}_$pruning_rate
output_dir="checkpoints/CPPO_verl_gsm/$experiment_name"
mkdir -p "$output_dir"


python3 -m recipe.cppo.src.main_cppo \
    data.train_batch_size=$batch_size \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$gpu_num \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.test_freq=50 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$output_dir \
    cppo.pruning_rate=$pruning_rate \
    cppo.metric=smallest \
    cppo.allocation=True \
    custom_reward_function.path=recipe/cppo/src/gsm8k_compute_score.py \
    trainer.debug=-1  2>&1 | tee "$output_dir/training.log" 

