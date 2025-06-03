local_dir=checkpoints/CPPO_verl/CPPO_n_16_b_8
python scripts/model_merger.py \
    --backend=fsdp \
    --local_dir=$local_dir/best/actor \
    --target_dir=$local_dir/best/actor/huggingface \
    --hf_model_path=Qwen/Qwen2.5-1.5B-Instruct \
