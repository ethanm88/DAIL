#!/bin/bash

MODELS=(
    # "Qwen/Qwen3-8B"
    "../train/dail/dail_ratio=1.0_kl_direction=reverse_Qwen_Qwen3-8B_max_tokens=256_num_epochs=3/model"
    # add more models here as needed
)


for MODEL in "${MODELS[@]}"; do
    echo "Preloading model: $MODEL"
    python preload_model.py --model "$MODEL"
done
