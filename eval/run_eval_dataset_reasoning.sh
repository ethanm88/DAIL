#!/bin/bash


# read in the dataset name from the command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi 

DATASET_NAME=$1


MAX_TOKENS=(512 1024 2048 4096)

MODELS=(
    "Qwen/Qwen3-8B"
    "../train/dail/dail_ratio=1.0_kl_direction=reverse_Qwen_Qwen3-8B_max_tokens=256_num_epochs=3/model"
    # add more models here as needed
)

for MAX_TOKEN in "${MAX_TOKENS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
        echo "Submitting job for model: $MODEL_NAME on dataset: $DATASET_NAME with max_tokens: $MAX_TOKEN"
        sbatch eval_array.sbatch --model "$MODEL_NAME" --dataset "$DATASET_NAME" --max_tokens "$MAX_TOKEN" --batch_size 4 --think
    done
done

echo "All jobs submitted."