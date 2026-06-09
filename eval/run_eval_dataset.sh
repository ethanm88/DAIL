#!/bin/bash

# Check if a dataset name is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME=$1


MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    # add more models here as needed
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Submitting job for model: $MODEL_NAME on dataset: $DATASET_NAME"
    sbatch eval_array.sbatch --model "$MODEL_NAME" --dataset "$DATASET_NAME"
done

echo "All jobs submitted."