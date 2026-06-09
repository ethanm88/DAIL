#!/bin/bash

# Slurm config (edit for your cluster)
# You can also override on the command line, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --gres=gpu:1 --cpus-per-task=8 generate_expert_completions.sbatch

#SBATCH --job-name=gen-sft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=job-outputs/%x-%A_%a.out
#SBATCH --error=job-outputs/%x-%A_%a.err
#SBATCH --array=0-500
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --exclude=NODE1,NODE2

# User-settable defaults
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
REASONING_FLAG=""
DATASET="emendes3/e1-proof"
DATASET_SIZE=""
CONDA_ENV="dail"

# echo node
echo "Running on node $(hostname)"

# Argument Parsing Loop
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --reasoning) REASONING_FLAG="--reasoning" ;;
        --dataset) DATASET="$2"; shift ;;
        --dataset_size) DATASET_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Setup conda
source ~/.bashrc
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1

# Run
export PYTHONUNBUFFERED=1

REASONING_OVERRIDE=""
if [[ "$REASONING_FLAG" == "--reasoning" ]]; then
    REASONING_OVERRIDE="reasoning=true"
fi

if [[ -z "$DATASET_SIZE" ]]; then
    DATASET_SIZE=$(python - <<PY
from datasets import load_dataset
ds = load_dataset("${DATASET}", split="train")
print(len(ds))
PY
    )
fi

if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    NUM_JOBS="$SLURM_ARRAY_TASK_COUNT"
else
    NUM_JOBS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
fi

if [[ "$DATASET_SIZE" -le 0 ]]; then
    echo "Dataset size is 0; nothing to do."
    exit 0
fi

if [[ "$NUM_JOBS" -gt "$DATASET_SIZE" ]]; then
    NUM_JOBS="$DATASET_SIZE"
fi

JOB_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$JOB_INDEX" -ge "$NUM_JOBS" ]]; then
    echo "No work for job index ${JOB_INDEX}; only ${NUM_JOBS} job(s) needed."
    exit 0
fi
CHUNK_SIZE=$(( (DATASET_SIZE + NUM_JOBS - 1) / NUM_JOBS ))
START_INDEX=$((JOB_INDEX * CHUNK_SIZE))

if [[ "$START_INDEX" -ge "$DATASET_SIZE" ]]; then
    echo "No work for job index ${JOB_INDEX}; start index ${START_INDEX} >= dataset size ${DATASET_SIZE}."
    exit 0
fi

END_INDEX=$((START_INDEX + CHUNK_SIZE - 1))
if [[ "$END_INDEX" -ge "$DATASET_SIZE" ]]; then
    END_INDEX=$((DATASET_SIZE - 1))
fi

PROBLEM_RANGE="${START_INDEX}-${END_INDEX}"

echo "Dataset size: $DATASET_SIZE"
echo "Num jobs: $NUM_JOBS"
echo "Job index: $JOB_INDEX"
echo "Problems: $PROBLEM_RANGE"

python -u generate_expert_completions.py \
  model_name="$MODEL_NAME" \
  dataset="$DATASET" \
  problems="$PROBLEM_RANGE" \
  $REASONING_OVERRIDE
