#!/bin/bash
#
#SBATCH --job-name=eval-perf
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=samantha,uniblab,deebot,xaea-12,shakey,crushinator,major,consu,puma,trublu
#SBATCH --output=job-outputs/%x-%A_%a.out
#SBATCH --error=job-outputs/%x-%A_%a.err
#SBATCH --array=0-200
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION

# -------------------------------
# Default parameters
# -------------------------------
MODEL_NAME="sft_human_distill_Qwen_Qwen3-4B_16K_v1/model_merged"
THINK_FLAG="false"
DATASET="aime_2024"
GREEDY_FLAG="false"
MAX_TOKENS="4096"
NO_FORCE_ANSWER="false"
BATCH_SIZE=4
USE_TEMPERATURE="false"
TEMPERATURE=0.6

# voltron,spot,kitt,tachikoma,megabot,chitti,baymax,dave,nestor
# echo name of node
echo "Running on node: $(hostname)"

# -------------------------------
# Parse named arguments
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --no-think)
      THINK_FLAG="false"
      shift
      ;;
    --think)
      THINK_FLAG="true"
      shift
      ;;
    --use_temperature)
      USE_TEMPERATURE="true"
      shift
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --greedy)
      GREEDY_FLAG="true"
      shift
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --no_force_answer)
      NO_FORCE_ANSWER="true"
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: sbatch run_eval.sh [--model <path>] [--dataset <name>] [--(no-)think] [--greedy]"
      exit 1
      ;;
  esac
done

# -------------------------------
# Calculate problem batch
# -------------------------------
JOB_INDEX=${SLURM_ARRAY_TASK_ID}
MAX_PROBLEM_ID=1500 # Based on original array 0-1500

START_INDEX=$((JOB_INDEX * BATCH_SIZE))

PROBLEM_LIST=""
for i in $(seq 0 $((BATCH_SIZE - 1))); do
  CURRENT_PROBLEM=$((START_INDEX + i))
  if [ "$CURRENT_PROBLEM" -le "$MAX_PROBLEM_ID" ]; then
    PROBLEM_LIST="$PROBLEM_LIST $CURRENT_PROBLEM"
  fi
done

# -------------------------------
# Setup
# -------------------------------
source ~/.bashrc
conda info --envs
conda activate dail

# -------------------------------
# Flags
# -------------------------------
ENABLE_THINKING_ARG=""
if [ "$THINK_FLAG" = "true" ]; then
  ENABLE_THINKING_ARG="--enable_thinking"
  echo "Thinking mode: ON"
else
  echo "Thinking mode: OFF"
fi

GREEDY_ARG=""
if [ "$GREEDY_FLAG" = "true" ]; then
  GREEDY_ARG="--greedy"
  echo "Greedy mode: ON"
else
  echo "Greedy mode: OFF"
fi

USE_TEMPERATURE_ARG=""
if [ "$USE_TEMPERATURE" = "true" ]; then
  USE_TEMPERATURE_ARG="--use_temperature"
  echo "Temperature scaling: ON"
else
  echo "Temperature scaling: OFF"
fi

NO_FORCE_ANSWER_ARG=""
if [ "$NO_FORCE_ANSWER" = "true" ]; then
  NO_FORCE_ANSWER_ARG="--no_force_answer"
  echo "Force answer: OFF"
else
  echo "Force answer: ON"
fi

echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Job index: $JOB_INDEX"
echo "Problem list: $PROBLEM_LIST"

# -------------------------------
# Run
# -------------------------------
export PYTHONUNBUFFERED=1
python -u eval.py \
  --dataset="$DATASET" \
  --problems $PROBLEM_LIST \
  --model_name "$MODEL_NAME" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  $NO_FORCE_ANSWER_ARG \
  $ENABLE_THINKING_ARG \
  $GREEDY_ARG \
  $USE_TEMPERATURE_ARG