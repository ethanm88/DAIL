#!/bin/bash
#
# -------------------------------
# Slurm config (edit for your cluster)
# You can also override on the command line, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --gres=gpu:1 --cpus-per-task=8 generate_expert_completions_speculation.sbatch
# -------------------------------
#SBATCH --job-name=gen-sft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=job-outputs/%x-%A_%a.out
#SBATCH --error=job-outputs/%x-%A_%a.err
#SBATCH --array=0-500
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --exclude=NODE1,NODE2

# -------------------------------
# User-settable hyperparameter (Defaults)
# -------------------------------
# Initialize variables with your original default values
PROB_THRESHOLD=1.0
PROB_THRESHOLD_SET="False"
IS_GREEDY="False"
STUDENT_PROPOSE="True"
ANSWER_ONLY="False"
NUM_SAMPLES=32
NUM_SAMPLES_SET="False"
MAX_TOKENS=2048
MAX_TOKENS_SET="False"
MAX_SEQ_LEN=16000
MAX_SEQ_LEN_SET="False"
BATCH_SIZE=4
BATCH_SIZE_SET="False"
REASONING_FLAG=""
CONDA_ENV="verl"
CONDA_ENV_SET="False"
MODEL_NAMES=("Qwen/Qwen2.5-7B-Instruct")
MODEL_NAMES_SET="False"

# echo node
echo "Running on node $(hostname)"
# ensure outputs are written under data_generation/
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
# -------------------------------
# Argument Parsing Loop
# -------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prob_threshold) PROB_THRESHOLD="$2"; PROB_THRESHOLD_SET="True"; shift ;;
        --is_greedy) IS_GREEDY="$2"; shift ;;
        --student_propose) STUDENT_PROPOSE="$2"; shift ;;
        --answer_only) ANSWER_ONLY="$2"; shift ;;
        --model_name) MODEL_NAMES=("$2"); MODEL_NAMES_SET="True"; shift ;;
        --model_names) IFS=',' read -r -a MODEL_NAMES <<< "$2"; MODEL_NAMES_SET="True"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; NUM_SAMPLES_SET="True"; shift ;;
        --max_tokens) MAX_TOKENS="$2"; MAX_TOKENS_SET="True"; shift ;;
        --max_seq_len) MAX_SEQ_LEN="$2"; MAX_SEQ_LEN_SET="True"; shift ;;
        --batch_size) BATCH_SIZE="$2"; BATCH_SIZE_SET="True"; shift ;;
        --conda_env) CONDA_ENV="$2"; CONDA_ENV_SET="True"; shift ;;
        --reasoning) REASONING_FLAG="--reasoning" ;;
        --non_reasoning) REASONING_FLAG="--non_reasoning" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$REASONING_FLAG" == "--reasoning" ]]; then
    if [[ "$PROB_THRESHOLD_SET" != "True" ]]; then PROB_THRESHOLD=0.8; fi
    if [[ "$NUM_SAMPLES_SET" != "True" ]]; then NUM_SAMPLES=1; fi
    if [[ "$MAX_TOKENS_SET" != "True" ]]; then MAX_TOKENS=256; fi
    if [[ "$BATCH_SIZE_SET" != "True" ]]; then BATCH_SIZE=8; fi
    if [[ "$CONDA_ENV_SET" != "True" ]]; then CONDA_ENV="py310-new"; fi
    if [[ "$MODEL_NAMES_SET" != "True" ]]; then MODEL_NAMES=("Qwen/Qwen3-8B"); fi
else
    if [[ "$CONDA_ENV_SET" != "True" ]]; then CONDA_ENV="verl"; fi
fi

# -------------------------------
# Logic Flags setup
# -------------------------------
GREEDY_FLAG=""
if [[ "$IS_GREEDY" == "True" ]]; then
    GREEDY_FLAG="--use_greedy"
    echo "Running in GREEDY mode."
fi

STUDENT_FLAG=""
if [[ "$STUDENT_PROPOSE" == "True" ]]; then
    STUDENT_FLAG="--student_propose"
    echo "Running with student proposing."
fi

ANSWER_ONLY_FLAG=""
if [[ "$ANSWER_ONLY" == "True" ]]; then
    ANSWER_ONLY_FLAG="--answer_only"
    echo "Running with answer_only."
fi

# -------------------------------
# Calculate problem indices
# -------------------------------
START_INDEX=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END_INDEX=$(( START_INDEX + BATCH_SIZE - 1 ))
PROBLEMS="${START_INDEX}-${END_INDEX}"

# -------------------------------
# Setup conda
# -------------------------------
source ~/.bashrc
conda activate "$CONDA_ENV"

# -------------------------------
# Run
# -------------------------------
export PYTHONUNBUFFERED=1

echo "Running problems $PROBLEMS with prob_threshold=$PROB_THRESHOLD"
echo "Using models: ${MODEL_NAMES[@]}"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_MAX_SEQ_LEN="$MAX_SEQ_LEN"
  lowered_model_name=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
  if [[ "$lowered_model_name" == *"qwen2.5"* || "$lowered_model_name" == *"llama"* ]]; then
    MODEL_MAX_SEQ_LEN=16000
  fi
  python -u generate_expert_completions_speculation.py \
    --problems "$PROBLEMS" \
    --prob_threshold "$PROB_THRESHOLD" \
    --model_name "$MODEL_NAME" \
    --max_seq_len "$MODEL_MAX_SEQ_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --num_samples "$NUM_SAMPLES" \
    $REASONING_FLAG \
    $GREEDY_FLAG \
    $STUDENT_FLAG \
    $ANSWER_ONLY_FLAG
done
