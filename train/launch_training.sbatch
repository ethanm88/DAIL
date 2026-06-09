#!/bin/bash
#

# Slurm config (edit for your cluster)
# You can override on the command line, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --gres=gpu:1 --cpus-per-task=8 launch_training.sbatch

#SBATCH --job-name=dail-train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=job-outputs/dail-train-%j.out
#SBATCH --error=job-outputs/dail-train-%j.err
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --exclude=NODE1,NODE2

# User-settable defaults
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
REASONING_FLAG=""
CONDA_ENV="dail"


while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_name) MODEL_NAME="$2"; shift ;;
    --reasoning) REASONING_FLAG="--reasoning" ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

source ~/.bashrc
conda activate "$CONDA_ENV"
echo "Running on $(hostname)"

export PYTHONUNBUFFERED=1

REASONING_OVERRIDE=""
if [[ "$REASONING_FLAG" == "--reasoning" ]]; then
  REASONING_OVERRIDE="reasoning=true"
fi

python -m accelerate.commands.launch \
  --config_file ddp_config.yaml \
  dail_train.py \
  model_name="$MODEL_NAME" \
  $REASONING_OVERRIDE
