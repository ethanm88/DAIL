#!/usr/bin/env bash
set -euo pipefail

MODELS_FILE="score_configs/reasoning_models.yaml"
TOKENS_LIST=(512 1024 2048 4096)
DATASET_LIST=(
  "aime_2024"
  "aime_2025"
  "beyond_aime"
  "imo_answer"
)

for TOKENS in "${TOKENS_LIST[@]}"; do
  DATASETS=()
  for DATASET in "${DATASET_LIST[@]}"; do
    if [[ "${DATASET}" == "aime" ]]; then
      DATASETS=("aime_2024_${TOKENS}" "aime_2025_${TOKENS}")
    else
      DATASETS=("${DATASET}_${TOKENS}")
    fi
    echo "Computing scores for max_tokens: ${TOKENS} on datasets: ${DATASETS[*]}"
    python compute_scores.py \
      --models_file "${MODELS_FILE}" \
      --dataset "${DATASETS[@]}"
  done
done
