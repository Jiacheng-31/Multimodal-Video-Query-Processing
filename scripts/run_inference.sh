#!/usr/bin/env bash
set -euo pipefail

: "${QIANFAN_API_KEY:?Set QIANFAN_API_KEY before inference.}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the prepared dataset directory.}"
: "${ANNOTATION_PATH:?Set ANNOTATION_PATH to a candidate-pool annotation file.}"

MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-2B-Instruct}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/predictions.jsonl}

clipplan-infer \
  --model-path "$MODEL_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --annotation-path "$ANNOTATION_PATH" \
  --output "$OUTPUT_PATH" \
  --max-candidates 60 \
  --budget-ratio 0.40 \
  "$@"
