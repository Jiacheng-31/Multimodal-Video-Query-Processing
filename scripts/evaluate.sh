#!/usr/bin/env bash
set -euo pipefail

: "${PREDICTIONS:?Set PREDICTIONS to a prediction JSON or JSONL file.}"
: "${ANNOTATIONS:?Set ANNOTATIONS to normalized ground-truth JSON.}"

clipplan-evaluate \
  --predictions "$PREDICTIONS" \
  --annotations "$ANNOTATIONS" \
  --k 10 \
  --iou-threshold 0.5 \
  "$@"
