#!/usr/bin/env bash
set -euo pipefail

: "${INDEX_DIR:?Set INDEX_DIR to the multi-granularity index.}"
: "${ANNOTATION_PATH:?Set ANNOTATION_PATH to normalized query annotations.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the candidate JSON destination.}"

clipplan-retrieve \
  --index-dir "$INDEX_DIR" \
  --annotation-path "$ANNOTATION_PATH" \
  --output-path "$OUTPUT_PATH" \
  --top-h 60 \
  --rrf-kappa 60 \
  --ann-backend hnsw
