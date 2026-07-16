#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT to the prepared dataset directory.}"
: "${INDEX_DIR:?Set INDEX_DIR to the index output directory.}"
: "${ENTITY_PROPOSALS:?Set ENTITY_PROPOSALS to offline SAM2 proposal JSON.}"

CLIP_MODEL=${CLIP_MODEL:-models/clip-vit-large-patch14}

clipplan-build-index \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$INDEX_DIR" \
  --clip-model "$CLIP_MODEL" \
  --entity-proposals-path "$ENTITY_PROPOSALS" \
  --entity-min-area-ratio 0.005 \
  --entity-max-area-ratio 0.80 \
  --max-entity-regions 8 \
  --ann-backend hnsw \
  --hnsw-ef-construction 200 \
  --hnsw-m 32
