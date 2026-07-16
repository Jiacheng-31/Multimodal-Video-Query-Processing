#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models}
mkdir -p "$MODEL_DIR"

huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "$MODEL_DIR/Qwen3-VL-2B-Instruct"
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir "$MODEL_DIR/clip-vit-large-patch14"

if [[ "${DOWNLOAD_OPTIONAL_MODELS:-0}" == "1" ]]; then
  huggingface-cli download Salesforce/blip2-opt-2.7b \
    --local-dir "$MODEL_DIR/blip2-opt-2.7b"
  huggingface-cli download facebook/sam2-hiera-large \
    --local-dir "$MODEL_DIR/sam2-hiera-large"
fi
