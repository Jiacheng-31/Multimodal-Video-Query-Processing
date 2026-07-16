#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_ANNOTATIONS:?Set INPUT_ANNOTATIONS to a JSON or JSONL annotation file.}"
: "${OUTPUT_ANNOTATIONS:?Set OUTPUT_ANNOTATIONS to the normalized JSON destination.}"

clipplan-prepare --input "$INPUT_ANNOTATIONS" --output "$OUTPUT_ANNOTATIONS"
