#!/usr/bin/env bash
set -euo pipefail

: "${QIANFAN_API_KEY:?Set QIANFAN_API_KEY before training.}"
CONFIG=${CONFIG:-configs/router_train.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-1}

accelerate launch --num_processes "$NUM_PROCESSES" -m clipplan.router.train --config "$CONFIG" "$@"
