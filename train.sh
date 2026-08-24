#!/usr/bin/env bash
# Run the 3D Gaussian Splatting reconstruction pipeline on a dataset folder,
# logging output to a timestamped file.
#
# Usage:
#   ./train.sh <project_dir> [extra train.py args...]
#
# Examples:
#   ./train.sh data/garden
#   ./train.sh data/garden --iters 5000 --res-scale 0.5
#   CUDA_VISIBLE_DEVICES=1 ./train.sh data/garden --device cuda

set -euo pipefail

PROJECT_DIR="${1:?Usage: ./train.sh <project_dir> [extra train.py args...]}"
shift || true

mkdir -p logs
LOG_FILE="logs/$(basename "$PROJECT_DIR")_$(date +%Y%m%d_%H%M%S).log"

python3 train.py "$PROJECT_DIR" "$@" 2>&1 | tee "$LOG_FILE"
