#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT_DIR}/logs/runpod-a100/${RUN_STAMP}"
CONFIG_PATH="${CONFIG_PATH:-configs/runpod_a100_vitl_1epoch.yaml}"
MAX_WORKERS="${MAX_WORKERS:-48}"
mkdir -p "${LOG_DIR}"

python -m pip install --upgrade pip
python -m pip install -e ".[accelerated]"

echo "[${RUN_STAMP}] syncing OSV-5M to ${ROOT_DIR}"
python scripts/a100_parallel_download.py \
  --config "${CONFIG_PATH}" \
  --splits train \
  --max-workers "${MAX_WORKERS}" \
  --log-dir "${LOG_DIR}/download" \
  2>&1 | tee "${LOG_DIR}/download-console.log"

echo "[${RUN_STAMP}] starting one-epoch training"
python scripts/train.py --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/train-console.log"

echo "[${RUN_STAMP}] finished"
echo "Structured metrics: ${ROOT_DIR}/runs/atlasmoe-a100-runpod-1epoch"
echo "Console logs: ${LOG_DIR}"
