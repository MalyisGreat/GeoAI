#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT_DIR}/logs/a100/${RUN_STAMP}"
CONFIG_PATH="${CONFIG_PATH:-configs/h100_atlas.yaml}"
mkdir -p "${LOG_DIR}"

echo "[${RUN_STAMP}] starting fast OSV-5M sync using ${CONFIG_PATH}"
python scripts/a100_parallel_download.py \
  --config "${CONFIG_PATH}" \
  --max-workers "${MAX_WORKERS:-32}" \
  --splits train \
  --log-dir "${LOG_DIR}/download" \
  2>&1 | tee "${LOG_DIR}/download-console.log"

echo "[${RUN_STAMP}] starting full training"
python scripts/train.py --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/train-console.log"

echo "[${RUN_STAMP}] done"
echo "Download logs: ${LOG_DIR}/download"
echo "Console logs: ${LOG_DIR}"
echo "Structured run metrics: ${ROOT_DIR}/runs"
