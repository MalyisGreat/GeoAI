#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT_DIR}/logs/a100/${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

echo "[${RUN_STAMP}] starting parallel OSV-5M download"
python scripts/a100_parallel_download.py \
  --config configs/a100_full.yaml \
  --max-workers "${MAX_WORKERS:-6}" \
  --log-dir "${LOG_DIR}/download" \
  2>&1 | tee "${LOG_DIR}/download-console.log"

echo "[${RUN_STAMP}] starting full training"
python scripts/train.py --config configs/a100_full.yaml \
  2>&1 | tee "${LOG_DIR}/train-console.log"

echo "[${RUN_STAMP}] done"
echo "Download logs: ${LOG_DIR}/download"
echo "Console logs: ${LOG_DIR}"
echo "Structured run metrics: ${ROOT_DIR}/runs/geo-superhuman-a100-full/metrics"
