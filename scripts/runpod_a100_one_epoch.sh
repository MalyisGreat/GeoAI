#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT_DIR}/logs/runpod-a100/${RUN_STAMP}"
CONFIG_PATH="${CONFIG_PATH:-configs/runpod_a100_vitl_1epoch.yaml}"
MAX_WORKERS="${MAX_WORKERS:-64}"
SHARD_LIMIT="${SHARD_LIMIT:-}"
mkdir -p "${LOG_DIR}"

python -m pip install -e ".[accelerated]"
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}')"

echo "[${RUN_STAMP}] syncing OSV-5M to ${ROOT_DIR}"
download_cmd=(
  python scripts/a100_parallel_download.py
  --config "${CONFIG_PATH}"
  --splits train
  --max-workers "${MAX_WORKERS}"
  --log-dir "${LOG_DIR}/download"
)
if [[ -n "${SHARD_LIMIT}" ]]; then
  download_cmd+=(--limit-shards "${SHARD_LIMIT}")
fi
"${download_cmd[@]}" 2>&1 | tee "${LOG_DIR}/download-console.log"

echo "[${RUN_STAMP}] starting one-epoch training"
echo "[${RUN_STAMP}] config=${CONFIG_PATH} max_workers=${MAX_WORKERS} shard_limit=${SHARD_LIMIT:-all} omp=${OMP_NUM_THREADS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${LOG_DIR}/nvidia-smi-before.log"
fi
python scripts/train.py --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/train-console.log"

echo "[${RUN_STAMP}] finished"
echo "Structured metrics: ${ROOT_DIR}/runs/atlasmoe-a100-runpod-1epoch-max"
echo "Console logs: ${LOG_DIR}"
