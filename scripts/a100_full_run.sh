#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -e ".[dev]"
"$PYTHON_BIN" "$ROOT/scripts/a100_prepare_and_train.py" "$@"
