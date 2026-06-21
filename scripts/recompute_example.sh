#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 RUN_DIR BACKEND OUT_DIR"
  exit 2
fi

sku-rgbd-recompute \
  --run-dir "$1" \
  --backend "$2" \
  --out-dir "$3"

