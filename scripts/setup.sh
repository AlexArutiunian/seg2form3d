#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[realsense]"

echo "Environment ready: $(pwd)/.venv"

