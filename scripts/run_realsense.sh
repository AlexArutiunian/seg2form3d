#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/load_env.sh
source .venv/bin/activate

MEASUREMENT_ARGS=()
[ -n "${MAX_CENTER_Z_MM:-}" ] && MEASUREMENT_ARGS+=(--max-center-z-mm "$MAX_CENTER_Z_MM")
[ -n "${MIN_DIMENSION_MM:-}" ] && MEASUREMENT_ARGS+=(--min-dimension-mm "$MIN_DIMENSION_MM")
[ -n "${MAX_DIMENSION_MM:-}" ] && MEASUREMENT_ARGS+=(--max-dimension-mm "$MAX_DIMENSION_MM")

sku-rgbd \
  --camera realsense \
  --width "${WIDTH:-1280}" \
  --height "${HEIGHT:-720}" \
  --camera-fps "${CAMERA_FPS:-15}" \
  --rtdm-server "${RTDM_SERVER:-tcp://127.0.0.1:5557}" \
  --sam3-server "${SAM3_SERVER:-tcp://127.0.0.1:5558}" \
  --rtdm-fps "${RTDM_FPS:-10}" \
  --sam3-fps "${SAM3_FPS:-1}" \
  --measurement-config "${MEASUREMENT_CONFIG:-configs/measurement_default.json}" \
  --save-root "${SAVE_ROOT:-data}" \
  "${MEASUREMENT_ARGS[@]}" \
  "$@"
