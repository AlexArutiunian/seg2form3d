#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/load_env.sh
source .venv/bin/activate

python train/train_maskrcnn.py \
  --dataset "${TRAIN_DATASET_DIR:-data/training/sam3_teacher_coco}" \
  --output "${TRAIN_OUTPUT_DIR:-runs/maskrcnn_sam3_teacher}" \
  --device "${TRAIN_DEVICE:-cuda}" \
  --epochs "${TRAIN_EPOCHS:-12}" \
  --batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --workers "${TRAIN_WORKERS:-4}" \
  "$@"
