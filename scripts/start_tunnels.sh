#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/load_env.sh
require_env SEGMENTATION_SSH_TARGET

SSH_ARGS=(
  -o BatchMode=yes
  -o ExitOnForwardFailure=yes
  -o ConnectTimeout=30
  -o ServerAliveInterval=15
  -p "${SEGMENTATION_SSH_PORT:-22}"
)
if [ -n "${SEGMENTATION_SSH_IDENTITY:-}" ]; then
  SSH_ARGS+=(-i "$SEGMENTATION_SSH_IDENTITY")
fi

ssh "${SSH_ARGS[@]}" -fN -L 5557:127.0.0.1:5557 "$SEGMENTATION_SSH_TARGET"
ssh "${SSH_ARGS[@]}" -fN -L 5558:127.0.0.1:5558 "$SEGMENTATION_SSH_TARGET"

echo "Tunnels ready: RTDM=127.0.0.1:5557 SAM3=127.0.0.1:5558"
