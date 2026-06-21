#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required setting: $name (configure $PROJECT_ROOT/.env)" >&2
    return 2
  fi
}
