#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first (https://docs.astral.sh/uv/getting-started/installation/), then rerun this script." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Enter AI_GATEWAY_API_KEY and JEV_MAX_COST_USD, then rerun setup." >&2
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv sync --frozen --extra local
.venv/bin/python -m jev_eval check-env
.venv/bin/python -m jev_eval validate-config "$CONFIG"

echo "Setup and validation complete. No experiment phase was run."
