#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv sync --extra local
.venv/bin/python -m jev_eval check-env
.venv/bin/python -m jev_eval validate-config "${CONFIG:-configs/pilot.yaml}"

echo "Setup and validation complete. No experiment phase was run."
