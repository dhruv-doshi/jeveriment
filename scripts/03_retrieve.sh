#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="${CONFIG:-configs/pilot.yaml}"

# The Hugging Face Xet transfer has stalled on this machine while fetching
# pinned model weights. HTTP supports an incomplete-file resume in its cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

.venv/bin/python -m jev_eval retrieve "$CONFIG"
