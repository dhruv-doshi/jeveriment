#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

# The Hugging Face Xet transfer has stalled on this machine while fetching
# pinned model weights. HTTP supports an incomplete-file resume in its cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

run_logged retrieve .venv/bin/python -m jev_eval retrieve "$CONFIG"
