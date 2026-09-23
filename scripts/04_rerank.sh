#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# Pass one or more systems to split this phase across sessions.
systems=("$@")
if [[ $# -eq 0 ]]; then systems=(qwen bge jev); fi
for system in "${systems[@]}"; do
  case "$system" in
    qwen|bge|jev) ;;
    *) echo "Unknown system: $system (use qwen, bge, jev)" >&2; exit 2 ;;
  esac
  run_logged "rerank_${system}" .venv/bin/python -m jev_eval rerank "$CONFIG" --system "$system" --resume
done
