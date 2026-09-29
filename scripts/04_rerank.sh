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
  budget_args=()
  if [[ "$system" == jev ]]; then
    if [[ -n "${JEV_RERANK_REQUEST_BUDGET:-}" ]]; then
      budget_args+=(--request-budget "$JEV_RERANK_REQUEST_BUDGET")
    fi
    if [[ -n "${JEV_RERANK_TOKEN_BUDGET:-}" ]]; then
      budget_args+=(--token-budget "$JEV_RERANK_TOKEN_BUDGET")
    fi
  fi
  run_logged "rerank_${system}" .venv/bin/python -m jev_eval rerank "$CONFIG" --system "$system" --resume "${budget_args[@]}"
done
