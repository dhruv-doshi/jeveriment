#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 FROZEN_THRESHOLD REQUEST_BUDGET TOKEN_BUDGET" >&2
  echo "Requires complete Jev fixed-pool scores from 04_rerank.sh jev." >&2
  exit 2
fi

run_logged benchmark_decisions .venv/bin/python -m jev_eval benchmark-decisions "$CONFIG" \
  --threshold "$1" --request-budget "$2" --token-budget "$3"
