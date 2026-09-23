#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 QUERIES APPROVED_PAIRS REQUEST_BUDGET TOKEN_BUDGET" >&2
  echo "Run after 03_retrieve.sh. Example SciFact pilot: 30 queries x 5183 documents = 155490 pairs." >&2
  exit 2
fi

run_logged jev_scan .venv/bin/python -m jev_eval jev-scan "$CONFIG" \
  --queries "$1" --approve-pairs "$2" --request-budget "$3" --token-budget "$4"
run_logged compare_retrieval .venv/bin/python -m jev_eval compare-retrieval "$CONFIG"
