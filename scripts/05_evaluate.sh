#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

run_logged evaluate .venv/bin/python -m jev_eval evaluate "$CONFIG"
run_logged plot .venv/bin/python -m jev_eval plot "$(run_directory)"

echo "Evaluation and plotting complete."
