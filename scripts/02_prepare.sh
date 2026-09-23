#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environment

run_logged prepare .venv/bin/python -m jev_eval prepare "$CONFIG" "$@"
