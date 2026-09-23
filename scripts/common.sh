#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="${CONFIG:-configs/pilot.yaml}"

require_environment() {
  if [[ ! -x .venv/bin/python ]]; then
    echo "Local environment missing. Run ./scripts/01_setup.sh first." >&2
    exit 1
  fi
}

run_directory() {
  .venv/bin/python -c 'import sys; from jev_eval.config import load_config; from jev_eval.pipeline import run_dir; print(run_dir(load_config(sys.argv[1])))' "$CONFIG"
}

run_logged() {
  local phase="$1"
  shift
  local directory timestamp
  directory="$(run_directory)"
  mkdir -p "$directory/logs"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  echo "Log: $directory/logs/${phase}_${timestamp}.log"
  "$@" 2>&1 | tee "$directory/logs/${phase}_${timestamp}.log"
}
