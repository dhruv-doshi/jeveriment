#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="${CONFIG:-configs/pilot.yaml}"

# Each command is resumable. If interrupted, rerun this script.
.venv/bin/python -m jev_eval rerank "$CONFIG" --system qwen --resume
.venv/bin/python -m jev_eval rerank "$CONFIG" --system bge --resume
.venv/bin/python -m jev_eval rerank "$CONFIG" --system jev --resume
