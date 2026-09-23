#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="${CONFIG:-configs/pilot.yaml}"

.venv/bin/python -m jev_eval evaluate "$CONFIG"
.venv/bin/python -m jev_eval plot runs/scifact_train_pilot_v1

echo "Evaluation and plotting complete."
