#!/bin/bash

set -euo pipefail

# Simple wrapper to run hierarchical SNMF steering locally (no Slurm)
# Defaults: STEPS=all, LAYERS=0, MODEL=gpt2-small

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/workspace/snmf_ohad"
LOG_DIR="/workspace/logs"

STEPS="${STEPS:-all}"
LAYERS="${LAYERS:-0}"
MODEL_NAME="${ACT_MODEL_NAME:-gpt2-small}"
RANKS="${RANKS:-50}"
LOG_FILE="$LOG_DIR/hier_snmf_steering_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "Using STEPS=$STEPS" 
echo "Using LAYERS=$LAYERS"
echo "Using MODEL=$MODEL_NAME"
echo "Using RANKS=$RANKS"
echo "Logs: $LOG_FILE"

# Activate virtualenv
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV; create it before running." >&2
  exit 1
fi
source "$VENV/bin/activate"

# Load API keys if present
if [[ -f /workspace/.api_keys ]]; then
  source /workspace/.api_keys
fi

# Disable hf_transfer fast path unless user opts in
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

cd "$REPO_ROOT"

CMD=(bash experiments/run_hier_snmf_steering.sh --steps "$STEPS" --layers "$LAYERS" --ranks "$RANKS" --act-model-name "$MODEL_NAME")

# Run and tee output to log file
"${CMD[@]}" >"$LOG_FILE" 2>&1

echo "Done. See log at $LOG_FILE"
