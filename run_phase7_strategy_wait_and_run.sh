#!/usr/bin/env bash
# Wait until selected GPUs are sufficiently free, then launch Phase7 strategy suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-600}"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-2000}"
LOG_PATH="${LOG_PATH:-phase7_strategy_wait_and_run.nohup}"

echo "[wait] $(date) waiting for GPUs=$GPU_LIST mem_threshold=${MEM_THRESHOLD_MIB}MiB interval=${CHECK_INTERVAL_SEC}s"

while true; do
  if pgrep -u "$USER" -f "run_phase7_strategy_suite.sh" >/dev/null 2>&1; then
    echo "[wait] $(date) phase7 strategy suite is already running for $USER; exiting"
    exit 0
  fi

  all_free=1
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  for gpu in "${GPUS[@]}"; do
    mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    echo "[wait] $(date) gpu=$gpu mem=${mem_used}MiB util=${util}%"
    if [[ "$mem_used" -gt "$MEM_THRESHOLD_MIB" ]]; then
      all_free=0
    fi
  done

  if [[ "$all_free" -eq 1 ]]; then
    echo "[wait] $(date) GPUs are free; launching strategy suite"
    env PY="${PY:-/data/shengwz/anaconda3/envs/devito/bin/python}" \
      GPU_LIST="$GPU_LIST" \
      RUN_ROOT="${RUN_ROOT:-runs/phase7_strategy}" \
      REPORT_DIR="${REPORT_DIR:-reports/phase7_strategy_suite}" \
      bash run_phase7_strategy_suite.sh
    exit $?
  fi

  sleep "$CHECK_INTERVAL_SEC"
done
