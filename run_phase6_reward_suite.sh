#!/bin/bash
# Phase 6 reward suite: run requested single-reward experiments sequentially.
# Each reward group is parallelized across the supplied CUDA devices by
# run_phase6_parallel.sh.
#
# Usage:
#   bash run_phase6_reward_suite.sh [seed] [gpu_list] [reward_list]
#
# Defaults:
#   seed        = 42
#   gpu_list    = 0 1 2 3
#   reward_list = l1l2 tt_only wasserstein wasserstein_w2 ncc_zero ncc_maxlag envelope_ncc awi

set -e

SEED=${1:-42}
GPU_LIST=${2:-"0 1 2 3"}
REWARD_LIST=${3:-"l1l2 tt_only wasserstein wasserstein_w2 ncc_zero ncc_maxlag envelope_ncc awi"}

PROJ=/data/shengwz/swz/RL-seismic-inversion
cd "$PROJ" || exit 1

MASTER_LOG="log_phase6_suite_seed${SEED}.txt"

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

wait_for_existing_phase6_train() {
    while pgrep -af "train_rl_fwi.py .*runs/phase6/" >/dev/null 2>&1; do
        log "WAIT existing Phase6 train_rl_fwi.py jobs are still running"
        sleep 300
    done
}

log "======== PHASE 6 REWARD SUITE START ========"
log "seed=${SEED} gpus=${GPU_LIST}"
log "rewards=${REWARD_LIST}"
log "============================================"

for REWARD in $REWARD_LIST; do
    wait_for_existing_phase6_train
    log "START reward=${REWARD}"
    bash run_phase6_parallel.sh "$REWARD" "$SEED" "$GPU_LIST"
    log "DONE reward=${REWARD}"
done

log "======== PHASE 6 REWARD SUITE DONE ========"
