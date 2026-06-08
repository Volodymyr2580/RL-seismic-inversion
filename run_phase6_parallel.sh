#!/bin/bash
# Phase 6: four-GPU parallel single-reward convergence test.
# Usage:
#   bash run_phase6_parallel.sh [reward] [seed] [gpu_list]
#
# Defaults:
#   reward   = ncc_maxlag
#   seed     = 42
#   gpu_list = 0 1 2 3
#
# This script runs one single-reward experiment per CVA B-spline model and
# distributes the model list across GPUs. It skips completed runs and also
# avoids launching a duplicate if the same output directory is already running.

set -e

REWARD=${1:-ncc_maxlag}
SEED=${2:-42}
GPU_LIST=${3:-"0 1 2 3"}

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib

PY=/data/shengwz/anaconda3/envs/devito/bin/python
PROJ=/data/shengwz/swz/RL-seismic-inversion
cd "$PROJ" || exit 1

MODELS=(1 2 5 6 8 10 15 16 18 50)
GPUS=($GPU_LIST)
N_GPUS=${#GPUS[@]}
COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100"
MASTER_LOG="log_phase6_parallel_${REWARD}_seed${SEED}.txt"

reward_args() {
    case "$REWARD" in
        l1l2|l1+l2)
            echo "--fwi_type l2 --reward_l2_weight 1.0 --reward_l1_weight 1.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0"
            ;;
        tt_only|tt)
            echo "--fwi_type l2 --reward_l2_weight 0.0 --reward_l1_weight 0.0 --reward_tt_weight 1.0 --reward_tt_log --reward_prior_weight 0.0"
            ;;
        wasserstein|wasserstein_w2|ncc_zero|ncc_maxlag|envelope_ncc|awi)
            echo "--fwi_type $REWARD --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0"
            ;;
        *)
            echo "Unknown reward preset: $REWARD" >&2
            echo "Allowed: l1l2, tt_only, wasserstein, wasserstein_w2, ncc_zero, ncc_maxlag, envelope_ncc, awi" >&2
            return 2
            ;;
    esac
}

REWARD_ARGS=$(reward_args)

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

is_running_out_dir() {
    local OUT=$1
    pgrep -af "train_rl_fwi.py .*--out_dir ${OUT}" >/dev/null 2>&1
}

has_visuals() {
    local OUT=$1
    [ -f "$OUT/phase6_visuals/models_residuals_curves.png" ] && \
        [ -f "$OUT/phase6_visuals/shot_wiggle_overlays.png" ]
}

run_visualization() {
    local GPU=$1
    local IDX=$2
    local OUT=$3
    local RUN_LOG=$4

    log "VIS gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/viz_phase6_single_reward.py --run_dir "$OUT" --device cuda >> "$RUN_LOG" 2>&1 || \
        log "WARN visualization failed reward=${REWARD} cva=${IDX} seed=${SEED}"
}

run_model() {
    local GPU=$1
    local IDX=$2
    local OUT="runs/phase6/FWI_${REWARD}_cva${IDX}_seed${SEED}"
    local RUN_LOG="runs/phase6/FWI_${REWARD}_cva${IDX}_seed${SEED}.log"

    mkdir -p "$(dirname "$RUN_LOG")" "$OUT"

    if [ -f "$OUT/policy_final.pt" ] && [ -f "$OUT/final_velocity.npy" ]; then
        log "SKIP completed reward=${REWARD} cva=${IDX} seed=${SEED}"
        if ! has_visuals "$OUT"; then
            run_visualization "$GPU" "$IDX" "$OUT" "$RUN_LOG"
        fi
        return 0
    fi
    if is_running_out_dir "$OUT"; then
        log "SKIP already running reward=${REWARD} cva=${IDX} seed=${SEED}"
        return 0
    fi

    log "RUN gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU $PY train_rl_fwi.py $COMMON \
        --cva_file_idx "$IDX" --seed "$SEED" \
        $REWARD_ARGS \
        --steps 5000 --out_dir "$OUT" > "$RUN_LOG" 2>&1
    log "DONE gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED}"

    run_visualization "$GPU" "$IDX" "$OUT" "$RUN_LOG"
}

worker() {
    local WORKER_ID=$1
    local GPU=${GPUS[$WORKER_ID]}
    local TOTAL=${#MODELS[@]}

    for ((i = WORKER_ID; i < TOTAL; i += N_GPUS)); do
        run_model "$GPU" "${MODELS[$i]}"
    done
}

log "======== PHASE 6 PARALLEL START ========"
log "reward=${REWARD} seed=${SEED} gpus=${GPUS[*]}"
log "models=${MODELS[*]}"
log "========================================"

for ((worker_id = 0; worker_id < N_GPUS; worker_id++)); do
    worker "$worker_id" &
    log "spawn worker=${worker_id} gpu=${GPUS[$worker_id]}"
done

wait
log "======== PHASE 6 PARALLEL DONE ========"
