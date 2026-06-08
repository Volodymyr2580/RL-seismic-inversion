#!/bin/bash
# Phase 6 corrected dense grid launcher.
# Runs reward × CVA tasks through a shared worker queue so multiple jobs can
# share each CUDA device.
#
# Usage:
#   bash run_phase6_dense_grid.sh [seed] [gpu_list] [slots_per_gpu] [out_root] [reward_list]
#
# Defaults:
#   seed          = 42
#   gpu_list      = "0 1 2 3"
#   slots_per_gpu = 4
#   out_root      = runs/phase6_fixed_layout
#   reward_list   = l1l2 tt_only wasserstein wasserstein_w2 ncc_zero ncc_maxlag envelope_ncc awi

set -e

SEED=${1:-42}
GPU_LIST=${2:-"0 1 2 3"}
SLOTS_PER_GPU=${3:-4}
OUT_ROOT=${4:-"runs/phase6_fixed_layout"}
REWARD_LIST=${5:-"l1l2 tt_only wasserstein wasserstein_w2 ncc_zero ncc_maxlag envelope_ncc awi"}

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

PY=/data/shengwz/anaconda3/envs/devito/bin/python
PROJ=/data/shengwz/swz/RL-seismic-inversion
cd "$PROJ" || exit 1

MODELS=(1 2 5 6 8 10 15 16 18 50)
GPUS=($GPU_LIST)
N_GPUS=${#GPUS[@]}
TOTAL_WORKERS=$((N_GPUS * SLOTS_PER_GPU))
COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100"
MASTER_LOG="log_phase6_dense_seed${SEED}.txt"

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

reward_args() {
    local REWARD=$1
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
    local REWARD=$2
    local IDX=$3
    local OUT=$4
    local RUN_LOG=$5

    log "VIS gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/viz_phase6_single_reward.py --run_dir "$OUT" --device cuda >> "$RUN_LOG" 2>&1 || \
        log "WARN visualization failed reward=${REWARD} cva=${IDX} seed=${SEED}"
}

run_task() {
    local GPU=$1
    local REWARD=$2
    local IDX=$3
    local OUT="${OUT_ROOT}/FWI_${REWARD}_cva${IDX}_seed${SEED}"
    local RUN_LOG="${OUT_ROOT}/FWI_${REWARD}_cva${IDX}_seed${SEED}.log"
    local ARGS

    ARGS=$(reward_args "$REWARD")
    mkdir -p "$(dirname "$RUN_LOG")" "$OUT"

    if [ -f "$OUT/policy_final.pt" ] && [ -f "$OUT/final_velocity.npy" ]; then
        log "SKIP completed reward=${REWARD} cva=${IDX} seed=${SEED}"
        if ! has_visuals "$OUT"; then
            run_visualization "$GPU" "$REWARD" "$IDX" "$OUT" "$RUN_LOG"
        fi
        return 0
    fi
    if is_running_out_dir "$OUT"; then
        log "SKIP already running reward=${REWARD} cva=${IDX} seed=${SEED}"
        return 0
    fi

    log "RUN gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED} out=${OUT}"
    CUDA_VISIBLE_DEVICES=$GPU $PY train_rl_fwi.py $COMMON \
        --cva_file_idx "$IDX" --seed "$SEED" \
        $ARGS \
        --steps 5000 --out_dir "$OUT" > "$RUN_LOG" 2>&1
    log "DONE gpu=${GPU} reward=${REWARD} cva=${IDX} seed=${SEED}"

    run_visualization "$GPU" "$REWARD" "$IDX" "$OUT" "$RUN_LOG"
}

TASK_REWARDS=()
TASK_MODELS=()
for REWARD in $REWARD_LIST; do
    for IDX in "${MODELS[@]}"; do
        TASK_REWARDS+=("$REWARD")
        TASK_MODELS+=("$IDX")
    done
done
TOTAL_TASKS=${#TASK_REWARDS[@]}

worker() {
    local WORKER_ID=$1
    local GPU=${GPUS[$((WORKER_ID % N_GPUS))]}

    for ((i = WORKER_ID; i < TOTAL_TASKS; i += TOTAL_WORKERS)); do
        run_task "$GPU" "${TASK_REWARDS[$i]}" "${TASK_MODELS[$i]}"
    done
}

log "======== PHASE 6 DENSE GRID START ========"
log "seed=${SEED} gpus=${GPUS[*]} slots_per_gpu=${SLOTS_PER_GPU} total_workers=${TOTAL_WORKERS}"
log "out_root=${OUT_ROOT}"
log "rewards=${REWARD_LIST}"
log "models=${MODELS[*]}"
log "total_tasks=${TOTAL_TASKS}"
log "=========================================="

for ((worker_id = 0; worker_id < TOTAL_WORKERS; worker_id++)); do
    worker "$worker_id" &
    log "spawn worker=${worker_id} gpu=${GPUS[$((worker_id % N_GPUS))]}"
done

wait
log "======== PHASE 6 DENSE GRID DONE ========"
