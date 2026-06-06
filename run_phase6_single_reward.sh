#!/bin/bash
# Phase 6: single-reward, single-seed convergence test on 10 CVA B-spline models.
# Usage:
#   bash run_phase6_single_reward.sh <gpu_id> [reward] [seed]
#
# Defaults:
#   reward = ncc_maxlag
#   seed   = 42
#
# Phase 5-compatible training parameters:
#   Gaussian policy, smooth B-spline models, transmission geometry,
#   G=32, 5000 steps, PPO epochs=4, lr=5e-3, temperature 2.0 -> 0.1.

set -e

GPU=$1
REWARD=${2:-ncc_maxlag}
SEED=${3:-42}

if [ -z "$GPU" ]; then
    echo "Usage: bash run_phase6_single_reward.sh <gpu_id> [reward] [seed]"
    echo "Example: bash run_phase6_single_reward.sh 0 ncc_maxlag 42"
    exit 1
fi

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU

PY=/data/shengwz/anaconda3/envs/devito/bin/python
PROJ=/data/shengwz/swz/RL-seismic-inversion
cd "$PROJ" || exit 1

MODELS="1 2 5 6 8 10 15 16 18 50"
COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100"
LOG="log_phase6_${REWARD}_seed${SEED}.txt"

echo "======== PHASE 6 START ========" | tee -a "$LOG"
echo "GPU=$GPU reward=$REWARD seed=$SEED" | tee -a "$LOG"
echo "Models: $MODELS" | tee -a "$LOG"
echo "===============================" | tee -a "$LOG"

for idx in $MODELS; do
    OUT="runs/phase6/FWI_${REWARD}_cva${idx}_seed${SEED}"
    mkdir -p "$OUT"

    if [ -f "$OUT/policy_final.pt" ] && [ -f "$OUT/final_velocity.npy" ]; then
        echo "=== $(date) SKIP completed $REWARD CVA[$idx] seed=$SEED ===" | tee -a "$LOG"
    else
        echo "=== $(date) RUN $REWARD CVA[$idx] seed=$SEED GPU=$GPU ===" | tee -a "$LOG"
        $PY train_rl_fwi.py $COMMON \
            --cva_file_idx "$idx" --seed "$SEED" \
            --fwi_type "$REWARD" \
            --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
            --steps 5000 --out_dir "$OUT" 2>&1 | tee -a "$LOG"
        echo "=== $(date) DONE $REWARD CVA[$idx] seed=$SEED ===" | tee -a "$LOG"
    fi

    echo "=== $(date) VIS $REWARD CVA[$idx] seed=$SEED ===" | tee -a "$LOG"
    $PY scripts/viz_phase6_single_reward.py --run_dir "$OUT" --device cuda 2>&1 | tee -a "$LOG" || \
        echo "=== $(date) WARN visualization failed for $OUT ===" | tee -a "$LOG"
done

echo "======== PHASE 6 DONE ========" | tee -a "$LOG"
