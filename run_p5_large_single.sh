#!/bin/bash
# Phase 5 Large-Scale Reproducibility: Single experiment launcher
# Usage: bash run_p5_large_single.sh <gpu> <seed> <fwi_type> [fwi_type2] [fwi_weight2]
# Models: 1,2,5,6,8,10,15,16,18,33,34,39,43,44,45,50

GPU=$1; SEED=$2; FWI=$3; FWI2=${4:-}; FWI2_W=${5:-0}

if [ -z "$GPU" ] || [ -z "$SEED" ] || [ -z "$FWI" ]; then
    echo "Usage: bash run_p5_large_single.sh <gpu> <seed> <fwi_type> [fwi_type2] [fwi_weight2]"
    exit 1
fi

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/shengwz/anaconda3/envs/devito/bin/python
cd /data/shengwz/swz/RL-seismic-inversion

MODELS="1 2 5 6 8 10 15 16 18 33 34 39 43 44 45 50"

# Build tag
if [ -n "$FWI2" ] && [ "$FWI2_W" != "0" ]; then
    TAG="MultiFWI_${FWI}_${FWI2}"
else
    TAG="FWI_${FWI}"
fi

COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100"

for idx in $MODELS; do
    OUT="runs/large/${TAG}_cva${idx}_seed${SEED}"
    mkdir -p "$OUT"
    
    echo "=== $(date) ${TAG} CVA[${idx}] seed=${SEED} GPU=$GPU ==="
    
    # Handle TT-only special case
    if [ "$FWI" = "tt" ]; then
        $PY train_rl_fwi.py $COMMON \
            --cva_file_idx $idx --seed $SEED \
            --fwi_type l2 --reward_l2_weight 0.0 --reward_tt_weight 1.0 --reward_tt_log --reward_l1_weight 0.0 --reward_prior_weight 0.0 \
            --steps 2500 --out_dir "$OUT"
    elif [ -n "$FWI2" ] && [ "$FWI2_W" != "0" ]; then
        # Multi-FWI
        $PY train_rl_fwi.py $COMMON \
            --cva_file_idx $idx --seed $SEED \
            --fwi_type $FWI --fwi_type2 $FWI2 --fwi_weight2 $FWI2_W \
            --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
            --steps 5000 --out_dir "$OUT"
    elif [ "$FWI" = "l1l2" ]; then
        # L1+L2
        $PY train_rl_fwi.py $COMMON \
            --cva_file_idx $idx --seed $SEED \
            --fwi_type l2 --reward_l2_weight 1.0 --reward_l1_weight 1.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
            --steps 5000 --out_dir "$OUT"
    else
        # Single reward
        $PY train_rl_fwi.py $COMMON \
            --cva_file_idx $idx --seed $SEED \
            --fwi_type $FWI --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
            --steps 5000 --out_dir "$OUT"
    fi
    
    echo "=== $(date) Done ${TAG} CVA[${idx}] seed=${SEED} ==="
done
echo "GPU${GPU} DONE: ${TAG} seed=${SEED}"
