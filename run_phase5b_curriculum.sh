#!/bin/bash
# Phase 5B: Curriculum training — 2-stage or 3-stage
# Usage: bash run_phase5b_curriculum.sh <gpu> <stage1_reward> <stage2_reward> [stage3_reward]
#   Stage lengths: 2-stage → 2500+2500; 3-stage → 2000+1500+1500

GPU=$1
S1_REWARD=$2
S2_REWARD=$3
S3_REWARD=$4

if [ -z "$GPU" ] || [ -z "$S1_REWARD" ] || [ -z "$S2_REWARD" ]; then
    echo "Usage: bash run_phase5b_curriculum.sh <gpu> <s1_reward> <s2_reward> [s3_reward]"
    echo "  reward: wasserstein|contrastive|ncc_zero|ncc_maxlag|envelope_ncc|awi|phase_func|l2"
    exit 1
fi

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/shengwz/anaconda3/envs/devito/bin/python
cd /data/shengwz/swz/RL-seismic-inversion || exit 1

MODELS="18 50 10 6 8 5"

# Determine if 2-stage or 3-stage
if [ -n "$S3_REWARD" ]; then
    IS_3STAGE=1
    S1_STEPS=2000 S2_STEPS=1500 S3_STEPS=1500
    TAG="${S1_REWARD}_${S2_REWARD}_${S3_REWARD}"
else
    IS_3STAGE=0
    S1_STEPS=2500 S2_STEPS=2500
    TAG="${S1_REWARD}_${S2_REWARD}"
fi

COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --reward_l1_weight 0.0 --reward_prior_weight 0.0 --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100 --seed 42"

# Helper: run one stage
run_stage() {
    local STAGE_NAME=$1
    local STEPS=$2
    local REWARD=$3
    local OUT_DIR=$4
    local RESUME=${5:-}
    
    local RESUME_ARG=""
    if [ -n "$RESUME" ]; then
        RESUME_ARG="--resume_ckpt $RESUME"
    fi
    
    # Handle special reward names
    local FWI_TYPE=$REWARD
    local TT_W=0.0
    local L2_W=1.0
    
    # TT-only → use L2 reward with tt-only logic
    if [ "$REWARD" = "tt" ]; then
        FWI_TYPE="l2"
        L2_W=0.0
        TT_W=1.0
    fi
    
    echo "=== $(date) Stage=$STAGE_NAME R=$REWARD Steps=$STEPS GPU=$GPU ==="
    
    for idx in $MODELS; do
        local CUR_OUT="${OUT_DIR}_cva${idx}"
        local RESUME_CKPT=""
        if [ -n "$RESUME" ]; then
            RESUME_CKPT="${RESUME}_cva${idx}/policy_best.pt"
            if [ ! -f "$RESUME_CKPT" ]; then
                echo "WARNING: resume ckpt not found: $RESUME_CKPT, skipping CVA${idx}"
                continue
            fi
            RESUME_ARG="--resume_ckpt $RESUME_CKPT"
        fi
        
        echo "  $(date) $STAGE_NAME CVA[$idx] GPU=$GPU"
        $PY train_rl_fwi.py \
            $COMMON \
            --cva_file_idx $idx \
            --fwi_type $FWI_TYPE \
            --reward_l2_weight $L2_W \
            --reward_tt_weight $TT_W \
            --steps $STEPS \
            --out_dir $CUR_OUT \
            $RESUME_ARG
        echo "  $(date) Done $STAGE_NAME CVA[$idx]"
    done
    echo "=== $(date) Done $STAGE_NAME ==="
}

# Execute stages
if [ "$IS_3STAGE" -eq 1 ]; then
    S1_OUT="runs/curriculum_${TAG}_s1"
    S2_OUT="runs/curriculum_${TAG}_s2"
    S3_OUT="runs/curriculum_${TAG}_s3"
    
    run_stage "S1" $S1_STEPS $S1_REWARD $S1_OUT
    run_stage "S2" $S2_STEPS $S2_REWARD $S2_OUT $S1_OUT
    run_stage "S3" $S3_STEPS $S3_REWARD $S3_OUT $S2_OUT
else
    S1_OUT="runs/curriculum_${TAG}_s1"
    S2_OUT="runs/curriculum_${TAG}_s2"
    
    run_stage "S1" $S1_STEPS $S1_REWARD $S1_OUT
    run_stage "S2" $S2_STEPS $S2_REWARD $S2_OUT $S1_OUT
fi

echo "GPU${GPU} ALL DONE ($TAG)"
