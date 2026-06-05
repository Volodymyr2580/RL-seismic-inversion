#!/bin/bash
# Parallel R11 launcher — uses existing R1/R3 5000-step checkpoints for C1/C2/C4/C5
# Usage: bash run_r11_parallel.sh <gpu> <task>

GPU=$1; TASK=$2
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/shengwz/anaconda3/envs/devito/bin/python
cd /data/shengwz/swz/RL-seismic-inversion

MODELS="1 2 5 6 8 10 15 16 18 33 34 39 43 44 45 50"
SEEDS=(42 123 456 789 101112)
COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100 --steps 2500"

# C1: Wass→Contra, C2: Wass→L1+L2, C4: NCCm→Contra, C5: Wass→NCCm
case $TASK in
    C1) S1_PREFIX="FWI_wasserstein"; S2_FWI="contrastive"; OUT="C1_wass_contra" ;;
    C2) S1_PREFIX="FWI_wasserstein"; S2_FWI="l1l2";       OUT="C2_wass_l1l2" ;;
    C3) S1_PREFIX="S1_tt2500";       S2_FWI="contrastive"; OUT="C3_tt_contra" ;;
    C4) S1_PREFIX="FWI_ncc_maxlag";  S2_FWI="contrastive"; OUT="C4_nccm_contra" ;;
    C5) S1_PREFIX="FWI_wasserstein"; S2_FWI="ncc_maxlag";  OUT="C5_wass_nccm" ;;
    *)  echo "Unknown task: $TASK"; exit 1 ;;
esac

for seed in "${SEEDS[@]}"; do
    echo "=== $(date) $TASK seed=$seed GPU=$GPU ==="
    for idx in $MODELS; do
        CKPT="runs/large/${S1_PREFIX}_cva${idx}_seed${seed}/policy_best.pt"
        OUTDIR="runs/large/${OUT}_cva${idx}_seed${seed}"
        mkdir -p "$OUTDIR"
        
        if [ -f "$OUTDIR/policy_final.pt" ]; then continue; fi
        if [ ! -f "$CKPT" ]; then echo "  MISSING CKPT: $CKPT"; continue; fi
        
        if [ "$S2_FWI" = "l1l2" ]; then
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $seed \
                --fwi_type l2 --reward_l2_weight 1.0 --reward_l1_weight 1.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
                --out_dir "$OUTDIR" --resume_ckpt "$CKPT" &>/dev/null
        else
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $seed \
                --fwi_type $S2_FWI --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
                --out_dir "$OUTDIR" --resume_ckpt "$CKPT" &>/dev/null
        fi
    done
    echo "=== $(date) $TASK seed=$seed DONE ==="
done
echo "$TASK ALL DONE GPU=$GPU"
