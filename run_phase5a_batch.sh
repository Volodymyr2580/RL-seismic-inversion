#!/bin/bash
# Phase 5A: Batch training for NCC_zero, NCC_maxlag, Envelope_NCC, AWI
# Usage: bash run_phase5a_batch.sh <gpu_id> <reward_type>
#   reward_type: ncc_zero | ncc_maxlag | envelope_ncc | awi
# Runs on all 6 CVA models (18,50,10,6,8,5)

GPU=$1
REWARD=$2

if [ -z "$GPU" ] || [ -z "$REWARD" ]; then
    echo "Usage: bash run_phase5a_batch.sh <gpu_id> <reward_type>"
    echo "  reward_type: ncc_zero | ncc_maxlag | envelope_ncc | awi"
    exit 1
fi

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/shengwz/anaconda3/envs/devito/bin/python
cd /data/shengwz/swz/RL-seismic-inversion || exit 1

MODELS="18 50 10 6 8 5"

for idx in $MODELS; do
    echo "=== $(date) $REWARD CVA[$idx] GPU=$GPU ==="
    $PY train_rl_fwi.py \
        --policy_type gaussian \
        --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx $idx \
        --geometry transmission \
        --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
        --fwi_type $REWARD \
        --best_criterion l2 \
        --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
        --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 \
        --entropy_bonus 0.02 \
        --out_dir runs/FWI_${REWARD}_cva${idx} \
        --device cuda \
        --early_stop_patience 500 --early_stop_window 100 \
        --save_every 100 --seed 42
    echo "=== $(date) Done $REWARD CVA[$idx] GPU=$GPU ==="
done
echo "GPU${GPU} ALL DONE ($REWARD)"
