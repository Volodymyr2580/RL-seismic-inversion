#!/bin/bash
# Phase 5 Large-Scale Master Script — FULLY AUTONOMOUS
# Runs ALL 11 rounds sequentially, 4 GPUs in parallel per round
# Total: ~1360 training runs, ~5-7 days
# Log: /data/shengwz/swz/RL-seismic-inversion/log_p5l_master.txt

set -e

MASTER_LOG="/data/shengwz/swz/RL-seismic-inversion/log_p5l_master.txt"
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
PY=/data/shengwz/anaconda3/envs/devito/bin/python
PROJ=/data/shengwz/swz/RL-seismic-inversion
cd "$PROJ"

MODELS="1 2 5 6 8 10 15 16 18 33 34 39 43 44 45 50"
SEEDS=(42 123 456 789 101112)
GPUS=(0 1 2 3)
N_GPUS=4

COMMON="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --geometry transmission --best_criterion l2 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run_one() {
    # Usage: run_one <gpu> <seed> <fwi_type> <steps> [fwi_type2] [fwi_weight2] [out_prefix] [tt_weight]
    local GPU=$1 SEED=$2 FWI=$3 STEPS=$4 FWI2=${5:-} FWI2_W=${6:-0} OUT_PREFIX=${7:-} TT_W=${8:-0}
    
    export CUDA_VISIBLE_DEVICES=$GPU
    
    # Build extra args
    local EXTRA=""
    if [ "$TT_W" != "0" ]; then
        EXTRA="--reward_tt_weight $TT_W --reward_tt_log"
    fi
    
    for idx in $MODELS; do
        local OUT
        if [ -n "$OUT_PREFIX" ]; then
            OUT="runs/large/${OUT_PREFIX}_cva${idx}_seed${SEED}"
        elif [ -n "$FWI2" ] && [ "$FWI2_W" != "0" ]; then
            OUT="runs/large/MultiFWI_${FWI}_${FWI2}_cva${idx}_seed${SEED}"
        else
            OUT="runs/large/FWI_${FWI}_cva${idx}_seed${SEED}"
        fi
        mkdir -p "$OUT"
        
        if [ -f "$OUT/policy_final.pt" ]; then
            continue  # skip completed
        fi
        
        if [ "$FWI" = "tt" ]; then
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED \
                --fwi_type l2 --reward_l2_weight 0.0 --reward_tt_weight 1.0 --reward_tt_log --reward_l1_weight 0.0 --reward_prior_weight 0.0 \
                --steps $STEPS --out_dir "$OUT" &>/dev/null
        elif [ "$FWI" = "l1l2" ]; then
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED \
                --fwi_type l2 --reward_l2_weight 1.0 --reward_l1_weight 1.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
                --steps $STEPS --out_dir "$OUT" &>/dev/null
        elif [ -n "$FWI2" ] && [ "$FWI2_W" != "0" ]; then
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED $EXTRA \
                --fwi_type $FWI --fwi_type2 $FWI2 --fwi_weight2 $FWI2_W \
                --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_prior_weight 0.0 \
                --steps $STEPS --out_dir "$OUT" &>/dev/null
        else
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED $EXTRA \
                --fwi_type $FWI --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_prior_weight 0.0 \
                --steps $STEPS --out_dir "$OUT" &>/dev/null
        fi
    done
}

# Curriculum stage2 — resumes from stage1 checkpoint
run_curr_s2() {
    local GPU=$1 SEED=$2 S1_PREFIX=$3 S2_FWI=$4 OUT_PREFIX=$5
    export CUDA_VISIBLE_DEVICES=$GPU
    
    for idx in $MODELS; do
        local CKPT="runs/large/${S1_PREFIX}_cva${idx}_seed${SEED}/policy_best.pt"
        local OUT="runs/large/${OUT_PREFIX}_cva${idx}_seed${SEED}"
        mkdir -p "$OUT"
        
        if [ -f "$OUT/policy_final.pt" ]; then continue; fi
        if [ ! -f "$CKPT" ]; then
            log "  SKIP ${OUT_PREFIX} CVA${idx} s${SEED}: no ckpt"
            continue
        fi
        
        if [ "$S2_FWI" = "l1l2" ]; then
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED \
                --fwi_type l2 --reward_l2_weight 1.0 --reward_l1_weight 1.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
                --steps 2500 --out_dir "$OUT" --resume_ckpt "$CKPT" > /dev/null 2>&1
        else
            $PY train_rl_fwi.py $COMMON --cva_file_idx $idx --seed $SEED \
                --fwi_type $S2_FWI --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
                --steps 2500 --out_dir "$OUT" --resume_ckpt "$CKPT" > /dev/null 2>&1
        fi
    done
}

# ═══════════════════════════════════════════════════════
log "======== PHASE 5 LARGE-SCALE MASTER START ========"
log "Models: $MODELS"
log "Seeds: ${SEEDS[*]}"
log "GPUs: ${GPUS[*]}"
log "=================================================="

ROUND=0

# ── ROUND 1: Wass_abs ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: Wasserstein_abs × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed wasserstein 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed wasserstein 5000; log "  GPU0: seed=$seed (catch-up)"; wait
log "ROUND $ROUND DONE"

# ── ROUND 2: Contrastive ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: Contrastive × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed contrastive 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed contrastive 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 3: NCC_maxlag ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: NCC_maxlag × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed ncc_maxlag 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed ncc_maxlag 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 4: NCC_zero ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: NCC_zero × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed ncc_zero 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed ncc_zero 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 5: Envelope_NCC ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: Envelope_NCC × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed envelope_ncc 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed envelope_ncc 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 6: AWI ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: AWI × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed awi 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed awi 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 7: Phase_func ──
ROUND=$((ROUND+1)); log "ROUND $ROUND/11: Phase_func × ${#SEEDS[@]} seeds"
for i in $(seq 0 $((N_GPUS-1))); do
    seed=${SEEDS[$i]}
    run_one ${GPUS[$i]} $seed phase_func 5000 &
    log "  GPU${GPUS[$i]}: seed=$seed"
done
wait
seed=${SEEDS[4]}; run_one 0 $seed phase_func 5000; log "  GPU0: seed=$seed"; wait
log "ROUND $ROUND DONE"

# ── ROUND 8: Mixed +TT (Wass+TT, Contra+TT, NCCm+TT) ──
ROUND=$((ROUND+1))
log "ROUND $ROUND/11: Mixed +TT (3 groups) × ${#SEEDS[@]} seeds"
for seed in "${SEEDS[@]}"; do
    log "  Mixed+TT seed=$seed"
    run_one 0 $seed wasserstein 5000 "" 0 "Mix_wass_TT" 1.0 &
    run_one 1 $seed contrastive 5000 "" 0 "Mix_contra_TT" 1.0 &
    run_one 2 $seed ncc_maxlag 5000 "" 0 "Mix_nccm_TT" 1.0 &
    wait
    log "  Mixed+TT seed=$seed done"
done
log "ROUND $ROUND DONE"

# ── ROUND 9: Multi-FWI (4 groups) ──
ROUND=$((ROUND+1))
log "ROUND $ROUND/11: Multi-FWI (4 groups) × ${#SEEDS[@]} seeds"
for seed in "${SEEDS[@]}"; do
    log "  MultiFWI seed=$seed"
    run_one 0 $seed wasserstein 5000 contrastive 1.0 "MultiFWI_wass_contra" &
    run_one 1 $seed wasserstein 5000 ncc_maxlag 1.0 "MultiFWI_wass_nccm" &
    run_one 2 $seed wasserstein 5000 awi 1.0 "MultiFWI_wass_awi" &
    run_one 3 $seed contrastive 5000 ncc_maxlag 1.0 "MultiFWI_contra_nccm" &
    wait
    log "  MultiFWI seed=$seed done"
done
log "ROUND $ROUND DONE"

# ── ROUND 10: Curriculum Stage1 (Wass2500, TT2500, NCCm2500) ──
ROUND=$((ROUND+1))
log "ROUND $ROUND/11: Curriculum Stage1 (3 groups) × ${#SEEDS[@]} seeds"
for seed in "${SEEDS[@]}"; do
    log "  Stage1 seed=$seed"
    run_one 0 $seed wasserstein 2500 "" 0 "S1_wass2500" &
    run_one 1 $seed tt 2500 "" 0 "S1_tt2500" &
    run_one 2 $seed ncc_maxlag 2500 "" 0 "S1_nccm2500" &
    wait
    log "  Stage1 seed=$seed done"
done
log "ROUND $ROUND DONE"

# ── ROUND 11: Curriculum Stage2 (5 groups) ──
ROUND=$((ROUND+1))
log "ROUND $ROUND/11: Curriculum Stage2 (5 groups) × ${#SEEDS[@]} seeds"
for seed in "${SEEDS[@]}"; do
    log "  Stage2 seed=$seed"
    # C1: Wass→Contra, C2: Wass→L1+L2, C3: TT→Contra, C4: NCCm→Contra, C5: Wass→NCCm
    run_curr_s2 0 $seed "S1_wass2500" contrastive "C1_wass_contra" &
    run_curr_s2 1 $seed "S1_wass2500" l1l2 "C2_wass_l1l2" &
    run_curr_s2 2 $seed "S1_tt2500" contrastive "C3_tt_contra" &
    run_curr_s2 3 $seed "S1_nccm2500" contrastive "C4_nccm_contra" &
    wait
    run_curr_s2 0 $seed "S1_wass2500" ncc_maxlag "C5_wass_nccm" &
    wait
    log "  Stage2 seed=$seed done"
done
log "ROUND $ROUND DONE"

log "=================================================="
log "ALL 11 ROUNDS COMPLETE"
log "Total: 1360 training runs"
log "Output: runs/large/"
log "=================================================="
