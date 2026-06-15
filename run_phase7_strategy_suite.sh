#!/usr/bin/env bash
# Phase7 strategy suite: test optimizer/parameterization fixes for high-dimensional B-spline RL.
#
# Default matrix:
#   3 models x 3 init models x
#   (5 full-grid strategies x 3 rewards + 2 grid-scale strategies x 2 rewards) = 171 jobs.
#
# Example:
#   bash run_phase7_strategy_suite.sh
#   STEPS=1000 GPU_LIST=0,1,2,3 bash run_phase7_strategy_suite.sh
#   MODELS=marmousi STRATEGIES=full_token_low,full_token_tether REWARDS=tt_only,ncc_zero bash run_phase7_strategy_suite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$SCRIPT_DIR}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/data/shengwz/anaconda3/envs/devito/bin/python" ]]; then
    PY="/data/shengwz/anaconda3/envs/devito/bin/python"
  else
    PY="python"
  fi
fi

if [[ -d "/data/shengwz/anaconda3/envs/devito/lib" ]]; then
  export LD_LIBRARY_PATH="/data/shengwz/anaconda3/envs/devito/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$PROJ"

STEPS="${STEPS:-1000}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-500}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-100}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NT="${NT:-1000}"
N_SHOTS="${N_SHOTS:-7}"
DX="${DX:-10.0}"
DT="${DT:-0.001}"
FREQ="${FREQ:-15.0}"
PML_WIDTH="${PML_WIDTH:-40}"
RUN_ROOT="${RUN_ROOT:-runs/phase7_strategy}"
INIT_ROOT="${INIT_ROOT:-outputs/phase7_sensitivity/initial_models}"
REPORT_DIR="${REPORT_DIR:-reports/phase7_strategy_suite}"

MODEL_FILTER="${MODELS:-marmousi,salt_A_90x270,synthetic}"
INIT_FILTER="${INITS:-far_uniform,medium_gradient,near_blur}"
REWARD_FILTER="${REWARDS:-tt_only,ncc_zero,wasserstein_w2}"
STRATEGY_FILTER="${STRATEGIES:-full_low_joint,full_token_high,full_token_low,full_token_tether,full_token_tether_strong,coarse_token_low,mid_token_low}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
if [[ "$NGPU" -lt 1 ]]; then
  echo "[error] GPU_LIST is empty"
  exit 1
fi

contains_csv() {
  local csv="$1"
  local item="$2"
  [[ ",$csv," == *",$item,"* ]]
}

model_args() {
  local model="$1"
  local scale="$2"
  case "$model" in
    marmousi)
      MODEL_PATH="outputs/phase7_sensitivity/marmousi_nx210_nz70_ctrl30x10/v_true.npy"
      NX=210; NZ=70; NREC=210; VMIN=1000; VMAX=5600
      case "$scale" in
        coarse) NX_CTRL=10; NZ_CTRL=4 ;;
        mid) NX_CTRL=20; NZ_CTRL=7 ;;
        full) NX_CTRL=30; NZ_CTRL=10 ;;
      esac
      ;;
    salt_A_90x270)
      MODEL_PATH="outputs/phase7_sensitivity/salt_nx270_nz90_ctrl36x12/v_true.npy"
      NX=270; NZ=90; NREC=270; VMIN=1000; VMAX=5600
      case "$scale" in
        coarse) NX_CTRL=12; NZ_CTRL=4 ;;
        mid) NX_CTRL=24; NZ_CTRL=8 ;;
        full) NX_CTRL=36; NZ_CTRL=12 ;;
      esac
      ;;
    synthetic)
      MODEL_PATH="outputs/phase7_sensitivity/synthetic_nx210_nz70_ctrl30x10/v_true.npy"
      NX=210; NZ=70; NREC=210; VMIN=1000; VMAX=5600
      case "$scale" in
        coarse) NX_CTRL=10; NZ_CTRL=4 ;;
        mid) NX_CTRL=20; NZ_CTRL=7 ;;
        full) NX_CTRL=30; NZ_CTRL=10 ;;
      esac
      ;;
    *)
      echo "[error] unknown model: $model"
      exit 1
      ;;
  esac
}

reward_args() {
  local reward="$1"
  REWARD_ARGS=()
  case "$reward" in
    tt_only)
      REWARD_ARGS=(--fwi_type l2 --reward_l1_weight 0.0 --reward_l2_weight 0.0 --reward_tt_weight 1.0 --reward_tt_log)
      ;;
    ncc_zero)
      REWARD_ARGS=(--fwi_type ncc_zero --reward_l1_weight 0.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    wasserstein_w2)
      REWARD_ARGS=(--fwi_type wasserstein_w2 --reward_l1_weight 0.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    *)
      echo "[error] unknown reward: $reward"
      exit 1
      ;;
  esac
}

strategy_args() {
  local strategy="$1"
  CTRL_SCALE="full"
  STRATEGY_ARGS=()
  case "$strategy" in
    full_low_joint)
      CTRL_SCALE="full"
      STRATEGY_ARGS=(--lr 5e-4 --init_temperature 0.30 --final_temperature 0.05 --entropy_bonus 0.0 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 40.0)
      ;;
    full_token_high)
      CTRL_SCALE="full"
      STRATEGY_ARGS=(--ratio_token_mean --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --entropy_bonus 0.02 --epsilon_low 0.20 --epsilon_high 0.27 --init_kappa 4.0)
      ;;
    full_token_low)
      CTRL_SCALE="full"
      STRATEGY_ARGS=(--ratio_token_mean --lr 5e-4 --init_temperature 0.30 --final_temperature 0.05 --entropy_bonus 0.0 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 40.0)
      ;;
    full_token_tether)
      CTRL_SCALE="full"
      STRATEGY_ARGS=(--ratio_token_mean --lr 5e-4 --init_temperature 0.30 --final_temperature 0.05 --entropy_bonus 0.0 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 40.0 --reward_tether_weight 0.5 --tether_scale 100.0)
      ;;
    full_token_tether_strong)
      CTRL_SCALE="full"
      STRATEGY_ARGS=(--ratio_token_mean --lr 2e-4 --init_temperature 0.20 --final_temperature 0.03 --entropy_bonus 0.0 --epsilon_low 0.03 --epsilon_high 0.05 --init_kappa 80.0 --reward_tether_weight 1.0 --tether_scale 100.0)
      ;;
    coarse_token_low)
      CTRL_SCALE="coarse"
      STRATEGY_ARGS=(--ratio_token_mean --lr 5e-4 --init_temperature 0.30 --final_temperature 0.05 --entropy_bonus 0.0 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 40.0)
      ;;
    mid_token_low)
      CTRL_SCALE="mid"
      STRATEGY_ARGS=(--ratio_token_mean --lr 5e-4 --init_temperature 0.30 --final_temperature 0.05 --entropy_bonus 0.0 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 40.0)
      ;;
    *)
      echo "[error] unknown strategy: $strategy"
      exit 1
      ;;
  esac
}

echo "[phase7-strategy] python: $PY"
echo "[phase7-strategy] project: $PROJ"
echo "[phase7-strategy] GPUs: $GPU_LIST"
echo "[phase7-strategy] steps=$STEPS group_size=$GROUP_SIZE ppo_epochs=$PPO_EPOCHS"

"$PY" scripts/phase7_prepare_initial_models.py --out_root "$INIT_ROOT"

MODELS_ALL=(marmousi salt_A_90x270 synthetic)
INITS_ALL=(far_uniform medium_gradient near_blur)
REWARDS_ALL=(tt_only ncc_zero wasserstein_w2)
STRATEGIES_ALL=(full_low_joint full_token_high full_token_low full_token_tether full_token_tether_strong coarse_token_low mid_token_low)

JOBS=()
for strategy in "${STRATEGIES_ALL[@]}"; do
  contains_csv "$STRATEGY_FILTER" "$strategy" || continue
  for reward in "${REWARDS_ALL[@]}"; do
    contains_csv "$REWARD_FILTER" "$reward" || continue
    # Grid-scale strategies focus on the two strongest/most stable rewards.
    if [[ "$strategy" == coarse_* || "$strategy" == mid_* ]]; then
      [[ "$reward" == "wasserstein_w2" ]] && continue
    fi
    for model in "${MODELS_ALL[@]}"; do
      contains_csv "$MODEL_FILTER" "$model" || continue
      for init in "${INITS_ALL[@]}"; do
        contains_csv "$INIT_FILTER" "$init" || continue
        JOBS+=("$strategy|$model|$init|$reward")
      done
    done
  done
done

TOTAL="${#JOBS[@]}"
if [[ "$TOTAL" -lt 1 ]]; then
  echo "[error] no jobs selected"
  exit 1
fi
echo "[phase7-strategy] selected jobs: $TOTAL"

mkdir -p "$RUN_ROOT/logs"

run_job() {
  local gpu="$1"
  local strategy="$2"
  local model="$3"
  local init="$4"
  local reward="$5"

  strategy_args "$strategy"
  model_args "$model" "$CTRL_SCALE"
  reward_args "$reward"

  local init_path="$INIT_ROOT/$model/$init.npy"
  local run_dir="$RUN_ROOT/$strategy/$model/$init/$reward"
  local log_path="$RUN_ROOT/logs/${strategy}_${model}_${init}_${reward}.log"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[error] missing true model: $MODEL_PATH"
    exit 1
  fi
  if [[ ! -f "$init_path" ]]; then
    echo "[error] missing init model: $init_path"
    exit 1
  fi

  mkdir -p "$run_dir"
  if [[ -f "$run_dir/policy_final.pt" && -f "$run_dir/final_velocity.npy" ]]; then
    echo "[skip][gpu $gpu] $strategy/$model/$init/$reward already complete"
    return 0
  fi

  echo "[run][gpu $gpu] $strategy/$model/$init/$reward ctrl=${NX_CTRL}x${NZ_CTRL}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train_rl_fwi.py \
    --model_source npy \
    --model_path "$MODEL_PATH" \
    --init_velocity_path "$init_path" \
    --policy_type mean \
    --geometry transmission \
    --source_depth bottom \
    --receiver_depth top \
    --nx_model "$NX" \
    --nz_model "$NZ" \
    --nx_ctrl "$NX_CTRL" \
    --nz_ctrl "$NZ_CTRL" \
    --v_min "$VMIN" \
    --v_max "$VMAX" \
    --n_shots "$N_SHOTS" \
    --n_receivers "$NREC" \
    --nt "$NT" \
    --dx "$DX" \
    --dt "$DT" \
    --freq "$FREQ" \
    --pml_width "$PML_WIDTH" \
    --group_size "$GROUP_SIZE" \
    --steps "$STEPS" \
    --ppo_epochs "$PPO_EPOCHS" \
    --anneal_steps "$STEPS" \
    --reward_prior_weight 0.0 \
    --best_criterion l2 \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --early_stop_window "$EARLY_STOP_WINDOW" \
    --save_every "$SAVE_EVERY" \
    --device cuda \
    --seed 202608 \
    --out_dir "$run_dir" \
    "${STRATEGY_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    > "$log_path" 2>&1

  echo "[viz][gpu $gpu] $strategy/$model/$init/$reward"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/phase7_visualize_run.py "$run_dir" --device cuda >> "$log_path" 2>&1; then
    echo "[warn] visualization failed for $strategy/$model/$init/$reward; see $log_path"
  fi
}

worker() {
  local worker_id="$1"
  local gpu="${GPUS[$worker_id]}"
  local idx="$worker_id"
  while [[ "$idx" -lt "$TOTAL" ]]; do
    IFS='|' read -r strategy model init reward <<< "${JOBS[$idx]}"
    run_job "$gpu" "$strategy" "$model" "$init" "$reward"
    idx=$((idx + NGPU))
  done
}

PIDS=()
for wid in $(seq 0 $((NGPU - 1))); do
  worker "$wid" &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[phase7-strategy] some jobs failed; building partial report"
else
  echo "[phase7-strategy] all jobs finished"
fi

"$PY" scripts/phase7_build_strategy_report.py --runs_root "$RUN_ROOT" --report_dir "$REPORT_DIR"
echo "[phase7-strategy] report: $REPORT_DIR/index.html"

exit "$FAIL"
