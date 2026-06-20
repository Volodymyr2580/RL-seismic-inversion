#!/usr/bin/env bash
# Phase7 multiscale suite: coarse -> mid -> full B-spline control grids.
#
# Goal: test whether general initial models (far_uniform / medium_gradient)
# can converge better when the optimization first captures large-scale
# structure and only then refines with a high-dimensional full grid.
#
# Default matrix:
#   5 pipelines x 3 models x 2 general initial models = 30 pipelines.
#   Each pipeline has 3 serial stages, while pipelines are distributed over GPUs.

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

GPU_LIST="${GPU_LIST:-0,1,2,3}"
RUN_ROOT="${RUN_ROOT:-runs/phase7_multiscale}"
REPORT_DIR="${REPORT_DIR:-reports/phase7_multiscale_suite}"
INIT_ROOT="${INIT_ROOT:-outputs/phase7_sensitivity/initial_models}"
MODEL_FILTER="${MODELS:-marmousi,salt_A_90x270,synthetic}"
INIT_FILTER="${INITS:-far_uniform,medium_gradient}"
PIPELINE_FILTER="${PIPELINES:-ms_tt,ms_tt_ncc,ms_tt_w2,ms_ncc,ms_tt_ncc_w2}"

GROUP_SIZE="${GROUP_SIZE:-32}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
COARSE_STEPS="${COARSE_STEPS:-800}"
MID_STEPS="${MID_STEPS:-800}"
FULL_STEPS="${FULL_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-500}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-100}"
NT="${NT:-1000}"
N_SHOTS="${N_SHOTS:-7}"
DX="${DX:-10.0}"
DT="${DT:-0.001}"
FREQ="${FREQ:-15.0}"
PML_WIDTH="${PML_WIDTH:-40}"

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

stage_schedule() {
  local pipeline="$1"
  case "$pipeline" in
    ms_tt)
      STAGE_REWARDS=(tt_only tt_only tt_only)
      ;;
    ms_tt_ncc)
      STAGE_REWARDS=(tt_only tt_only ncc_zero)
      ;;
    ms_tt_w2)
      STAGE_REWARDS=(tt_only tt_only wasserstein_w2)
      ;;
    ms_ncc)
      STAGE_REWARDS=(ncc_zero ncc_zero ncc_zero)
      ;;
    ms_tt_ncc_w2)
      STAGE_REWARDS=(tt_only ncc_zero wasserstein_w2)
      ;;
    *)
      echo "[error] unknown pipeline: $pipeline"
      exit 1
      ;;
  esac
}

stage_opt_args() {
  local scale="$1"
  local stage_idx="$2"
  OPT_ARGS=(--ratio_token_mean --entropy_bonus 0.0)
  case "$scale" in
    coarse)
      OPT_ARGS+=(--lr 8e-4 --init_temperature 0.35 --final_temperature 0.06 --epsilon_low 0.06 --epsilon_high 0.10 --init_kappa 35.0)
      ;;
    mid)
      OPT_ARGS+=(--lr 5e-4 --init_temperature 0.25 --final_temperature 0.04 --epsilon_low 0.05 --epsilon_high 0.08 --init_kappa 45.0)
      ;;
    full)
      OPT_ARGS+=(--lr 3e-4 --init_temperature 0.20 --final_temperature 0.03 --epsilon_low 0.04 --epsilon_high 0.06 --init_kappa 60.0)
      # Tether full-grid refinement to the mid-scale output to reduce destructive drift.
      OPT_ARGS+=(--reward_tether_weight 0.5 --tether_scale 100.0)
      ;;
  esac
}

echo "[phase7-multiscale] python: $PY"
echo "[phase7-multiscale] project: $PROJ"
echo "[phase7-multiscale] GPUs: $GPU_LIST"
echo "[phase7-multiscale] steps coarse/mid/full=${COARSE_STEPS}/${MID_STEPS}/${FULL_STEPS} G=${GROUP_SIZE}"

"$PY" scripts/phase7_prepare_initial_models.py --out_root "$INIT_ROOT"

MODELS_ALL=(marmousi salt_A_90x270 synthetic)
INITS_ALL=(far_uniform medium_gradient)
PIPELINES_ALL=(ms_tt ms_tt_ncc ms_tt_w2 ms_ncc ms_tt_ncc_w2)

JOBS=()
for pipeline in "${PIPELINES_ALL[@]}"; do
  contains_csv "$PIPELINE_FILTER" "$pipeline" || continue
  for model in "${MODELS_ALL[@]}"; do
    contains_csv "$MODEL_FILTER" "$model" || continue
    for init in "${INITS_ALL[@]}"; do
      contains_csv "$INIT_FILTER" "$init" || continue
      JOBS+=("$pipeline|$model|$init")
    done
  done
done

TOTAL="${#JOBS[@]}"
if [[ "$TOTAL" -lt 1 ]]; then
  echo "[error] no jobs selected"
  exit 1
fi
echo "[phase7-multiscale] selected pipelines: $TOTAL"

mkdir -p "$RUN_ROOT/logs"

run_stage() {
  local gpu="$1"
  local pipeline="$2"
  local model="$3"
  local init="$4"
  local stage_idx="$5"
  local scale="$6"
  local reward="$7"
  local init_path="$8"
  local stage_steps="$9"

  model_args "$model" "$scale"
  reward_args "$reward"
  stage_opt_args "$scale" "$stage_idx"

  local stage_name="stage${stage_idx}_${scale}_${reward}"
  local run_dir="$RUN_ROOT/$pipeline/$model/$init/$stage_name"
  local log_path="$RUN_ROOT/logs/${pipeline}_${model}_${init}_${stage_name}.log"

  if [[ -f "$run_dir/policy_final.pt" && -f "$run_dir/final_velocity.npy" ]]; then
    echo "[skip][gpu $gpu] $pipeline/$model/$init/$stage_name"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[run][gpu $gpu] $pipeline/$model/$init/$stage_name ctrl=${NX_CTRL}x${NZ_CTRL} init=$init_path"
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
    --steps "$stage_steps" \
    --ppo_epochs "$PPO_EPOCHS" \
    --anneal_steps "$stage_steps" \
    --reward_prior_weight 0.0 \
    --best_criterion l2 \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --early_stop_window "$EARLY_STOP_WINDOW" \
    --save_every "$SAVE_EVERY" \
    --device cuda \
    --seed 202609 \
    --out_dir "$run_dir" \
    "${OPT_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    > "$log_path" 2>&1
}

run_pipeline() {
  local gpu="$1"
  local pipeline="$2"
  local model="$3"
  local init="$4"

  stage_schedule "$pipeline"
  local init_path="$INIT_ROOT/$model/$init.npy"
  local base_dir="$RUN_ROOT/$pipeline/$model/$init"

  run_stage "$gpu" "$pipeline" "$model" "$init" 1 coarse "${STAGE_REWARDS[0]}" "$init_path" "$COARSE_STEPS"
  local stage1_dir="$base_dir/stage1_coarse_${STAGE_REWARDS[0]}"
  run_stage "$gpu" "$pipeline" "$model" "$init" 2 mid "${STAGE_REWARDS[1]}" "$stage1_dir/final_velocity.npy" "$MID_STEPS"
  local stage2_dir="$base_dir/stage2_mid_${STAGE_REWARDS[1]}"
  run_stage "$gpu" "$pipeline" "$model" "$init" 3 full "${STAGE_REWARDS[2]}" "$stage2_dir/final_velocity.npy" "$FULL_STEPS"
  local stage3_dir="$base_dir/stage3_full_${STAGE_REWARDS[2]}"

  echo "[viz][gpu $gpu] $pipeline/$model/$init final"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/phase7_visualize_run.py "$stage3_dir" --device cuda >> "$RUN_ROOT/logs/${pipeline}_${model}_${init}_final_viz.log" 2>&1; then
    echo "[warn] final visualization failed for $pipeline/$model/$init"
  fi
}

worker() {
  local worker_id="$1"
  local gpu="${GPUS[$worker_id]}"
  local idx="$worker_id"
  while [[ "$idx" -lt "$TOTAL" ]]; do
    IFS='|' read -r pipeline model init <<< "${JOBS[$idx]}"
    run_pipeline "$gpu" "$pipeline" "$model" "$init"
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
  echo "[phase7-multiscale] some pipelines failed; building partial report"
else
  echo "[phase7-multiscale] all pipelines finished"
fi

"$PY" scripts/phase7_build_multiscale_report.py --runs_root "$RUN_ROOT" --report_dir "$REPORT_DIR"
echo "[phase7-multiscale] report: $REPORT_DIR/index.html"

exit "$FAIL"
