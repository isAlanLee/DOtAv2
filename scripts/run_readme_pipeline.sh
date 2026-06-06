#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CUDA_DEVICES="0"
NPROC_PER_NODE="1"
INITIAL_HYPES="opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml"
DOTA_HYPES="opencood/hypes_yaml/point_pillar_intermediate_fusion_dota.yaml"
FUSION_METHOD="intermediate"
MBE_OUTPUT_DIR="/root/autodl-tmp/out_mbe"
PSEUDO_LABEL_ROOT="/root/autodl-tmp/out_pseudo_lables"
LOG_ROOT="pipeline_logs"
INITIAL_DETECTOR_DIR=""
FINAL_CHECKPOINT_DIR=""
SKIP_TEST=0
DRY_RUN=0
NO_SYSTEM_SHUTDOWN=0
SHUTDOWN_COMMAND="shutdown -h now"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_readme_pipeline.sh [options]

Options:
  --python PATH                  Python executable. Default: python
  --cuda-devices IDS             CUDA_VISIBLE_DEVICES value. Default: 0
  --nproc-per-node N             torch distributed processes. Default: 1
  --initial-hypes PATH           Initial detector yaml.
  --dota-hypes PATH              DOTA training yaml.
  --fusion-method NAME           Fusion method. Default: intermediate
  --mbe-output-dir PATH          MBE output/cache root. Default: /root/autodl-tmp/out_mbe
  --pseudo-label-root PATH       Initial pseudo-label root. Default: /root/autodl-tmp/out_pseudo_lables
  --log-root PATH                Log root under repo. Default: pipeline_logs
  --initial-detector-dir PATH    Existing initial detector checkpoint; skips initial training.
  --final-checkpoint-dir PATH    Existing final checkpoint; skips final pseudo-label training.
  --skip-test                    Skip final inference test.
  --dry-run                      Print/log commands without executing them; no shutdown.
  --shutdown-command CMD         Command run after failure. Default: shutdown -h now
  --no-system-shutdown           Stop pipeline on failure without powering off.
  -h, --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-devices) CUDA_DEVICES="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --initial-hypes) INITIAL_HYPES="$2"; shift 2 ;;
    --dota-hypes) DOTA_HYPES="$2"; shift 2 ;;
    --fusion-method) FUSION_METHOD="$2"; shift 2 ;;
    --mbe-output-dir) MBE_OUTPUT_DIR="$2"; shift 2 ;;
    --pseudo-label-root) PSEUDO_LABEL_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --initial-detector-dir) INITIAL_DETECTOR_DIR="$2"; shift 2 ;;
    --final-checkpoint-dir) FINAL_CHECKPOINT_DIR="$2"; shift 2 ;;
    --skip-test) SKIP_TEST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --shutdown-command) SHUTDOWN_COMMAND="$2"; shift 2 ;;
    --no-system-shutdown) NO_SYSTEM_SHUTDOWN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_ROOT/$LOG_ROOT/$RUN_ID"
SUMMARY_LOG="$RUN_DIR/pipeline_summary.log"
mkdir -p "$RUN_DIR"

CURRENT_STEP="startup"
CURRENT_LOG="$SUMMARY_LOG"

log_summary() {
  local message="$1"
  local line
  line="[$(date '+%Y-%m-%d %H:%M:%S')] $message"
  echo "$line"
  printf '%s\n' "$line" >> "$SUMMARY_LOG"
}

system_shutdown() {
  local reason="$1"
  local shutdown_log="$RUN_DIR/shutdown.log"
  {
    echo "reason: $reason"
    echo "command: $SHUTDOWN_COMMAND"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S')"
  } >> "$shutdown_log"

  if [[ "$DRY_RUN" -eq 1 || "$NO_SYSTEM_SHUTDOWN" -eq 1 ]]; then
    log_summary "SYSTEM SHUTDOWN SKIPPED by --dry-run or --no-system-shutdown"
    return
  fi

  log_summary "SYSTEM SHUTDOWN START: $SHUTDOWN_COMMAND"
  set +e
  bash -lc "$SHUTDOWN_COMMAND" >> "$shutdown_log" 2>&1
  local rc=$?
  set -e
  echo "return_code: $rc" >> "$shutdown_log"
  echo "finished_at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$shutdown_log"
  if [[ "$rc" -ne 0 ]]; then
    log_summary "SYSTEM SHUTDOWN COMMAND FAILED with return code $rc; see $shutdown_log"
  else
    log_summary "SYSTEM SHUTDOWN COMMAND ISSUED"
  fi
}

fail_shutdown() {
  local reason="$1"
  local rc="${2:-1}"
  log_summary "SHUTDOWN: $reason"
  system_shutdown "$reason"
  exit "$rc"
}

on_unexpected_error() {
  local rc=$?
  fail_shutdown "unexpected error in step $CURRENT_STEP; see $CURRENT_LOG" "$rc"
}

trap on_unexpected_error ERR

step_log_path() {
  local number="$1"
  local name="$2"
  printf '%s/%02d_%s.log' "$RUN_DIR" "$number" "$name"
}

run_step() {
  local number="$1"
  local name="$2"
  shift 2
  local log_path
  log_path="$(step_log_path "$number" "$name")"
  CURRENT_STEP="$name"
  CURRENT_LOG="$log_path"

  log_summary "STEP $(printf '%02d' "$number") START $name: $*"
  {
    echo "step: $name"
    echo "cwd: $REPO_ROOT"
    echo "command: $*"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
  } > "$log_path"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: command not executed." >> "$log_path"
    log_summary "STEP $(printf '%02d' "$number") DRY-RUN $name"
    return 0
  fi

  set +e
  (
    cd "$REPO_ROOT"
    "$@"
  ) >> "$log_path" 2>&1
  local rc=$?
  set -e
  {
    echo
    echo "finished_at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "return_code: $rc"
  } >> "$log_path"

  if [[ "$rc" -ne 0 ]]; then
    fail_shutdown "step $(printf '%02d' "$number") $name failed with return code $rc; see $log_path" "$rc"
  fi
  log_summary "STEP $(printf '%02d' "$number") DONE $name: log=$log_path"
}

write_step_log() {
  local number="$1"
  local name="$2"
  local log_path
  shift 2
  log_path="$(step_log_path "$number" "$name")"
  CURRENT_STEP="$name"
  CURRENT_LOG="$log_path"
  log_summary "STEP $(printf '%02d' "$number") START $name"
  printf '%s\n' "$@" > "$log_path"
  log_summary "STEP $(printf '%02d' "$number") DONE $name: log=$log_path"
}

require_file() {
  local path="$1"
  local description="$2"
  [[ -f "$path" ]] || fail_shutdown "missing $description: $path"
}

require_dir() {
  local path="$1"
  local description="$2"
  [[ -d "$path" ]] || fail_shutdown "missing $description: $path"
}

require_glob() {
  local pattern="$1"
  local description="$2"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  [[ "${#matches[@]}" -gt 0 ]] || fail_shutdown "missing $description: $pattern"
}

train_cmd() {
  local hypes="$1"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --use_env \
    opencood/tools/train.py \
    --hypes_yaml "$hypes"
}

parse_checkpoint_dir() {
  local log_path="$1"
  local parsed
  parsed="$(grep -oE 'Training Finished, checkpoints saved to .+' "$log_path" | tail -n 1 | sed 's/^Training Finished, checkpoints saved to //')"
  if [[ -n "$parsed" ]]; then
    printf '%s\n' "$parsed"
    return 0
  fi

  local latest=""
  if [[ -d "$REPO_ROOT/opencood/logs" ]]; then
    latest="$(find "$REPO_ROOT/opencood/logs" -maxdepth 1 -type d -name '*point_pillar*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
  fi
  [[ -n "$latest" ]] || fail_shutdown "could not parse checkpoint directory from $log_path"
  printf '%s\n' "$latest"
}

verify_checkpoint_dir() {
  local path="$1"
  local label="$2"
  require_dir "$path" "$label"
  shopt -s nullglob
  local checkpoints=( "$path"/net_epoch*.pth )
  shopt -u nullglob
  if [[ "${#checkpoints[@]}" -eq 0 && ! -f "$path/latest.pth" ]]; then
    fail_shutdown "$label has no checkpoint file: $path"
  fi
}

prepare_dota_hypes() {
  local src="$REPO_ROOT/$DOTA_HYPES"
  local dst_dir="$RUN_DIR/generated_hypes"
  local dst="$dst_dir/point_pillar_intermediate_fusion_dota.pipeline.yaml"
  local score_dir="$MBE_OUTPUT_DIR/score"
  mkdir -p "$dst_dir"
  "$PYTHON_BIN" -c '
import sys, yaml
src, dst, score_dir = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["iterative_training"] = True
data["pseudo_lable_path"] = score_dir
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
' "$src" "$dst" "$score_dir"
  printf '%s\n' "$dst"
}

log_summary "Pipeline created. Shutdown mode: $([[ "$DRY_RUN" -eq 1 || "$NO_SYSTEM_SHUTDOWN" -eq 1 ]] && echo 'disabled' || echo "$SHUTDOWN_COMMAND")"

write_step_log 0 preflight \
  "repo_root: $REPO_ROOT" \
  "run_dir: $RUN_DIR" \
  "python: $PYTHON_BIN" \
  "cuda_devices: $CUDA_DEVICES" \
  "nproc_per_node: $NPROC_PER_NODE" \
  "initial_hypes: $REPO_ROOT/$INITIAL_HYPES" \
  "dota_hypes: $REPO_ROOT/$DOTA_HYPES" \
  "mbe_output_dir: $MBE_OUTPUT_DIR" \
  "pseudo_label_root: $PSEUDO_LABEL_ROOT"

require_file "$REPO_ROOT/$INITIAL_HYPES" "initial hypes yaml"
require_file "$REPO_ROOT/$DOTA_HYPES" "DOTA hypes yaml"
require_file "$REPO_ROOT/opencood/tools/train.py" "train.py"
require_file "$REPO_ROOT/opencood/tools/inference.py" "inference.py"
require_file "$REPO_ROOT/opencood/tools/MBE.py" "MBE.py"
require_file "$REPO_ROOT/opencood/tools/box_score_for_mbe.py" "box_score_for_mbe.py"

if [[ -z "$INITIAL_DETECTOR_DIR" ]]; then
  run_step 1 train_initial_detector \
    env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    "$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --use_env \
    opencood/tools/train.py \
    --hypes_yaml "$INITIAL_HYPES"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    INITIAL_DETECTOR_DIR="\$INITIAL_DETECTOR_CHECKPOINT_FOLDER"
  else
    INITIAL_DETECTOR_DIR="$(parse_checkpoint_dir "$(step_log_path 1 train_initial_detector)")"
  fi
else
  write_step_log 1 train_initial_detector_skipped "using existing initial_detector_dir: $INITIAL_DETECTOR_DIR"
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  verify_checkpoint_dir "$INITIAL_DETECTOR_DIR" "initial detector checkpoint directory"
fi

run_step 2 generate_initial_pseudo_labels \
  "$PYTHON_BIN" opencood/tools/inference.py \
  --model_dir "$INITIAL_DETECTOR_DIR" \
  --fusion_method "$FUSION_METHOD" \
  --pseudo_lable_save 0

if [[ "$DRY_RUN" -eq 0 ]]; then
  require_glob "$PSEUDO_LABEL_ROOT/pre_box_test_full/pre_*.npy" "initial pseudo-label boxes"
  require_glob "$PSEUDO_LABEL_ROOT/pre_score_test_full/score_*.npy" "initial pseudo-label scores"
fi

run_step 3 mbe_filter "$PYTHON_BIN" opencood/tools/MBE.py
if [[ "$DRY_RUN" -eq 0 ]]; then
  require_glob "$MBE_OUTPUT_DIR/out_pseduo_labels_v1_*.npy" "MBE accepted pseudo-labels"
  require_glob "$MBE_OUTPUT_DIR/out_pseduo_labels_noise_v1_*.npy" "MBE rejected pseudo-labels"
  require_glob "$MBE_OUTPUT_DIR/multi_agent_point_remove_ground/multi_agent_point*.npy" "MBE point caches"
  require_glob "$MBE_OUTPUT_DIR/multi_agent_point_pose/multi_agent_point_pose*.npy" "MBE pose caches"
fi

run_step 4 score_mbe_boxes "$PYTHON_BIN" opencood/tools/box_score_for_mbe.py
if [[ "$DRY_RUN" -eq 0 ]]; then
  require_glob "$MBE_OUTPUT_DIR/score/out_pseduo_labels_with_score_v4_*.npy" "scored accepted pseudo-labels"
  require_glob "$MBE_OUTPUT_DIR/score/out_pseduo_labels_noise_with_score_v4_*.npy" "scored rejected pseudo-labels"
fi

mkdir -p "$RUN_DIR/generated_hypes"
run_step 5 prepare_dota_hypes \
  "$PYTHON_BIN" -c 'import sys,yaml; src,dst,score=sys.argv[1:4]; data=yaml.safe_load(open(src, encoding="utf-8")); data["iterative_training"]=True; data["pseudo_lable_path"]=score; yaml.safe_dump(data, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)' \
  "$REPO_ROOT/$DOTA_HYPES" \
  "$RUN_DIR/generated_hypes/point_pillar_intermediate_fusion_dota.pipeline.yaml" \
  "$MBE_OUTPUT_DIR/score"
GENERATED_DOTA_HYPES="$RUN_DIR/generated_hypes/point_pillar_intermediate_fusion_dota.pipeline.yaml"
require_file "$GENERATED_DOTA_HYPES" "generated DOTA hypes yaml"

if [[ -z "$FINAL_CHECKPOINT_DIR" ]]; then
  run_step 6 train_with_pseudo_labels \
    env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    "$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --use_env \
    opencood/tools/train.py \
    --hypes_yaml "$GENERATED_DOTA_HYPES"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    FINAL_CHECKPOINT_DIR="\$CHECKPOINT_FOLDER"
  else
    FINAL_CHECKPOINT_DIR="$(parse_checkpoint_dir "$(step_log_path 6 train_with_pseudo_labels)")"
  fi
else
  write_step_log 6 train_with_pseudo_labels_skipped "using existing final_checkpoint_dir: $FINAL_CHECKPOINT_DIR"
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  verify_checkpoint_dir "$FINAL_CHECKPOINT_DIR" "final checkpoint directory"
fi

if [[ "$SKIP_TEST" -eq 1 ]]; then
  write_step_log 7 test_final_model_skipped "skip_test: true"
else
  run_step 7 test_final_model "$PYTHON_BIN" opencood/tools/inference.py \
    --model_dir "$FINAL_CHECKPOINT_DIR" \
    --fusion_method "$FUSION_METHOD"
fi

log_summary "PIPELINE COMPLETE. Logs are in $RUN_DIR"
